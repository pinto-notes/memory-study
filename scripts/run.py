#!/usr/bin/env python3
"""Notebook-memory experiment on LoCoMo.

Each LoCoMo conversation has 19-35 sessions spread over months. We treat every
session as a separate chat with the assistant, then ask the benchmark questions
in a NEW chat. Conditions:

  none      new chat, no memory at all
  facts     after each chat the model may save up to 5 one-line facts (like a
            built-in "saved memories" list); the new chat sees the list
  notebook  after each chat the model rewrites a markdown notebook (people,
            dated events, plans); the new chat sees the notebook
  full      the new chat sees every transcript (upper bound; not possible in a
            real new chat)

All model calls go through `claude -p` (subscription), no tools, no MCP.
Usage: run.py --convs 0 --model haiku --conds none,facts,notebook,full
"""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = json.load(open(ROOT / "data" / "locomo10.json"))
EMPTY = os.environ.get("MEMSTUDY_WORKDIR", "/tmp/memstudy-empty")  # an empty folder, so no project files reach the model
os.makedirs(EMPTY, exist_ok=True)
CAP = int(os.environ.get("NOTEBOOK_CAP", "2000"))
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


CODEX_HOME = os.environ.get("MEMSTUDY_CODEX_HOME", "/tmp/memstudy-codex-home")  # a Codex home with only auth.json: no apps, connectors or memories


def call(prompt, model, system="You are a careful assistant. Follow the format exactly.", tries=7, mcp=None, max_turns=None):
    """One model call. model "haiku"/"sonnet" -> claude -p; "codex:<slug>" -> codex exec.
    mcp: a URL of the local Pinto MCP server (claude only); the model may call its tools."""
    for i in range(tries):
        try:
            if model.startswith("codex:"):
                out = f"/tmp/memstudy-codex-{os.getpid()}-{threading.get_ident()}.txt"
                p = subprocess.run(
                    ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
                     "-m", model.split(":", 1)[1], "-c", 'model_reasoning_effort="low"', "-o", out, "-"],
                    input=f"{system}\n\n{prompt}", capture_output=True, text=True, cwd=EMPTY, timeout=900,
                    env={**os.environ, "CODEX_HOME": CODEX_HOME})
                text = open(out).read().strip()
                os.remove(out)
                if not text:
                    raise RuntimeError(p.stderr[-300:])
                m = re.search(r"tokens used\s*\n\s*([\d,]+)", p.stderr)
                return text, int(m.group(1).replace(",", "")) if m else 0
            cmd = ["claude", "-p", "--model", model, "--system-prompt", system, "--tools", "",
                   "--strict-mcp-config", "--setting-sources", "local", "--no-session-persistence",
                   "--output-format", "json"]
            if mcp:
                cmd += ["--mcp-config", json.dumps({"mcpServers": {"pinto": {"type": "http", "url": mcp}}}),
                        "--allowedTools", "mcp__pinto__*"]
            if max_turns:
                cmd += ["--max-turns", str(max_turns)]
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=EMPTY, timeout=1200,
                               env={**os.environ, "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"})
            d = json.loads(p.stdout)
            if d.get("is_error") and not (mcp and d.get("subtype") == "error_max_turns"):
                raise RuntimeError(d.get("result") or d.get("subtype"))
            u = d.get("usage", {})
            return d.get("result") or "", (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                                           + u.get("cache_read_input_tokens", 0))
        except Exception as e:
            print(f"  retry {i}: {str(e)[:200]}", file=sys.stderr)
            time.sleep(20 * (i + 1))
    raise RuntimeError("call failed")


def sessions(conv):
    c = conv["conversation"]
    ids = sorted(int(m.group(1)) for k in c if (m := re.match(r"session_(\d+)$", k)))
    out = []
    for i in ids:
        lines = []
        for t in c[f"session_{i}"]:
            txt = t["text"]
            if t.get("blip_caption"):
                txt += f" [shares a photo: {t['blip_caption']}]"
            lines.append(f"{t['speaker']}: {txt}")
        out.append((c[f"session_{i}_date_time"], "\n".join(lines)))
    return out


def extract(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return (m.group(1) if m else text).strip()


NOTEBOOK_PROMPT = """You keep a notebook for {a} and {b}, two friends who chat with each other. You will not see this chat again: tomorrow you will only have your notebook. Rewrite the notebook so it holds everything worth remembering from all chats so far.

Rules:
- Organise by person, then by topic (family, work, hobbies, plans, feelings, possessions, places).
- Every event gets an absolute date or date range (resolve "yesterday", "last week", "next month" against the chat date).
- Keep names, numbers, titles, places and reasons. Keep earlier notes unless the new chat corrects them.
- Keep the whole notebook under {cap} words: merge, condense and drop small talk. Short lines, not prose.
- Plain markdown, no commentary.

Current notebook:
<notebook>
{notebook}
</notebook>

New chat on {date}:
<chat>
{chat}
</chat>

Reply with the full updated notebook inside <notebook></notebook>."""

FACTS_PROMPT = """You are an assistant with a small memory feature. {a} and {b} just had this chat. You may save up to 5 short facts (one line each, at most 15 words) worth remembering for future chats. You already saved:
<saved>
{facts}
</saved>

Chat on {date}:
<chat>
{chat}
</chat>

Reply with only the new facts, one per line, inside <facts></facts>. Save nothing that is already saved."""

ANSWER_PROMPT = """{context}

Answer each question about {a} and {b} in a few words (a name, date, number or short phrase). Use what you have above; if it is not there, give your best guess. Dates: give the actual date or period, not a relative one.

Questions:
{questions}

Reply with a JSON object mapping question number to answer, inside <answers></answers>."""

JUDGE_PROMPT = """You grade answers to memory questions. For each item, decide if the generated answer is CORRECT or WRONG against the gold answer. Be generous: it is CORRECT if it refers to the same thing, even if longer or phrased differently; dates in a different format or with the same period count. It is WRONG if it is missing, contradicts, or names something else.

{items}

Reply with a JSON object mapping item number to "CORRECT" or "WRONG", inside <grades></grades>."""


MCP_BASE = os.environ.get("PINTO_MCP", "http://127.0.0.1:8791")
PINTO_SYSTEM = "You are a helpful assistant. Pinto Notes, the user's notes app, is connected through its tools."
PINTO_SAVE_PROMPT = """You were the assistant {a} and {b} talked with in the chat below ({date}). Remember this chat in Pinto Notes for next time: save or update whatever is worth knowing in future chats, organised as you think best. You won't see this chat again.

<chat>
{chat}
</chat>

When the notes are saved, reply DONE."""


def mcp_tool(url, name, args):
    req = urllib.request.Request(url, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                 "params": {"name": name, "arguments": args}}).encode(), headers={"content-type": "application/json"})
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=60).read())["result"]["content"][0]["text"])


def pinto_snapshot(url):
    """Every note in the account, as text, to measure what the model kept."""
    notes = {}
    for m in mcp_tool(url, "list", {"pattern": "**", "limit": 1000}).get("matches", []):
        if m.get("type") == "note":
            t = mcp_tool(url, "fetch", {"id": m["path"]})["text"]
            notes[m["path"]] = "\n".join(l.split("\t", 1)[-1] for l in t.split("\n"))
    return notes


def build_memory(ci, conv, cond, model, outdir):
    a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
    path = outdir / f"conv{ci}_{cond}.json"
    if path.exists():
        return json.load(open(path))
    mem, log = "", []
    url = f"{MCP_BASE}/{outdir.name}-conv{ci}"
    for date, chat in sessions(conv):
        if cond == "pinto":
            calls = ROOT / "runs" / "mcp-logs" / f"{outdir.name}-conv{ci}.jsonl"
            for attempt in range(4):
                before = len(open(calls).readlines()) if calls.exists() else 0
                r, tok = call(PINTO_SAVE_PROMPT.format(a=a, b=b, date=date, chat=chat), model, system=PINTO_SYSTEM, mcp=url, max_turns=30)
                if (len(open(calls).readlines()) if calls.exists() else 0) > before:
                    break
                print(f"  conv{ci} pinto {date}: no tool calls, retrying", file=sys.stderr)
            else:
                raise RuntimeError("the model never reached Pinto")
            mem = "\n\n".join(f"## {k}\n{v}" for k, v in pinto_snapshot(url).items())
        elif cond == "notebook":
            r, tok = call(NOTEBOOK_PROMPT.format(cap=CAP, a=a, b=b, notebook=mem or "(empty)", date=date, chat=chat), model)
            mem = extract(r, "notebook")
        else:
            r, tok = call(FACTS_PROMPT.format(a=a, b=b, facts=mem or "(none)", date=date, chat=chat), model)
            new = [l.strip("-• ").strip() for l in extract(r, "facts").splitlines() if l.strip()][:5]
            mem = "\n".join([*filter(None, mem.split("\n")), *new])
        log.append({"date": date, "tokens_in": tok, "mem_words": len(mem.split())})
        print(f"  conv{ci} {cond} {date}: {len(mem.split())} words", file=sys.stderr)
    res = {"memory": mem, "log": log}
    json.dump(res, open(path, "w"), indent=1)
    return res


def context_for(conv, cond, mem):
    a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
    if cond == "none":
        return f"This is a new chat. You have no memory of earlier chats with {a} and {b}."
    if cond == "facts":
        return f"This is a new chat. Your saved memories:\n<saved>\n{mem}\n</saved>"
    if cond == "pinto":
        return (f"This is a new chat. Pinto Notes is connected; your notes from earlier chats with {a} and {b} are there. "
                "Look things up in Pinto Notes before answering. Don't change any notes.")
    if cond == "notebook":
        return f"This is a new chat. Your notebook from earlier chats:\n<notebook>\n{mem}\n</notebook>"
    full = "\n\n".join(f"Chat on {d}:\n{c}" for d, c in sessions(conv))
    return f"Full transcripts of every earlier chat:\n<chats>\n{full}\n</chats>"


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def answer_and_judge(ci, conv, cond, mem, model, judge, outdir, limit):
    path = outdir / f"conv{ci}_{cond}_answers.json"
    if path.exists():
        return json.load(open(path))
    a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
    qa = [q for q in conv["qa"] if q["category"] in CATS][:limit]
    ctx = context_for(conv, cond, mem)
    url = f"{MCP_BASE}/{outdir.name}-conv{ci}"
    if cond == "pinto" and not pinto_snapshot(url):
        raise RuntimeError("Pinto account is empty (server restarted?): delete the build file and rerun")
    rows = []
    for chunk in batched(qa, 10 if cond == "pinto" else 20):
        qs = "\n".join(f"{i + 1}. {q['question']}" for i, q in enumerate(chunk))
        if cond == "pinto":
            r, tok = call(ANSWER_PROMPT.format(context=ctx, a=a, b=b, questions=qs), model, system=PINTO_SYSTEM, mcp=url, max_turns=40)
        else:
            r, tok = call(ANSWER_PROMPT.format(context=ctx, a=a, b=b, questions=qs), model)
        try:
            ans = json.loads(extract(r, "answers"))
        except Exception:
            ans = {}
        items = "\n".join(
            f"{i + 1}. Question: {q['question']}\n   Gold: {q['answer']}\n   Generated: {ans.get(str(i + 1), '')}"
            for i, q in enumerate(chunk))
        g, _ = call(JUDGE_PROMPT.format(items=items), judge)
        try:
            grades = json.loads(extract(g, "grades"))
        except Exception:
            grades = {}
        for i, q in enumerate(chunk):
            rows.append({"q": q["question"], "gold": str(q["answer"]), "cat": CATS[q["category"]],
                         "ans": str(ans.get(str(i + 1), "")), "grade": grades.get(str(i + 1), "WRONG"),
                         "ctx_tokens": tok})
    json.dump(rows, open(path, "w"), indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", default="0")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--judge", default="sonnet")
    ap.add_argument("--conds", default="none,facts,notebook,full")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    convs = [int(x) for x in args.convs.split(",")] if args.convs != "all" else list(range(10))
    conds = args.conds.split(",")
    outdir = ROOT / "runs" / (args.tag or args.model)
    outdir.mkdir(parents=True, exist_ok=True)

    def job(ci, cond):
        conv = DATA[ci]
        mem = build_memory(ci, conv, cond, args.model, outdir)["memory"] if cond in ("facts", "notebook", "pinto") else ""
        return answer_and_judge(ci, conv, cond, mem, args.model, args.judge, outdir, args.limit)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(job, ci, c): (ci, c) for ci in convs for c in conds}
        for f in cf.as_completed(futs):
            ci, c = futs[f]
            try:
                rows = f.result()
                acc = sum(r["grade"] == "CORRECT" for r in rows) / max(1, len(rows))
                print(f"conv{ci} {c}: {acc:.1%} of {len(rows)}", flush=True)
            except Exception as e:
                print(f"conv{ci} {c}: FAILED {e}")


if __name__ == "__main__":
    main()
