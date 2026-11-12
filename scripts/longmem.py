#!/usr/bin/env python3
"""Long-history test on LongMemEval-S (Wu et al. 2024): each question has its own history of about
50 chats between one user and an assistant (~105k tokens). Is a kept notebook better than pasting
the whole history?

Conditions: none, notebook (rewritten after every BATCH chats, capped words), full (all chats pasted).
Stratified sample: N questions per question type, fixed seed.
Usage: longmem.py --per-type 8 --model haiku --conds none,notebook,full --tag lme-haiku
"""
import argparse, concurrent.futures as cf, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import call, extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAP, BATCH = 2000, 5

NOTEBOOK_PROMPT = """You are an AI assistant that keeps a notebook about your user. You will not see these chats again: in future chats you will only have this notebook. Rewrite the notebook so it holds everything about the user worth remembering from all chats so far.

Rules:
- Facts about the user: life, people, work, possessions, plans, preferences, numbers, places, events.
- What you (the assistant) recommended or told them, when it was specific (names, lists, numbers).
- Every event gets an absolute date (resolve relative dates against the chat date). When a fact changes, keep the newest and note the old one with its date.
- Ignore generic chit-chat and tasks that say nothing about the user.
- Keep the whole notebook under {cap} words: merge, condense. Short lines, plain markdown, no commentary.

Current notebook:
<notebook>
{notebook}
</notebook>

New chats:
{chats}

Reply with the full updated notebook inside <notebook></notebook>."""

ANSWER_PROMPT = """{context}

Today is {qdate}. The user asks: {question}

Answer briefly and directly. If the information was never mentioned, say you don't know. Put the answer inside <answer></answer>."""

JUDGE = {
    "default": "I will give you a question, a correct answer, and a response from a model. Answer yes if the response contains the correct answer, or is equivalent to it, or contains all the intermediate steps to get the correct answer. Answer no if the response only contains a subset of the information required.",
    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Answer yes if the response contains the correct answer or is equivalent to it. Do not penalize off-by-one errors in the number of days. Answer no if the response only contains a subset of the information required.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Answer yes if the response contains the correct answer. If the response contains some previous information along with an updated answer, it counts as correct as long as the updated answer is the required answer.",
    "single-session-preference": "I will give you a question, a rubric for the desired personalized response, and a response from a model. Answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric; it is correct as long as it recalls and uses the user's personal information correctly.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Answer yes if the model correctly identifies the question as unanswerable (says the information is incomplete or not mentioned). Otherwise answer no.",
}


def render(session, date):
    return f"Chat on {date}:\n" + "\n".join(f"{t['role']}: {t['content']}" for t in session)


def sample(data, per_type, seed=7):
    random.seed(seed)
    by = {}
    for q in data:
        by.setdefault(q["question_type"], []).append(q)
    out = []
    for t in sorted(by):
        out += random.sample(by[t], min(per_type, len(by[t])))
    return out


def notebook_for(q, model, outdir):
    path = outdir / f"{q['question_id']}_notebook.json"
    if path.exists():
        return json.load(open(path))["memory"]
    pairs = sorted(zip(q["haystack_dates"], q["haystack_sessions"]), key=lambda p: p[0])
    mem = ""
    for i in range(0, len(pairs), BATCH):
        chats = "\n\n".join(render(s, d) for d, s in pairs[i:i + BATCH])
        r, _ = call(NOTEBOOK_PROMPT.format(cap=CAP, notebook=mem or "(empty)", chats=chats), model)
        mem = extract(r, "notebook")
    json.dump({"memory": mem, "words": len(mem.split())}, open(path, "w"), indent=1)
    return mem


def run_one(q, cond, model, judge, outdir):
    path = outdir / f"{q['question_id']}_{cond}.json"
    if path.exists():
        return json.load(open(path))
    if cond == "none":
        ctx = "This is a new chat. You have no memory of earlier chats with this user."
    elif cond == "notebook":
        ctx = f"This is a new chat. Your notebook about this user from earlier chats:\n<notebook>\n{notebook_for(q, model, outdir)}\n</notebook>"
    else:
        pairs = sorted(zip(q["haystack_dates"], q["haystack_sessions"]), key=lambda p: p[0])
        ctx = "Every earlier chat with this user:\n<chats>\n" + "\n\n".join(render(s, d) for d, s in pairs) + "\n</chats>"
    r, tok = call(ANSWER_PROMPT.format(context=ctx, qdate=q["question_date"], question=q["question"]), model,
                  system="You are a helpful assistant.")
    ans = extract(r, "answer")
    kind = "abstention" if q["question_id"].endswith("_abs") else q["question_type"]
    g, _ = call(f"{JUDGE.get(kind, JUDGE['default'])}\n\nQuestion: {q['question']}\n\nCorrect answer: {q['answer']}\n\n"
                f"Model response: {ans}\n\nIs the model response correct? Answer yes or no only.", judge)
    row = {"id": q["question_id"], "type": q["question_type"], "abs": kind == "abstention", "q": q["question"],
           "gold": str(q["answer"]), "ans": ans, "correct": g.strip().lower().startswith("yes"), "ctx_tokens": tok}
    json.dump(row, open(path, "w"), indent=1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--judge", default="sonnet")
    ap.add_argument("--conds", default="none,notebook,full")
    ap.add_argument("--tag", default="lme-haiku")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    data = json.load(open(ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"))
    qs = sample(data, args.per_type)
    outdir = ROOT / "runs" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(q, c) for q in qs for c in args.conds.split(",")]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_one, q, c, args.model, args.judge, outdir): (q["question_id"], c) for q, c in jobs}
        for f in cf.as_completed(futs):
            qid, c = futs[f]
            try:
                r = f.result()
                print(f"{qid} {c}: {'OK' if r['correct'] else 'x'}", flush=True)
            except Exception as e:
                print(f"{qid} {c}: FAILED {e}", flush=True)


if __name__ == "__main__":
    main()
