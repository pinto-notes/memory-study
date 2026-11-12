#!/usr/bin/env python3
"""Every number in the README, computed from results/. Run: python3 scripts/summarize.py
With --readable (needs data/ from fetch_data.sh) it also writes runs/readable_<run>.csv with each
question, gold answer and model answer side by side."""
import argparse, csv, glob, json, os, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ORDER = ["none", "facts", "notebook", "pinto", "full"]
NAMES = {"none": "No memory", "facts": "Saved facts", "notebook": "One notebook", "pinto": "Pinto", "full": "Full transcripts"}
CATS = ["single-hop", "multi-hop", "temporal", "open-domain"]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def locomo(run):
    by = collections.defaultdict(list)
    for f in glob.glob(str(RES / "locomo" / run / "conv*_*.json")):
        name = os.path.basename(f)[:-5]
        if name.endswith("_memory_size"):
            continue
        cond = name.split("_", 1)[1]
        by[cond] += json.load(open(f))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readable", action="store_true")
    args = ap.parse_args()

    # Pinto's run lives in its own folder; show it next to Haiku run 1.
    for run in ["haiku-run1", "haiku-run2", "haiku-pinto", "gpt-6-luna", "sonnet-contaminated"]:
        by = locomo(run)
        print(f"\n## LoCoMo, {run}")
        print("| Condition | All | " + " | ".join(CATS) + " | n |")
        print("|---|---|" + "---|" * len(CATS) + "---|")
        for c in ORDER:
            rows = by.get(c)
            if not rows:
                continue
            ok = lambda rs: sum(r["grade"] == "CORRECT" for r in rs)
            cells = [pct(ok([r for r in rows if r["cat"] == k]), len([r for r in rows if r["cat"] == k])) for k in CATS]
            print(f"| {NAMES[c]} | {pct(ok(rows), len(rows))} | " + " | ".join(cells) + f" | {len(rows)} |")

    print("\n## Memory size at the end (words), per conversation")
    for run, cond in [("haiku-run1", "notebook"), ("haiku-run1", "facts"), ("haiku-pinto", "pinto"), ("gpt-6-luna", "notebook")]:
        ws = [json.load(open(f))["per_chat"][-1]["memory_words_after"]
              for f in sorted(glob.glob(str(RES / "locomo" / run / f"conv*_{cond}_memory_size.json")))]
        print(f"- {run} {cond}: {ws}")
    tc = json.load(open(RES / "locomo" / "pinto_tool_calls.json"))
    tot = collections.Counter()
    for k, v in tc.items():
        if k.startswith("haiku-pinto"):
            tot.update(v["calls"])
    print(f"- haiku-pinto tool calls, all 10 accounts: {dict(tot)}")

    print("\n## LongMemEval-S, Haiku (48 questions, 8 per type)")
    L = [r for r in json.load(open(RES / "longmemeval" / "haiku" / "results.json")) if r["cond"] != "notebook_size"]
    for c in ["none", "notebook", "full"]:
        rs = [r for r in L if r["cond"] == c]
        avg = sum(r["tokens_in_call"] for r in rs) // max(1, len(rs))
        print(f"- {c}: {pct(sum(r['correct'] for r in rs), len(rs))} of {len(rs)}, ~{avg:,} tokens per question")

    G = json.load(open(ROOT / "grading" / "opus_vs_model_100.json"))
    print(f"\n## Grader check: Opus and the Sonnet grader agree on {sum(g['model_grade'] == g['opus_grade'] for g in G)} of {len(G)}")

    if args.readable:
        data = json.load(open(ROOT / "data" / "locomo10.json"))
        os.makedirs(ROOT / "runs", exist_ok=True)
        for run in ["haiku-run1", "haiku-pinto", "gpt-6-luna"]:
            with open(ROOT / "runs" / f"readable_{run}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["conv", "qi", "condition", "category", "question", "gold", "answer", "grade"])
                for f in sorted(glob.glob(str(RES / "locomo" / run / "conv*_*.json"))):
                    name = os.path.basename(f)[:-5]
                    if name.endswith("_memory_size"):
                        continue
                    ci, cond = int(name[4:name.index("_")]), name.split("_", 1)[1]
                    qa = [q for q in data[ci]["qa"] if q["category"] in (1, 2, 3, 4)]
                    for r in json.load(open(f)):
                        q = qa[r["qi"]]
                        w.writerow([ci, r["qi"], cond, r["cat"], q["question"], q["answer"], r["answer"], r["grade"]])
        print("\nwrote runs/readable_*.csv")


if __name__ == "__main__":
    main()
