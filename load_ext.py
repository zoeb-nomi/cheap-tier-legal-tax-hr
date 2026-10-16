"""Build a balanced extension sample from two more LegalBench tasks (nguha/legalbench, test split).
Writes data/sample_ext.jsonl and data/prompts_ext.json. Seed 42, up to 100 rows/task, balanced True/False.

  sara_entailment          US tax statutes: does the statute, applied to the facts, entail the conclusion?
                           (Entailment -> True, Contradiction -> False). Per-row conclusion = the `question` column.
  learned_hands_employment does the post raise an employment-law issue? (Yes -> True, No -> False)
"""
import json, os, random
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N, SEED = 100, 42

PROMPTS = {
    "sara_entailment": ("Does the statute, applied to these facts, entail the following conclusion: <conclusion>? "
                        "(the conclusion is filled in per row from the task's own statement column)"),
    # instruction line of the task's base_prompt.txt (HazyResearch/legalbench)
    "learned_hands_employment": ("Does the post discuss issues related to working at a job, including discrimination and "
                                 "harassment, worker's compensation, workers rights, unions, getting paid, pensions, "
                                 "being fired, and more?"),
}


def load_rows(task):
    p = hf_hub_download("nguha/legalbench", f"data/{task}/test.tsv", repo_type="dataset")
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False).to_dict("records")


def main():
    os.makedirs(OUT, exist_ok=True)
    out, dropped = [], {}
    for task, q in PROMPTS.items():
        rows = load_rows(task)
        keep, drop = [], 0
        for r in rows:
            lab = (r.get("answer") or "").strip().lower()
            if task == "sara_entailment":
                if lab not in ("entailment", "contradiction"):
                    drop += 1; continue
                text = f"STATUTE:\n{r['statute'].strip()}\n\nFACTS:\n{r['description'].strip()}"
                question = ("Does the statute, applied to these facts, entail the following conclusion: "
                            f"{r['question'].strip()}?")
                human = lab == "entailment"
            else:
                if lab not in ("yes", "no"):
                    drop += 1; continue
                text, question, human = r["text"], q, lab == "yes"
            keep.append(dict(id=f"{task}-{r['index']}", task=task, text=text, question=question, human_label=human))
        dropped[task] = drop
        rng = random.Random(SEED)
        yes = [r for r in keep if r["human_label"]]; no = [r for r in keep if not r["human_label"]]
        rng.shuffle(yes); rng.shuffle(no)
        if len(keep) <= N:
            sel = keep
        else:
            h = N // 2
            ny = min(h, len(yes)); nn = min(h, len(no))
            if ny < h: nn = min(len(no), N - ny)
            if nn < h: ny = min(len(yes), N - nn)
            sel = yes[:ny] + no[:nn]
        rng.shuffle(sel)
        out += sel
        print(task, "total", len(rows), "kept", len(sel),
              "true", sum(r["human_label"] for r in sel), "false", sum(not r["human_label"] for r in sel),
              "dropped", drop)
    with open(os.path.join(OUT, "sample_ext.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "prompts_ext.json"), "w", encoding="utf-8") as f:
        json.dump(PROMPTS, f, indent=2, ensure_ascii=False)
    print("total rows", len(out), "dropped", dropped)

if __name__ == "__main__":
    main()
