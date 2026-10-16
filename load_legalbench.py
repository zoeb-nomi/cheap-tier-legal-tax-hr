"""Build a balanced LegalBench sample (nguha/legalbench, test split, 6 tasks).
Writes data/sample.jsonl and data/prompts.json. Seed 42, 100 rows/task, balanced Yes/No."""
import csv, json, os, random, io
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N, SEED = 100, 42

# Question wording = instruction line of each task's base_prompt.txt (HazyResearch/legalbench)
PROMPTS = {
    "cuad_anti-assignment": "Does the clause require consent or notice of a party if the contract is assigned to a third party?",
    "cuad_non-compete": "Does the clause restrict the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?",
    "cuad_change_of_control": "Does the clause give one party the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?",
    # LegalBench's own one-liner ("Hearsay is an out-of-court statement introduced to prove the truth of the
    # matter asserted. Is there hearsay?") inverted the task in the pilot (model said yes on 7/7 "no" items),
    # so the run uses a fuller statement of the rule. Both wordings are kept in data/prompts.json.
    "hearsay": ("Hearsay is an out-of-court statement introduced to prove the truth of the matter asserted. "
                "A statement made in court during the current trial or hearing is not hearsay. "
                "Non-verbal conduct counts as a statement only if it was intended as an assertion. "
                "A statement offered for a purpose other than its truth, such as to show its effect on the listener, "
                "the speaker's state of mind, or simply that the statement was made, is not hearsay. "
                "Applying these rules to the facts, is the evidence described hearsay?"),
    "personal_jurisdiction": "There is personal jurisdiction over a defendant in the state where the defendant is domiciled, or when (1) the defendant has sufficient contacts with the state, such that they have availed itself of the privileges of the state and (2) the claim arises out of the nexus of the defendant's contacts with the state. Is there personal jurisdiction?",
    "consumer_contracts_qa": "per-row question",
}

def load_rows(task):
    p = hf_hub_download("nguha/legalbench", f"data/{task}/test.tsv", repo_type="dataset")
    import pandas as pd  # same parsing as the HF loader (handles quoted multi-line fields)
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False).to_dict("records")

def main():
    os.makedirs(OUT, exist_ok=True)
    out, dropped = [], {}
    for task, q in PROMPTS.items():
        rows = load_rows(task)
        keep, drop = [], 0
        for r in rows:
            lab = (r.get("answer") or "").strip().lower()
            if lab not in ("yes", "no"):
                drop += 1
                continue
            if task == "consumer_contracts_qa":
                text, question = r["contract"], r["question"]
            else:
                text, question = r["text"], q
            keep.append(dict(id=f"{task}-{r['index']}", task=task, text=text,
                             question=question, human_label=(lab == "yes")))
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
    with open(os.path.join(OUT, "sample.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(PROMPTS, f, indent=2, ensure_ascii=False)
    print("total rows", len(out), "dropped", dropped)

if __name__ == "__main__":
    main()
