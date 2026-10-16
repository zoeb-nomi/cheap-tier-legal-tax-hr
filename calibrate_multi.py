"""Score the four cheap-tier vendors (results/multi/<vendor>.jsonl) against the human labels, side by side with the
Decisions API baseline (results/items.csv) on the six LegalBench tasks.

Writes results/multi/metrics.json, results/multi/summary.csv (one row per vendor x task), results/multi/trust_curves.csv,
and prints a markdown table. No sklearn.

Per vendor x task: n, accuracy at p=0.5, AUROC, ECE, Brier, refusal and error rates, coverage and accuracy when the
stated confidence is 0.9 or more, confident-wrong counts by direction, mean cost per item, cost per correct answer.
"""
import csv, glob, json, os
from collections import defaultdict
import numpy as np

MULTI = "results/multi"
BASE = "results/items.csv"
DECISIONS_PRICE = 0.10  # USD per 1M input tokens, no output tokens
os.makedirs(MULTI, exist_ok=True)


def auroc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float)
    pos, neg = p[y], p[~y]
    if len(pos) == 0 or len(neg) == 0: return None
    order = np.argsort(p); ranks = np.empty(len(p)); sp = p[order]; i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0: continue
        e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(e)


def load_vendor(path):
    latest = {}
    for l in open(path):
        r = json.loads(l); latest[r["id"]] = r
    return list(latest.values())


def load_baseline():
    """Decisions API per-item probabilities as a pseudo-vendor. items.csv has no token counts; cost is estimated
    from metrics.json's total if present, else from the published $0.019 for 544 items."""
    if not os.path.exists(BASE): return []
    rows = []
    with open(BASE) as f:
        for r in csv.DictReader(f):
            p = r["probability"]
            rows.append({"id": r["id"], "task": r["task"], "human_label": r["human_label"] in ("1", "True", "true"),
                         "probability": float(p) if p not in ("", "None") else None,
                         "status": "ok" if p not in ("", "None") else "refusal",
                         "usd": 0.019 / 544, "vendor": "decisions", "model": "gpt-6-luna (Decisions API)"})
    return rows


def score(rs):
    ok = [r for r in rs if r.get("status") == "ok" and r.get("probability") is not None]
    n = len(rs)
    out = {"n": n, "n_scored": len(ok),
           "refusal_rate": sum(r.get("status") == "refusal" for r in rs) / n if n else None,
           "error_rate": sum(r.get("status") in ("error", "no_prob") for r in rs) / n if n else None,
           "usd_total": float(sum(r.get("usd", 0.0) or 0.0 for r in rs)),
           "in_tokens": int(sum(r.get("input_tokens", 0) or 0 for r in rs)),
           "out_tokens": int(sum(r.get("output_tokens", 0) or 0 for r in rs))}
    if not ok:
        out.update(accuracy=None, auroc=None, ece=None, brier=None, usd_per_correct=None); return out
    y = np.array([r["human_label"] for r in ok], bool); p = np.array([r["probability"] for r in ok], float)
    pred = p >= 0.5; correct = pred == y; conf = np.maximum(p, 1 - p)
    hi = conf >= 0.9
    out.update(
        accuracy=float(correct.mean()), auroc=auroc(y, p), ece=ece(y, p), brier=float(((p - y) ** 2).mean()),
        # accuracy over everything the model was asked, counting refusals and errors as not correct
        accuracy_all_asked=float(correct.sum() / n),
        coverage_at_90=float(hi.mean()), accuracy_at_90=float(correct[hi].mean()) if hi.any() else None,
        confident_wrong=int((hi & ~correct).sum()),
        confident_no_but_yes=int((hi & ~pred & y).sum()), confident_yes_but_no=int((hi & pred & ~y).sum()),
        usd_per_item=out["usd_total"] / n, usd_per_correct=out["usd_total"] / max(1, int(correct.sum())),
        usd_per_1000_items=out["usd_total"] / n * 1000)
    return out


def trust_curve(rs):
    ok = [r for r in rs if r.get("status") == "ok" and r.get("probability") is not None]
    if not ok: return []
    y = np.array([r["human_label"] for r in ok], bool); p = np.array([r["probability"] for r in ok], float)
    conf = np.maximum(p, 1 - p); correct = (p >= 0.5) == y
    rows = []
    for t in np.arange(0.5, 1.0001, 0.05):
        m = conf >= t - 1e-9
        rows.append({"threshold": round(float(t), 2), "coverage": float(m.mean()),
                     "accuracy": float(correct[m].mean()) if m.any() else None, "n": int(m.sum())})
    return rows


def main():
    vendors = {}
    for path in sorted(glob.glob(f"{MULTI}/*.jsonl")):
        name = os.path.basename(path)[:-6]
        vendors[name] = load_vendor(path)
    base = load_baseline()
    if base: vendors["decisions"] = base

    metrics, summary, curves = {}, [], []
    for v, rs in vendors.items():
        by_task = defaultdict(list)
        for r in rs: by_task[r["task"]].append(r)
        groups = {"ALL_legalbench": [r for r in rs if not r["task"].startswith("alinia_")],
                  "ALL_alinia": [r for r in rs if r["task"].startswith("alinia_")]}
        groups.update(by_task)
        metrics[v] = {}
        for task, trs in groups.items():
            if not trs: continue
            s = score(trs); metrics[v][task] = s
            summary.append({"vendor": v, "task": task, **{k: s.get(k) for k in (
                "n", "n_scored", "accuracy", "accuracy_all_asked", "auroc", "ece", "brier", "refusal_rate", "error_rate",
                "coverage_at_90", "accuracy_at_90", "confident_wrong", "confident_no_but_yes", "confident_yes_but_no",
                "usd_per_1000_items", "usd_per_correct")}})
            for c in trust_curve(trs): curves.append({"vendor": v, "task": task, **c})

    json.dump(metrics, open(f"{MULTI}/metrics.json", "w"), indent=2)
    if summary:
        with open(f"{MULTI}/summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    if curves:
        with open(f"{MULTI}/trust_curves.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curves[0].keys())); w.writeheader(); w.writerows(curves)

    # console table: one block per task, vendors as rows
    tasks = sorted({s["task"] for s in summary}, key=lambda t: (not t.startswith("ALL"), t))
    fmt = lambda x, d=3: "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    for t in tasks:
        print(f"\n### {t}")
        print("| vendor | n | acc | acc(all asked) | AUROC | ECE | refuse | cov@.9 | acc@.9 | conf-wrong (no>yes / yes>no) | $/1k items | $/correct |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in summary:
            if s["task"] != t: continue
            print(f"| {s['vendor']} | {s['n']} | {fmt(s['accuracy'])} | {fmt(s['accuracy_all_asked'])} | {fmt(s['auroc'])} | "
                  f"{fmt(s['ece'])} | {fmt(s['refusal_rate'])} | {fmt(s['coverage_at_90'],2)} | {fmt(s['accuracy_at_90'])} | "
                  f"{s['confident_wrong']} ({s['confident_no_but_yes']}/{s['confident_yes_but_no']}) | "
                  f"{fmt(s['usd_per_1000_items'],4)} | {fmt(s['usd_per_correct'],6)} |")


if __name__ == "__main__":
    main()
