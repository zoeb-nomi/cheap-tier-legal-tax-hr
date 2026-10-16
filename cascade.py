"""Per-domain cost and accuracy of single models and two-model systems, from results/multi/*.jsonl.

Domains (task groups):
  legal: cuad_anti-assignment, cuad_non-compete, cuad_change_of_control, hearsay, personal_jurisdiction,
         consumer_contracts_qa, alinia_legal
  tax:   sara_entailment, alinia_tax
  hr:    learned_hands_employment

Systems compared per domain (items = the rows every listed model answered):
  A  <frontier> on everything                       (the "previously" line; needs results/multi/<frontier>.jsonl)
  B  <cheap> on everything, one per cheap model
  C  Haiku gate: Haiku clears items it is >= GATE sure of; the frontier model answers the rest
  D  Dissent check: <cheapest> answers everything; Haiku also answers; where they disagree the frontier decides
  E  Haiku gate to humans: Haiku clears >= GATE; the rest goes to a person (accuracy counted on cleared items only,
     plus the share of the queue a person still reads)

Cost = the per-item USD recorded at run time (input + output tokens at the vendor's list price), scaled to 1,000 items.
Usage: python cascade.py [--frontier opus] [--gate 0.9] [--cheap haiku,luna,flashlite,gemma]
"""
import argparse, glob, json, os
from collections import defaultdict

MULTI = "results/multi"
DOMAINS = {
    "legal": ["cuad_anti-assignment", "cuad_non-compete", "cuad_change_of_control", "hearsay",
              "personal_jurisdiction", "consumer_contracts_qa", "alinia_legal"],
    "tax": ["sara_entailment", "alinia_tax"],
    "hr": ["learned_hands_employment"],
}


def load(v):
    path = f"{MULTI}/{v}.jsonl"
    if not os.path.exists(path): return {}
    latest = {}
    for l in open(path):
        r = json.loads(l); latest[r["id"]] = r
    return {k: r for k, r in latest.items() if r.get("status") == "ok" and r.get("probability") is not None}


def pred(r): return r["probability"] >= 0.5
def conf(r): return max(r["probability"], 1 - r["probability"])
def truth(r): return bool(r["human_label"])


def system_rows(name, acc, usd_per_1k, human_share=0.0, n=0):
    return {"system": name, "accuracy": acc, "usd_per_1k": usd_per_1k, "human_share": human_share, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", default="opus")
    ap.add_argument("--gate", type=float, default=0.9)
    ap.add_argument("--cheap", default="haiku,luna,flashlite,gemma")
    a = ap.parse_args()
    cheap = a.cheap.split(",")
    data = {v: load(v) for v in cheap + [a.frontier]}
    present = [v for v in cheap if data[v]]
    frontier = a.frontier if data[a.frontier] else None
    out = {}

    for dom, tasks in DOMAINS.items():
        ids = None
        for v in present + ([frontier] if frontier else []):
            s = {i for i, r in data[v].items() if r["task"] in tasks}
            ids = s if ids is None else ids & s
        ids = sorted(ids or [])
        if not ids:
            print(f"\n### {dom}: no rows yet"); continue
        n = len(ids)
        rows = []

        def cost(v, subset): return sum(data[v][i].get("usd", 0.0) for i in subset) / n * 1000
        def acc(v, subset): return sum(pred(data[v][i]) == truth(data[v][i]) for i in subset) / len(subset) if subset else None

        if frontier:
            rows.append(system_rows(f"A  {frontier} on everything", acc(frontier, ids), cost(frontier, ids), 0, n))
        for v in present:
            rows.append(system_rows(f"B  {v} on everything", acc(v, ids), cost(v, ids), 0, n))
        if "haiku" in present:
            cleared = [i for i in ids if conf(data["haiku"][i]) >= a.gate]
            rest = [i for i in ids if i not in set(cleared)]
            if frontier:
                correct = sum(pred(data["haiku"][i]) == truth(data["haiku"][i]) for i in cleared) + \
                          sum(pred(data[frontier][i]) == truth(data[frontier][i]) for i in rest)
                rows.append(system_rows(f"C  Haiku gate {a.gate:.0%}, {frontier} on the rest ({len(rest)/n:.0%})",
                                        correct / n, cost("haiku", ids) + cost(frontier, rest), 0, n))
            acc_cleared = acc("haiku", cleared)
            rows.append(system_rows(f"E  Haiku gate {a.gate:.0%}, people read the rest ({len(rest)/n:.0%})",
                                    acc_cleared, cost("haiku", ids), len(rest) / n, n))
            for cheapest in [v for v in present if v != "haiku"]:
                if not frontier: break
                agree = [i for i in ids if pred(data[cheapest][i]) == pred(data["haiku"][i])]
                dis = [i for i in ids if i not in set(agree)]
                correct = sum(pred(data[cheapest][i]) == truth(data[cheapest][i]) for i in agree) + \
                          sum(pred(data[frontier][i]) == truth(data[frontier][i]) for i in dis)
                rows.append(system_rows(f"D  {cheapest} + Haiku, {frontier} decides disagreements ({len(dis)/n:.0%})",
                                        correct / n, cost(cheapest, ids) + cost("haiku", ids) + cost(frontier, dis), 0, n))
        out[dom] = rows
        print(f"\n### {dom}  ({n} items; tasks: {', '.join(t for t in tasks if any(data[v].get(i, {}).get('task') == t for v in present for i in ids[:1]) or True)})")
        print("| system | accuracy | $ per 1,000 items | still read by a person |")
        print("|---|---|---|---|")
        for r in rows:
            acc_s = "-" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
            print(f"| {r['system']} | {acc_s} | {r['usd_per_1k']:.3f} | {r['human_share']:.0%} |")
    json.dump(out, open(f"{MULTI}/cascade.json", "w"), indent=2)
    if not frontier:
        print(f"\n(no results/multi/{a.frontier}.jsonl yet: systems A, C and D need the frontier run)")


if __name__ == "__main__":
    main()
