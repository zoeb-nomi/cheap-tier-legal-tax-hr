"""Where the cheap models disagree, and who is right. Reads results/multi/*.jsonl (latest record per id).

Prints, for the vendors given on the command line (default: haiku luna flashlite gemma):
  - pairwise agreement (same yes/no) and accuracy of each model on the items where that pair disagrees
  - majority-vote accuracy vs the best single model, per dataset
  - "second opinion" queue: items where model A is 90%+ sure and model B disagrees; how often A is wrong there
  - items every model got wrong (candidates for a label audit), by task
Writes results/multi/agreement.json.
"""
import glob, json, os, sys
from collections import defaultdict
from itertools import combinations

MULTI = "results/multi"
vendors = sys.argv[1:] or ["haiku", "luna", "flashlite", "gemma"]


def load(v):
    latest = {}
    for l in open(f"{MULTI}/{v}.jsonl"):
        r = json.loads(l); latest[r["id"]] = r
    return {k: r for k, r in latest.items() if r.get("status") == "ok" and r.get("probability") is not None}


data = {v: load(v) for v in vendors if os.path.exists(f"{MULTI}/{v}.jsonl")}
vendors = list(data)
common = set.intersection(*(set(d) for d in data.values()))
ids = sorted(common)
print(f"{len(ids)} items answered by all of {vendors}")

def label(r): return r["probability"] >= 0.5
def truth(r): return bool(r["human_label"])
def dataset(i): return "alinia" if i.startswith("alinia_") else "legalbench"

out = {"n_common": len(ids), "vendors": vendors, "pairs": {}, "majority": {}, "second_opinion": {}, "all_wrong": {}}

print("\n## Pairwise: agreement rate, and each model's accuracy on the items where the pair disagrees")
for a, b in combinations(vendors, 2):
    agree = [i for i in ids if label(data[a][i]) == label(data[b][i])]
    dis = [i for i in ids if i not in set(agree)]
    acc_a = sum(label(data[a][i]) == truth(data[a][i]) for i in dis) / len(dis) if dis else None
    acc_b = sum(label(data[b][i]) == truth(data[b][i]) for i in dis) / len(dis) if dis else None
    acc_agree = sum(label(data[a][i]) == truth(data[a][i]) for i in agree) / len(agree) if agree else None
    out["pairs"][f"{a}|{b}"] = dict(agree=len(agree), disagree=len(dis), acc_when_agree=acc_agree, acc_a_on_disagree=acc_a, acc_b_on_disagree=acc_b)
    print(f"{a} vs {b}: agree on {len(agree)}/{len(ids)} (accuracy when they agree {acc_agree:.3f}); "
          f"disagree on {len(dis)}: {a} right {acc_a:.2f}, {b} right {acc_b:.2f}" if dis else f"{a} vs {b}: full agreement")

print("\n## Majority vote vs best single model")
for ds in ("legalbench", "alinia"):
    sub = [i for i in ids if dataset(i) == ds]
    if not sub: continue
    single = {v: sum(label(data[v][i]) == truth(data[v][i]) for i in sub) / len(sub) for v in vendors}
    maj = 0
    for i in sub:
        votes = sum(label(data[v][i]) for v in vendors)
        pred = votes * 2 > len(vendors) if len(vendors) % 2 else (votes * 2 > len(vendors) or (votes * 2 == len(vendors) and label(data[vendors[0]][i])))
        maj += pred == truth(data[vendors[0]][i])
    out["majority"][ds] = dict(n=len(sub), single=single, majority=maj / len(sub))
    best = max(single, key=single.get)
    print(f"{ds}: majority {maj/len(sub):.3f} vs best single {best} {single[best]:.3f}; all: " + ", ".join(f"{v} {s:.3f}" for v, s in single.items()))

print("\n## Second opinion: A says 90%+ sure, B disagrees -> how often A is wrong (and the size of that queue)")
for a in vendors:
    for b in vendors:
        if a == b: continue
        q = [i for i in ids if max(data[a][i]["probability"], 1 - data[a][i]["probability"]) >= 0.9 and label(data[a][i]) != label(data[b][i])]
        wrong = sum(label(data[a][i]) != truth(data[a][i]) for i in q)
        out["second_opinion"][f"{a}<-{b}"] = dict(queue=len(q), a_wrong=wrong)
        if q: print(f"{a} sure, {b} disagrees: {len(q)} items, {a} wrong on {wrong} ({wrong/len(q):.0%})")

print("\n## Items every model got wrong (label-audit candidates), by task")
by_task = defaultdict(list)
for i in ids:
    if all(label(data[v][i]) != truth(data[v][i]) for v in vendors):
        by_task[data[vendors[0]][i]["task"]].append(i)
for t, lst in sorted(by_task.items()):
    out["all_wrong"][t] = lst
    print(f"{t}: {len(lst)}  e.g. {', '.join(lst[:3])}")

json.dump(out, open(f"{MULTI}/agreement.json", "w"), indent=2)
