"""Four cheap-tier models, one yes/no question per row, a stated confidence per answer.

Vendors: haiku (Claude Haiku 5.5), luna (GPT-6 Luna), flashlite (Gemini 3.1 Flash-Lite), gemma (Gemma 4 26B-A4B, free via the Gemini API).
Note: Google retired gemini-2.5-flash-lite ($0.10) for new users on or before 2026-10-15; 3.1 Flash-Lite ($0.25) is the cheapest current Gemini.
Datasets: legalbench (data/sample.jsonl, 544 lawyer-labelled items), alinia (data/alinia_sample.jsonl).
Repository: cheap-tier-legal-tax-hr; the OpenAI Decisions API baseline mentioned in calibrate_multi.py comes from the earlier decisions-api-legal-quiz repo and is not part of this study.
Output: results/multi/<vendor>.jsonl, one record per row, same shape as results/decisions.jsonl:
  id, task, human_label, probability (= p(yes)), status (ok | refusal | no_prob | error), input_tokens, output_tokens, usd.

Usage:
  source ~/Documents/HQ/01_Holding/machine/load-keys.sh      # keys from the Keychain, never from a file here
  python judge_multi.py --vendor haiku --probe                # one call, print the parsed answer
  python judge_multi.py --vendor all --dataset all --limit-per-task 10   # pilot, then STOP and look
  python judge_multi.py --vendor all --dataset all            # everything not yet done (resumes)
"""
import argparse, json, os, re, sys, time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM = ("You answer one yes/no question about a piece of text. Reply with JSON only, no prose, exactly of the form "
          '{"answer": "yes" or "no", "confidence": N} where N is an integer from 50 to 100 giving the probability, '
          "in percent, that your answer is correct.")

# USD per 1M tokens (input, output). Haiku 5.5 and GPT-6 Luna prices are the vendors' published rates for prompts
# under 100k tokens; Gemini 3.1 Flash-Lite and Gemma 4 ("free of charge") from ai.google.dev/pricing (2026-10-15).
VENDORS = {
    "haiku":     dict(model="claude-haiku-5-5",      price=(0.10, 0.50), env="ANTHROPIC_API_KEY", concurrency=6, effort="low"),
    # fairness row: Haiku at its default effort (medium), since GPT-6 Luna reasons by default
    "haiku_med": dict(model="claude-haiku-5-5",      price=(0.10, 0.50), env="ANTHROPIC_API_KEY", concurrency=6, effort="medium"),
    "luna":      dict(model="gpt-6-luna",            price=(0.10, 0.50), env="OPENAI_API_KEY",    concurrency=6),
    "flashlite": dict(model="gemini-3.1-flash-lite", price=(0.25, 1.50), env="GEMINI_API_KEY",    concurrency=3),
    # Gemma 4 runs locally through Ollama (thinking off). The Gemini API also serves gemma-4-26b-a4b-it free of
    # charge, but there it spends a fixed ~2,000 thinking tokens per answer (not switchable), takes ~43 s a call and
    # returned 500s on most of a 10-per-task pilot on 2026-10-15, so the hosted route was dropped.
    "gemma":     dict(model="gemma4:26b",            price=(0.00, 0.00), env=None, local=True,    concurrency=2),
    "qwen":      dict(model="qwen3:30b-a3b",         price=(0.00, 0.00), env=None, local=True,    concurrency=2),
    # frontier reference for the two-model systems: Opus 5.5 at its default effort (medium), thinking always on
    "opus":      dict(model="claude-opus-5-5",       price=(4.00, 20.00), env="ANTHROPIC_API_KEY", concurrency=4, effort="medium"),
}
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DATASETS = {"legalbench": "data/sample.jsonl", "alinia": "data/alinia_sample.jsonl", "ext": "data/sample_ext.jsonl"}
lock = threading.Lock()


# ---------- vendor calls: each returns (text, input_tokens, output_tokens, refused: bool) ----------

def call_haiku(client, cfg, text, question):
    import anthropic
    r = client.messages.create(
        model=cfg["model"], max_tokens=4000, system=SYSTEM,
        output_config={"effort": cfg.get("effort", "low")},
        messages=[{"role": "user", "content": f"{text}\n\nQUESTION: {question}"}])
    out = "".join(b.text for b in r.content if b.type == "text")
    return out, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason == "refusal"


def call_luna(client, cfg, text, question):
    """GPT-6 Luna reasons by default (~150 reasoning tokens on hearsay fact patterns) and those count against
    max_output_tokens, so the cap is generous; output_tokens (and cost) include the reasoning tokens."""
    r = client.responses.create(
        model=cfg["model"], instructions=SYSTEM, max_output_tokens=2000,
        input=f"{text}\n\nQUESTION: {question}")
    refused = False
    for o in (r.output or []):
        if getattr(o, "type", "") == "refusal": refused = True
        for p in (getattr(o, "content", None) or []):
            if getattr(p, "type", "") == "refusal": refused = True
    return r.output_text or "", r.usage.input_tokens, r.usage.output_tokens, refused


def call_flashlite(client, cfg, text, question):
    from google.genai import types
    r = client.models.generate_content(
        model=cfg["model"], contents=f"{text}\n\nQUESTION: {question}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=300,
                                           response_mime_type="application/json"))
    um = r.usage_metadata
    blocked = bool(getattr(r, "prompt_feedback", None) and getattr(r.prompt_feedback, "block_reason", None))
    fin = ""
    try: fin = str(r.candidates[0].finish_reason)
    except Exception: pass
    refused = blocked or "SAFETY" in fin or "PROHIBITED" in fin
    return (r.text or "") if not blocked else "", um.prompt_token_count or 0, um.candidates_token_count or 0, refused


def call_gemma(client, cfg, text, question):
    """Gemma on the Gemini API: no system instruction and no JSON mode, so the instruction rides in the user turn."""
    from google.genai import types
    r = client.models.generate_content(
        model=cfg["model"], contents=f"{SYSTEM}\n\n{text}\n\nQUESTION: {question}",
        config=types.GenerateContentConfig(max_output_tokens=300))
    um = r.usage_metadata
    blocked = bool(getattr(r, "prompt_feedback", None) and getattr(r.prompt_feedback, "block_reason", None))
    fin = ""
    try: fin = str(r.candidates[0].finish_reason)
    except Exception: pass
    refused = blocked or "SAFETY" in fin or "PROHIBITED" in fin
    return (r.text or "") if not blocked else "", um.prompt_token_count or 0, um.candidates_token_count or 0, refused


def call_ollama(client, cfg, text, question):
    """Local model through Ollama's native chat endpoint: thinking off, JSON mode, greedy decoding."""
    import urllib.request
    body = json.dumps({
        "model": cfg["model"], "stream": False, "think": False, "format": "json",
        "options": {"num_predict": 300, "temperature": 0},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"{text}\n\nQUESTION: {question}"}]}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.loads(r.read())
    return (obj.get("message") or {}).get("content", "") or "", obj.get("prompt_eval_count", 0) or 0, obj.get("eval_count", 0) or 0, False


CALLS = {"haiku": call_haiku, "haiku_med": call_haiku, "opus": call_haiku, "luna": call_luna, "flashlite": call_flashlite,
         "gemma": call_ollama, "qwen": call_ollama}


def make_client(vendor):
    if VENDORS[vendor].get("local"):
        import urllib.request
        try:
            tags = json.loads(urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5).read())
        except Exception as e:
            sys.exit(f"Ollama not reachable at {OLLAMA}: {type(e).__name__}")
        names = {m["name"] for m in tags.get("models", [])}
        want = VENDORS[vendor]["model"]
        if want not in names and f"{want}:latest" not in names:
            sys.exit(f"model {want} is not pulled. Run: ollama pull {want}   (installed: {sorted(names)})")
        return None
    key = os.environ.get(VENDORS[vendor]["env"])
    if not key:
        sys.exit(f"{VENDORS[vendor]['env']} is not set. Run: source ~/Documents/HQ/01_Holding/machine/load-keys.sh")
    if vendor in ("haiku", "haiku_med", "opus"):
        import anthropic; return anthropic.Anthropic(api_key=key, max_retries=0)
    if vendor == "luna":
        import openai; return openai.OpenAI(api_key=key, max_retries=0)
    if vendor in ("flashlite", "gemma"):
        from google import genai; return genai.Client(api_key=key)


# ---------- parsing ----------

JSON_RE = re.compile(r"\{.*?\}", re.S)
REFUSAL_WORDS = re.compile(r"\b(I can(?:'|no)t|I cannot|I'm unable|I am unable|not able to (?:help|assist|answer))\b", re.I)


def parse(text):
    """Return (p_yes, answer, confidence) or (None, None, None) if the reply is not a usable JSON answer."""
    if not text: return None, None, None
    cand = text.strip()
    cand = re.sub(r"^```(?:json)?|```$", "", cand, flags=re.M).strip()
    m = JSON_RE.search(cand)
    if not m: return None, None, None
    try: obj = json.loads(m.group(0))
    except Exception: return None, None, None
    ans = str(obj.get("answer", "")).strip().lower()
    if ans not in ("yes", "no"): return None, None, None
    try: conf = float(obj.get("confidence"))
    except Exception: conf = 50.0
    if conf <= 1.0: conf *= 100  # a model that answers 0.9 instead of 90
    conf = max(50.0, min(100.0, conf)) / 100.0
    return (conf if ans == "yes" else 1.0 - conf), ans, conf


def is_rate_limit(e):
    s = str(e).lower(); code = getattr(e, "status_code", None) or getattr(e, "code", None)
    return code in (429, 500, 502, 503, 504, 529) or "429" in s or "rate" in s and "limit" in s or "overloaded" in s or "resource_exhausted" in s or "503" in s


def run_row(vendor, client, row, retries=8):
    cfg = VENDORS[vendor]
    rec = {"id": row["id"], "task": row["task"], "human_label": row["human_label"], "vendor": vendor, "model": cfg["model"]}
    delay = 2.0
    for attempt in range(retries):
        try:
            text, tin, tout, refused = CALLS[vendor](client, cfg, row["text"], row["question"])
            break
        except Exception as e:
            if is_rate_limit(e) and attempt < retries - 1:
                time.sleep(delay + random.random()); delay = min(delay * 2, 60); continue
            # keep only the exception class and a short message; never a raw body
            rec.update(probability=None, status="error", error={"type": type(e).__name__, "message": str(e)[:200]},
                       input_tokens=0, output_tokens=0, usd=0.0)
            return rec
    usd = tin / 1e6 * cfg["price"][0] + tout / 1e6 * cfg["price"][1]
    rec.update(input_tokens=tin, output_tokens=tout, usd=usd)
    p, ans, conf = parse(text)
    if p is None:
        status = "refusal" if (refused or REFUSAL_WORDS.search(text or "")) else "no_prob"
        rec.update(probability=None, status=status, raw=(text or "")[:400])
    else:
        rec.update(probability=p, answer=ans, confidence=conf, status="ok")
    return rec


def load_rows(dataset):
    path = DATASETS[dataset]
    if not os.path.exists(path):
        sys.exit(f"{path} missing. Run load_legalbench.py or load_alinia.py first.")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", default="all", help="haiku | luna | flashlite | gemma | qwen | all (all = the four headline vendors)")
    ap.add_argument("--dataset", default="all", help="legalbench | alinia | all")
    ap.add_argument("--limit-per-task", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=0, help="override the per-vendor default")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    vendors = ["haiku", "luna", "flashlite", "gemma"] if a.vendor == "all" else a.vendor.split(",")
    datasets = list(DATASETS) if a.dataset == "all" else a.dataset.split(",")
    os.makedirs("results/multi", exist_ok=True)

    for vendor in vendors:
        client = make_client(vendor)
        rows = [r for d in datasets for r in load_rows(d)]
        if a.probe:
            rec = run_row(vendor, client, rows[0])
            print(json.dumps({k: v for k, v in rec.items() if k != "raw"} | {"raw": rec.get("raw", "")[:200]}, indent=2)); continue
        out = f"results/multi/{vendor}.jsonl"
        done = set()
        if os.path.exists(out):
            for l in open(out):
                r = json.loads(l)
                if r.get("status") in ("ok", "refusal"): done.add(r["id"])
        todo, per_task = [], {}
        for r in rows:
            if r["id"] in done:
                per_task[r["task"]] = per_task.get(r["task"], 0) + 1; continue
            if a.limit_per_task and per_task.get(r["task"], 0) >= a.limit_per_task: continue
            per_task[r["task"]] = per_task.get(r["task"], 0) + 1
            todo.append(r)
        conc = a.concurrency or VENDORS[vendor]["concurrency"]
        print(f"[{vendor}] {len(done)} done, {len(todo)} to run, concurrency {conc}", file=sys.stderr)
        n = ok = ref = bad = 0; usd = 0.0
        with open(out, "a") as f, ThreadPoolExecutor(conc) as ex:
            for fut in as_completed([ex.submit(run_row, vendor, client, r) for r in todo]):
                rec = fut.result()
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                n += 1; usd += rec.get("usd", 0.0)
                ok += rec["status"] == "ok"; ref += rec["status"] == "refusal"; bad += rec["status"] in ("error", "no_prob")
                if n % 25 == 0:
                    print(f"[{vendor}] {n}/{len(todo)} ok={ok} refusals={ref} errors={bad} ${usd:.4f}", file=sys.stderr)
        print(f"[{vendor}] DONE {n} ok={ok} refusals={ref} errors={bad} usd={usd:.4f}")


if __name__ == "__main__":
    main()
