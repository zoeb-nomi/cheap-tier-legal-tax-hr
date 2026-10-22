# Where a 10-cent model goes in a tax, legal or HR workflow

By [Zoeb Nomi](https://www.zoebnomi.com).

Five models answered the same yes/no question, with a stated confidence, on 1,414 expert-labelled items: 744 from LegalBench (eight tasks) and 670 English items from Alinia's regulated-advice-bench (tax and legal subsets). Four are cheap-tier models; the fifth, Opus, is a frontier reference. The cheap models land within a few points of each other on accuracy, but what they mean by "sure" differs a lot. Haiku is rarely sure (28 of 100 answers at a stated confidence of 0.9 or more) and is right on 99 of 100 of those. The other three are sure on 92 to 93 of 100 answers and are wrong on 15 to 16 of every 100 of those. The study then compares simple workflows per domain (legal, tax, HR): one cheap model alone, a confidence gate, a second cheap reader, and a frontier referee. Write-up: [zoebnomi.com/notes/cheap-tier-workflows](https://www.zoebnomi.com/notes/cheap-tier-workflows/). Plain-language findings: [FINDINGS.md](FINDINGS.md).

This repository holds the code, the prompts, the item ids and the result summaries. It does not hold any dataset text, labels or per-item model outputs.

## Models

| Model | Vendor | Price per 1M tokens, input / output (USD, as used) | Notes |
|---|---|---|---|
| Claude Haiku 5.5 | Anthropic | 0.10 / 0.50 | effort "low" |
| GPT-6 Luna | OpenAI | 0.10 / 0.50 | reasons by default |
| Gemini 3.1 Flash-Lite | Google | 0.25 / 1.50 | |
| Gemma 4 26B | Google, run locally through Ollama | 0.00 / 0.00 | thinking off |
| Opus 5.5 | Anthropic | 4.00 / 20.00 | frontier reference, default effort (medium) |

Prices are list prices as of 2026-10-15.

## Data

| Dataset | Items used | Source and licence |
|---|---|---|
| LegalBench: cuad_anti-assignment, cuad_non-compete, cuad_change_of_control (100 each), hearsay (94), personal_jurisdiction (50), consumer_contracts_qa (100) | 544 | Guha et al. 2023, "LegalBench", Hugging Face `nguha/legalbench`, CC BY 4.0 per the task card. The three cuad_ tasks derive from CUAD, by The Atticus Project, CC BY 4.0. |
| LegalBench: sara_entailment, learned_hands_employment (100 each) | 200 | Same collection (`nguha/legalbench`, test split). SARA and Learned Hands are separate source datasets; check each task card on Hugging Face for its licence and attribution. |
| Alinia regulated-advice-bench, tax (338) and legal (332) subsets | 670 | `alinia/regulated-advice-bench`, CC BY-NC-SA 4.0, published 2026-09-28. English rows only. No rows are redistributed here. |

The exact sample can be reproduced: `data/item_ids.txt` lists the 1,414 item ids, one per line (ids only, no text, label or score). The LegalBench samples are seeded (seed 42), balanced yes/no, from the test split. Because the Alinia data is non-commercial and share-alike, use that part only under those terms.

## The one prompt

Every model got the same system prompt, from `judge_multi.py`, and the text plus the question in the user turn:

```
SYSTEM = ("You answer one yes/no question about a piece of text. Reply with JSON only, no prose, exactly of the form "
          '{"answer": "yes" or "no", "confidence": N} where N is an integer from 50 to 100 giving the probability, '
          "in percent, that your answer is correct.")
```

The question wordings for the LegalBench tasks are in `data/prompts.json` and `data/prompts_ext.json`. The stated confidence is converted to p(yes), and all calibration numbers are computed from that.

## Reproduce

1. Python 3.13. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. For Gemma, install Ollama and pull `gemma4:26b`.
2. Put your API keys in the environment as variables, never in files: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and a Hugging Face token if your setup needs one. `source` a key loader of your own (for example one that reads from your OS keychain). The docstring in `judge_multi.py` mentions the author's own loader path; ignore it.
3. Build the samples (this writes the gitignored `data/*.jsonl` files):
   ```
   python load_legalbench.py
   python load_ext.py
   python load_alinia.py
   ```
4. Pilot, then look at the output before spending more:
   ```
   python judge_multi.py --vendor all --dataset all --limit-per-task 10
   ```
5. Full run (resumes, skips rows already done):
   ```
   python judge_multi.py --vendor all --dataset all
   ```
6. Score and compare:
   ```
   python calibrate_multi.py
   python agreement.py
   python cascade.py --frontier opus --gate 0.85
   ```

Outputs go to `results/multi/`. The per-item `*.jsonl` files stay local (gitignored). `summary.csv`, `metrics.json`, `trust_curves.csv`, `agreement.json` and `cascade.json` are the committed summaries. The rows labelled "decisions" in `summary.csv` and `metrics.json` are the OpenAI Decisions API baseline from the earlier `decisions-api-legal-quiz` repo, not part of this study, and `calibrate_multi.py` only adds them if that repo's `results/items.csv` is present. `haiku_med` is Haiku at its default effort, run as a check.

## Results

All numbers are rounded to 2 decimals and come from `results/multi/summary.csv` and `results/multi/cascade.json`.

### Accuracy per task

Share of items answered correctly (answer is "yes" when p(yes) is 0.5 or more).

| Task | n | Haiku | Luna | Flash-Lite | Gemma | Opus |
|---|---|---|---|---|---|---|
| CUAD anti-assignment | 100 | 0.79 | 0.79 | 0.80 | 0.78 | 0.92 |
| CUAD non-compete | 100 | 0.83 | 0.82 | 0.88 | 0.86 | 0.90 |
| CUAD change of control | 100 | 0.76 | 0.77 | 0.77 | 0.66 | 0.92 |
| Hearsay | 94 | 0.85 | 0.81 | 0.80 | 0.72 | 0.89 |
| Personal jurisdiction | 50 | 0.96 | 0.98 | 0.92 | 0.86 | 0.96 |
| Consumer contracts QA | 100 | 0.97 | 0.98 | 0.98 | 0.97 | 0.97 |
| SARA entailment | 100 | 0.83 | 0.86 | 0.79 | 0.79 | 0.98 |
| Learned Hands employment | 100 | 0.96 | 0.95 | 0.95 | 0.93 | 0.96 |
| Alinia tax | 338 | 0.85 | 0.84 | 0.91 | 0.96 | 0.95 |
| Alinia legal | 332 | 0.69 | 0.68 | 0.70 | 0.71 | 0.72 |
| All LegalBench | 744 | 0.86 | 0.86 | 0.86 | 0.82 | 0.94 |
| All Alinia | 670 | 0.77 | 0.76 | 0.81 | 0.84 | 0.84 |

### Coverage at 0.9 per task

Share of answers where the model stated a confidence of 0.9 or more. Accuracy on those answers is in the `accuracy_at_90` column of `summary.csv`.

| Task | n | Haiku | Luna | Flash-Lite | Gemma | Opus |
|---|---|---|---|---|---|---|
| CUAD anti-assignment | 100 | 0.26 | 0.96 | 0.99 | 1.00 | 0.70 |
| CUAD non-compete | 100 | 0.53 | 0.94 | 0.98 | 0.99 | 0.61 |
| CUAD change of control | 100 | 0.32 | 0.95 | 0.99 | 0.98 | 0.57 |
| Hearsay | 94 | 0.21 | 0.78 | 0.99 | 0.93 | 0.39 |
| Personal jurisdiction | 50 | 0.40 | 0.94 | 0.98 | 0.90 | 0.66 |
| Consumer contracts QA | 100 | 0.55 | 0.93 | 0.98 | 0.97 | 0.60 |
| SARA entailment | 100 | 0.36 | 0.94 | 1.00 | 0.98 | 0.48 |
| Learned Hands employment | 100 | 0.73 | 1.00 | 1.00 | 0.99 | 0.86 |
| Alinia tax | 338 | 0.19 | 0.93 | 0.89 | 1.00 | 0.43 |
| Alinia legal | 332 | 0.04 | 0.88 | 0.82 | 0.76 | 0.19 |
| All LegalBench | 744 | 0.42 | 0.93 | 0.99 | 0.97 | 0.61 |
| All Alinia | 670 | 0.11 | 0.90 | 0.86 | 0.88 | 0.31 |

### Workflows per domain, gate 0.85

Domains: legal = the five LegalBench contract and evidence tasks plus consumer_contracts_qa and Alinia legal; tax = SARA entailment plus Alinia tax; HR = Learned Hands employment. "Gate 0.85" means Haiku clears an item when its stated confidence is 0.85 or more. For the "a person reads the rest" row, accuracy is counted on the cleared items only, and the last column is the share of items still sent to a person. For the two-readers row the table shows the pairing with the highest accuracy in that domain (ties go to the cheaper); all pairings are in `cascade.json`. Cost is the recorded token cost at list price, scaled to 1,000 items.

| Domain (n) | System | Accuracy | USD per 1,000 items | Share a person still reads |
|---|---|---|---|---|
| Legal (876) | Opus alone | 0.85 | 4.78 | 0.00 |
| Legal (876) | Haiku alone | 0.79 | 0.16 | 0.00 |
| Legal (876) | Haiku gate, Opus reads the rest | 0.84 | 3.41 | 0.00 |
| Legal (876) | Haiku gate, a person reads the rest | 0.93 (cleared items only) | 0.16 | 0.52 |
| Legal (876) | Two cheap readers, Opus decides disagreements (flashlite + Haiku) | 0.82 | 1.23 | 0.00 |
| Tax (438) | Opus alone | 0.96 | 5.24 | 0.00 |
| Tax (438) | Haiku alone | 0.85 | 0.18 | 0.00 |
| Tax (438) | Haiku gate, Opus reads the rest | 0.95 | 3.47 | 0.00 |
| Tax (438) | Haiku gate, a person reads the rest | 0.99 (cleared items only) | 0.18 | 0.50 |
| Tax (438) | Two cheap readers, Opus decides disagreements (gemma + Haiku) | 0.95 | 1.51 | 0.00 |
| HR (100) | Opus alone | 0.96 | 2.63 | 0.00 |
| HR (100) | Haiku alone | 0.96 | 0.07 | 0.00 |
| HR (100) | Haiku gate, Opus reads the rest | 0.96 | 0.48 | 0.00 |
| HR (100) | Haiku gate, a person reads the rest | 1.00 (cleared items only) | 0.07 | 0.11 |
| HR (100) | Two cheap readers, Opus decides disagreements (gemma + Haiku) | 0.96 | 0.16 | 0.00 |

## Limitations

- The LegalBench texts are short excerpts and clauses, much shorter than a real contract file. Results on full documents may differ.
- The advice labels in the Alinia data are themselves noisy: the dataset card reports Krippendorff alpha of 0.63 for tax and 0.71 for legal. Part of the legal ceiling is the labels, not the models.
- Confidence is verbalised: the model states a number. It is not a token probability. The numbers reflect how each model uses that scale, which differs by model.
- Prices are list prices on 2026-10-15 and will change.
- Gemma was run locally through Ollama with thinking off. The hosted Gemini API route for Gemma was unusable on 2026-10-15 (fixed thinking budget, slow, many 500 errors on a pilot), so it was dropped.
- Haiku was run at effort "low". A run at its default effort (`haiku_med`) shows the same accuracy: 0.85 vs 0.85 on the 544 original LegalBench items and 0.77 vs 0.77 on the Alinia items.
- One prompt, one run per model, no repeated sampling, no prompt tuning per model.
- Samples are balanced yes/no within each LegalBench task, so base rates differ from real workloads.

## Cost

The whole study cost $7.60 in API calls, of which $6.75 was Opus.

## Licence

Code: MIT, see `LICENSE`. The datasets keep their own licences (see the Data table); none of their rows are included.
