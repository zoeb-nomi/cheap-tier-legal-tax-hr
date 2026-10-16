# Findings

Results from `results/multi/summary.csv`, `agreement.json` and `cascade.json`. Four cheap models (Haiku, Luna, Flash-Lite, Gemma) and one frontier reference (Opus) answered 1,414 lawyer-labelled yes/no items with a stated confidence. "Sure" means a stated confidence of 0.9 or more. Pooled figures are the item-weighted combination of the two "ALL" rows in `summary.csv`.

## Accuracy is close among the cheap models

Pooled over all 1,414 items the four cheap models score 0.81 to 0.83 (Luna 0.81, Haiku 0.82, Gemma 0.83, Flash-Lite 0.83). Opus scores 0.89. On LegalBench the cheap models sit between 0.82 (Gemma) and 0.86 (Haiku, Luna and Flash-Lite), and Opus at 0.94. On the Alinia advice items the cheap range is 0.76 (Luna) to 0.84 (Gemma), and Opus is at 0.84, no better than Gemma.

## What "sure" means differs

Haiku says it is sure on 28 of 100 answers, and it is right on 99 of 100 of those (3 wrong among 392).

The other three are sure far more often: Luna on 92 of 100 answers, Flash-Lite on 93, Gemma on 93. On those answers they are right 84 to 85 of 100 times, so 15 to 16 of every 100 "sure" answers are wrong. Opus is sure on 47 of 100 and is wrong on 1 of its 661 sure answers.

The same scale carries a different meaning per model, so a confidence gate has to be set per model from its own trust curve (`trust_curves.csv`).

## The dissent rule

When two cheap models disagree, each is right about half the time: Haiku and Luna disagree on 147 of 1,414 items and are right 0.52 and 0.48 of the time there. When they agree, accuracy is 0.85 to 0.91.

Haiku is almost never contradicted when it is sure: another cheap model disagrees with a sure Haiku answer on only 2, 4 and 6 items (Luna, Flash-Lite, Gemma), and Haiku was wrong on none of them. In the other direction, when Haiku disagrees with a sure answer from another model, that other model is wrong on 48 of 102 (Luna), 78 of 159 (Flash-Lite) and 100 of 220 (Gemma). The rule that follows: have two cheap models read, and send only the disagreements to a stronger reader.

## Majority vote does not beat the best single model

A four-model vote on LegalBench scores 0.87 against 0.86 for the best single model, a gain of about one point. On the Alinia items the vote scores 0.81 and the best single model, Gemma, scores 0.84, so the vote is worse. There are 74 items that all four cheap models got wrong (`all_wrong` in `agreement.json`), mostly in the CUAD tasks and Alinia legal.

## Per domain, with cost per 1,000 items

All at gate 0.85, from `cascade.json`.

**HR (100 items).** Haiku alone scores 0.96 at 0.07 USD per 1,000 items. Opus alone also scores 0.96, at 2.63. Opus behind a Haiku gate (0.48) or as a referee (0.16) does not improve it. Haiku clears 0.89 of items and a person reads the other 0.11. HR is solved by Haiku alone.

**Tax (438 items).** Opus alone scores 0.96 at 5.24. Haiku alone scores 0.85 at 0.18. Two cheap readers (Gemma and Haiku) with Opus deciding their disagreements, which are 18 of 100 items, score 0.95 at 1.51. A Haiku gate with Opus on the rest scores 0.95 at 3.47. Tax is best served by two cheap readers plus a frontier referee: nearly Opus accuracy at under a third of its cost.

**Legal (876 items).** Opus alone scores 0.85 at 4.78. Haiku alone scores 0.79 at 0.16. The best two-reader system (Flash-Lite and Haiku, Opus deciding 14 of 100 items) scores 0.82 at 1.23, and the Haiku gate with Opus on the rest scores 0.84 at 3.41. Even the frontier model stops at 0.85, so legal is capped by the labels, not by the model tier. The Alinia card reports Krippendorff alpha of 0.63 for tax and 0.71 for legal, and Opus scores only 0.72 on the Alinia legal subset.

The texts are short, there is one prompt, and confidence is verbalised; see the limitations in the README.
