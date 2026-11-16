# RAG evaluation governance

`gates.yaml` defines the policy thresholds for retrieval quality, citation integrity,
answer quality, regression margins, and required dataset splits. These are policy
targets, not measured outcomes.

## Gate meanings

- Precision@5: fraction of the five retrieved chunks judged relevant.
- Recall@5: fraction of the question's annotated relevant chunks retrieved.
- Required reference accuracy: fraction of emitted citations that resolve to the
  actual retrieved source chunk and its metadata; the required value is 100%.
- Fabricated citations: citations that do not resolve to a retrieved chunk; none are
  permitted.
- No-answer accuracy: proportion of unanswerable questions correctly declined.
- Supported-claim accuracy: proportion of factual answer claims supported by cited
  evidence.
- Regression margins: maximum allowed decrease from a frozen baseline.
- Dataset requirements: both tuning and held-out splits, document-level separation,
  and category counts must be available.

## Current evidence and limits

Four source files are available under `data/documents/` and represented in the
current index. The older `data/evaluation_results.json` report has 11 query records
but lacks chunk-level relevance labels, verified answer keys, document-separated
splits, and a dataset hash. It cannot support governed metrics.

`datasets/tuning.jsonl` and `datasets/held_out.jsonl` contain 27 source-checked,
synthetic candidate questions: 10 tuning and 17 held-out, based on four documents
with document-level separation. Tuning has 3 no-answer questions; held-out has 5.
`datasets/manifest.json` records source and dataset hashes and category counts.
Every row is marked `candidate_requires_human_review`; these are not human-verified
ground truth and must not be used for evaluation until a human reviews and approves
the questions, answers, evidence, and labels. Do not tune on held-out questions.

The sample is small and category counts are uneven because only four available
documents support source-backed questions. It cannot establish broad academic RAG
quality even after review. The held-out split has not been run.

The local repository now protects the gates file in commit
`5ace2f1c94959bee41089b4fe24a4493fec51163`. Held-out evaluation remains blocked
until the candidate dataset is human-reviewed and tuning is frozen.
