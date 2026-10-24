# Basira live RAG evaluation

- Run: `eval_runs/final_regression_closure_20261024T182101Z/final7_regression_final_check`
- Questions: 4 (executed 4)
- API calls: 28; spent: $0.002824 of $0.25
- Fatal: none

## Aggregate

- **all**: n=4, answer_pass=1.0 (4/4 scored answerable), task_pass=1.0 (4/4), heldout_source_citation=None (0), abstention=1.0, fact=1.0, in_context=1.0, coverage=0.625, citation_validity=1.0, citation_support=1.0, citation_gold_precision=0.9167, unsupported_final=0, manual=[], cost=$0.002824
- **primary**: n=0, answer_pass=None (0/0 scored answerable), task_pass=None (0/0), heldout_source_citation=None (0), abstention=None, fact=None, in_context=None, coverage=None, citation_validity=None, citation_support=None, citation_gold_precision=None, unsupported_final=0, manual=[], cost=$0.000000
- **heldout**: n=0, answer_pass=None (0/0 scored answerable), task_pass=None (0/0), heldout_source_citation=None (0), abstention=None, fact=None, in_context=None, coverage=None, citation_validity=None, citation_support=None, citation_gold_precision=None, unsupported_final=0, manual=[], cost=$0.000000

## Per source

- **quran**: n=1, answer_pass=1.0, in_context=1.0, coverage=1.0, citation_gold_precision=1.0, unsupported_final=0
- **hadith**: n=3, answer_pass=1.0, in_context=1.0, coverage=0.5, citation_gold_precision=0.8889, unsupported_final=0
- **fiqh**: n=4, answer_pass=1.0, in_context=1.0, coverage=0.625, citation_gold_precision=0.9167, unsupported_final=0

## Failures

- none
