# Basira live RAG evaluation

- Run: `eval_runs/final_regression_closure_20261024T182101Z/final7_final`
- Questions: 47 (executed 47)
- API calls: 329; spent: $0.062433 of $0.25
- Fatal: none

## Aggregate

- **all**: n=47, answer_pass=1.0 (42/42 scored answerable), task_pass=1.0 (46/46), heldout_source_citation=None (0), abstention=1.0, fact=1.0, in_context=0.9767, coverage=0.7946, citation_validity=1.0, citation_support=1.0, citation_gold_precision=0.7411, unsupported_final=0, manual=['ambiguous_sunan'], cost=$0.062433
- **primary**: n=0, answer_pass=None (0/0 scored answerable), task_pass=None (0/0), heldout_source_citation=None (0), abstention=None, fact=None, in_context=None, coverage=None, citation_validity=None, citation_support=None, citation_gold_precision=None, unsupported_final=0, manual=[], cost=$0.000000
- **heldout**: n=0, answer_pass=None (0/0 scored answerable), task_pass=None (0/0), heldout_source_citation=None (0), abstention=None, fact=None, in_context=None, coverage=None, citation_validity=None, citation_support=None, citation_gold_precision=None, unsupported_final=0, manual=[], cost=$0.000000

## Per source

- **quran**: n=15, answer_pass=1.0, in_context=1.0, coverage=0.8, citation_gold_precision=0.7689, unsupported_final=0
- **hadith**: n=15, answer_pass=1.0, in_context=0.9333, coverage=0.6444, citation_gold_precision=0.6889, unsupported_final=0
- **tafsir**: n=6, answer_pass=1.0, in_context=1.0, coverage=0.9167, citation_gold_precision=0.8056, unsupported_final=0
- **fiqh**: n=20, answer_pass=1.0, in_context=0.95, coverage=0.6333, citation_gold_precision=0.7767, unsupported_final=0
- **sira**: n=9, answer_pass=1.0, in_context=1.0, coverage=0.9444, citation_gold_precision=0.6111, unsupported_final=0
- **unanswerable**: n=4, answer_pass=None, in_context=None, coverage=None, citation_gold_precision=None, unsupported_final=0

## Failures

- `hadith_itikaf` pass=True fact=True abstain=True gate=True in_ctx=False reasons=['no gold evidence in context']: كان النبي صلى الله عليه وسلم يعتكف في العشر الأواخر من رمضان (مصدر 2). فقد ورد في حديث عائشة رضي الله عنها: "كان رسول الله صلى الله عليه وسلم إذا أراد أن يعتكف صلى الصبح ثم دخل في المكان الذي يريد أن يعتكف فيه فأراد أن يعتكف العشر الأواخر من رمضان" (مصدر 1).
