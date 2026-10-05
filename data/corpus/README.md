# Benchmark corpus

Data source: **CounselBench-Eval** only, pinned revision in `configs/corpus.yaml`.

Released files:
- `situations.jsonl` - the 76 situations used in the paper: `situation_id`,
  CounselBench `question_id`, and `topic`. The question text is not
  redistributed (CounselBench-Eval is CC BY-NC-ND 4.0).
- `situations.meta.json` - build provenance (HF revision, filters, count)

Rebuild the situations with their text from Hugging Face:
```bash
python scripts/build_situations.py --force
```

Then generate vignettes + opening turns (228 conditions = 76 x 3 personas):
```bash
python run.py --config configs/corpus.yaml --stage stimulus
```

The rebuilt files contain CounselBench-Eval text or material adapted from it.
Keep them local; do not commit them.
