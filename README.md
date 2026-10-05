# RELATE Framework

RELATE measures relational orientation in supportive model language: whether a response positions the assistant as the user's ongoing source of support (inward-facing) or directs the user toward people in their own social world (outward-scaffolding). The benchmark crosses 76 help-seeking situations from CounselBench-Eval with three disclosure styles and evaluates seven models, producing 1,596 six-turn dialogues labelled at the sentence level.

## Two-phase workflow

**Phase 1: Corpus**

The released `data/corpus/situations.jsonl` lists the 76 situations by ID and
topic only (see Data release below). To run the generation pipeline, first
rebuild it with the CounselBench-Eval text, then write the stimuli:

```bash
python scripts/inspect_counselbench.py   # optional: topics + schema check
python scripts/build_situations.py --force   # downloads CounselBench-Eval from HF (pinned revision)
python run.py --config configs/corpus.yaml --stage stimulus   # needs OPENAI_API_KEY
```

-> `data/corpus/situations.jsonl` + `situations.meta.json` (76 seeds)
-> `data/corpus/stimuli.jsonl` (228 conditions: vignette + fixed_turn1)

The rebuilt situations match the released IDs and topics. Vignettes and
opening turns are regenerated with GPT-4o-mini, so they will not match the
paper's stimuli word for word. Do not commit the rebuilt `situations.jsonl`
or `stimuli.jsonl` (see Data release).

**Phase 2: Experiment (repeat)**

```bash
python run.py --config configs/experiment/relate_main.yaml --stage dialogue
```

Results go to `outputs/relate_main/`. The runner covers the corpus and dialogue
stages; sentence-level judging is run separately (see LLM Judge below).

## Topic filtering

`configs/corpus.yaml` excludes topics that fit RELATE poorly by default:

- `counseling-fundamentals`, `professional-ethics`, `legal-regulatory`

Edit `exclude_topics` / `include_topics` in `configs/corpus.yaml`. Situations are
deduped by `questionID`, assigned `S000...` sorted by `question_id`, with HF
revision pinned. Rebuild with `python scripts/build_situations.py --force`.

## Data release

The paper's experiment produced 1,596 six-turn dialogues (76 situations × 3
personas × 7 target models). The situations come from CounselBench-Eval,
which is licensed under CC BY-NC-ND 4.0 and does not permit sharing adapted
material. This repository therefore does not include:

- the CounselBench-Eval question text (`data/corpus/situations.jsonl` keeps
  `question_id`, `topic` and `situation_id` only),
- the rewritten vignettes and opening turns (`data/corpus/stimuli.jsonl`),
- the generated dialogues (`dialogues.jsonl`),
- simulated-user turns in the human annotation workbooks.

It does include the generation pipeline and configs, every sentence-level
judgment of the assistant responses, the human annotations of assistant
sentences, and the code and intermediate tables that reproduce every table
and figure in the paper. None of the analyses need the excluded files.

## Configs

| Config | Purpose |
|--------|---------|
| `configs/corpus.yaml` | Build the filtered corpus (GPT-4o-mini stimulus writer) |
| `configs/corpus_mock.yaml` | Same corpus with a mock stimulus writer, written under `outputs/` |
| `configs/experiment/relate_main.yaml` | Paper's main experiment: 76 situations × 3 personas, 7 target models (dialogue stage) |
| `configs/experiment/full.yaml` | Offline smoke test over the full corpus (mock providers, no API calls) |
| `configs/experiment/pilot.yaml` | Early pilot: 10 situations, 4 models |

## Quick start (mock)

```bash
pip install -r requirements.txt
python run.py --config configs/corpus_mock.yaml --stage stimulus
python run.py --config configs/experiment/full.yaml --stage dialogue
```

This downloads CounselBench-Eval from Hugging Face and uses mock providers
throughout, so it needs no API keys. Outputs go to `outputs/` (gitignored).

## LLM Judge

Prompt: `scripts/llm_judge/prompt.txt`. Each row is one
(job, criterion) pair keyed by `(dialogue_id, turn_id, sentence_id, class)`.

The judge scripts read `dialogues.jsonl` from the repository root. After
regenerating dialogues, copy `outputs/relate_main/dialogues.jsonl` there.

Dialogue generation uses seed `0`. The secondary-judge sample and the
human-validation sample use seed `11`. Dialogue-clustered bootstrap intervals
use seed `20260913`.

API keys are read from the process environment (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `DEEPINFRA_API_KEY`). This repository does not include
an env file.

### DeepSeek-R1-Distill-Qwen-32B (primary judge)

Merged, deduped corpus (1,596 dialogues, 345,970 unique keys):

- `deepseek_judge/judgments.jsonl.gz` — canonical merged file (gzip; ~15 MB)
- `deepseek_judge/manifest.json` — merge strategy and source list

The uncompressed `judgments.jsonl` (~188 MB) exceeds GitHub’s 100 MB limit, so
only the `.gz` is tracked. Decompress before use:

```bash
gzip -dk deepseek_judge/judgments.jsonl.gz
# -> deepseek_judge/judgments.jsonl
```

Or stream without writing a local copy:

```bash
zcat deepseek_judge/judgments.jsonl.gz | head
python -c "
import gzip, json
with gzip.open('deepseek_judge/judgments.jsonl.gz', 'rt') as f:
    print(json.loads(f.readline()))
"
```

### OpenAI gpt-4o (validation judge, 10% sample)

161 dialogues (~10% of corpus), 7 target models, via Batch API:

- `openai_judge/judgments.jsonl.gz` — canonical file (gzip; ~2 MB)
- `openai_judge/sampled_dialogue_ids.json` — which dialogues were judged
- `openai_judge/manifest.json` — source path and counts

Decompress before use:

```bash
gzip -dk openai_judge/judgments.jsonl.gz
# -> openai_judge/judgments.jsonl
```

### Human annotations

Human validation covers 881 assistant sentences, 1.3% of the corpus, drawn
from 21 dialogues (three per target model, seed 11). Two researchers
independently labelled inward-facing (IF) and outward-scaffolding (OS) on
those sentences. A few assistant turns were left out of that labelling. In
those turns, the stored assistant text reads as the simulated user rather
than as the model, so the annotators did not evaluate them. Four such turns
were removed, and the 881 sentences are what remained. The released
workbooks are those labels. Rows for simulated-user turns, which the
annotators saw as context but did not label, are omitted:

- `annotation/annotator1_judgments.xlsx`
- `annotation/annotator2_judgments.xlsx`

```bash
python analysis/human_agreement.py
```

### Reproducing tables

`analysis/analyze.py` reads `analysis/data/sentences.parquet`. To check that
table against the gzipped judgment files, rebuild it independently:

```bash
python analysis/build_table_from_canonical.py --verify
```

This writes `analysis/data/sentences_from_canonical.parquet`, which matches
`sentences.parquet` row for row, and prints sentence and dialogue counts.

The paper's figures and cited numbers come from:

```bash
python analysis/make_paper_figures.py           # -> analysis/paper_figures/
python analysis/make_human_agreement_figures.py # -> analysis/paper_figures/
python analysis/paper_numbers.py
python analysis/model_turn_rates.py             # IF/OS by model and turn (appendix)
python analysis/high_stakes_by_model.py         # high-stakes topics by model (appendix)
```

## Layout

```
configs/corpus.yaml
configs/experiment/
data/corpus/                  situation IDs and topics, build metadata
deepseek_judge/               merged DeepSeek judgments
openai_judge/                 gpt-4o validation judgments (10% sample)
annotation/                   human judgments (annotator 1 and annotator 2)
analysis/                     tables, figures, and reproduction scripts
analysis/paper_figures/       figures as they appear in the paper
relate/                       corpus and dialogue generation
outputs/                      experiment + judge run outputs (gitignored)
scripts/build_situations.py
scripts/inspect_counselbench.py
scripts/llm_judge/
run.py
```

## License

Code is released under the MIT License (see `LICENSE`).

The data files are released under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), the
license of the source dataset:

- `data/corpus/`
- `deepseek_judge/`, `openai_judge/`
- `annotation/`
- `analysis/data/`

The experiment is built on
[CounselBench-Eval](https://huggingface.co/datasets/izi-ano/CounselBench-Eval)
(CC BY-NC-ND 4.0; revision pinned in `data/corpus/situations.meta.json`),
whose questions were originally posted by users of CounselChat, a public
online counseling forum. No CounselBench-Eval text or adapted material is
redistributed here. Please cite CounselBench when using this work.
