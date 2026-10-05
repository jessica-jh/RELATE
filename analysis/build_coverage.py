import pandas as pd

df = pd.read_parquet("analysis/data/sentences.parquet")
df = df[df["judge"] == "deepseek-r1-distill-qwen-32b"]

rows = []
for (model, persona), g in df.groupby(["target_model", "persona_id"]):
    n_dialogues = g["dialogue_id"].nunique()
    n_sentences = len(g)
    n_turns = g.groupby("dialogue_id")["turn_id"].nunique().sum()
    n_parsefail = g["parse_fail"].sum()
    expected = 76
    rows.append({
        "target_model": model,
        "persona_id": persona,
        "dialogues": n_dialogues,
        "sentences": n_sentences,
        "turns": n_turns,
        "parse_fail": n_parsefail,
        "expected_dialogues": expected,
        "coverage_pct": 100 * n_dialogues / expected,
        "parse_fail_pct": 100 * n_parsefail / n_sentences if n_sentences else 0.0,
    })

out = pd.DataFrame(rows)
out.to_csv("analysis/data/coverage.csv", index=False)
print(out)
