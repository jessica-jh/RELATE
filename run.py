#!/usr/bin/env python3
"""RELATE Framework runner.

Corpus build (once; freeze under data/corpus/):
    python run.py --config configs/corpus.yaml --stage stimulus

Experiment (reads frozen corpus, writes to outputs/<run>/):
    python run.py --config configs/experiment/pilot.yaml --stage dialogue
    python run.py --config configs/experiment/pilot.yaml --stage all
"""
import argparse
import sys

from relate.util import load_config, load_env, stages_for_config, preflight_check
from relate.stimulus import build_stimuli
from relate.dialogue import run_dialogues
from relate.aggregate import aggregate


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/pilot.yaml")
    ap.add_argument("--stage", default="all",
                    choices=["all", "stimulus", "dialogue", "judge", "aggregate", "report"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        stages = stages_for_config(cfg, args.stage)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        preflight_check(cfg, stages)
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

    for s in stages:
        print(f"\n=== stage: {s} ===")
        if s == "stimulus":
            build_stimuli(cfg)
        elif s == "dialogue":
            run_dialogues(cfg)
        elif s == "judge":
            print("error: judge stage not available in this branch (judge.py removed)",
                  file=sys.stderr)
            sys.exit(1)
        elif s == "aggregate":
            aggregate(cfg)
        elif s == "report":
            print("error: report stage not available in this branch (report.py removed)",
                  file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
