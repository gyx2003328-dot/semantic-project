#!/usr/bin/env python3
"""Create leaderboard-style submission JSON from training_report.json."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="output/training_report.json", help="Path to training_report.json")
    parser.add_argument("--output", required=True, help="Output submission JSON path")
    parser.add_argument("--team", default="Team Alpha", help="Team/group name")
    parser.add_argument("--repo-url", default="https://github.com/xxxxxx.git", help="Private repo URL")
    args = parser.parse_args()

    report_path = Path(args.report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = json.loads(report_path.read_text())
    tm = report.get("test_metrics", {})

    submission = {
        "group_name": args.team,
        "project_private_repo_url": args.repo_url,
        "metrics": {
            "dice_score": round(float(tm.get("dice_score", 0.0)), 2),
            "miou": round(float(tm.get("miou", 0.0)), 2),
            "fwiou": round(float(tm.get("fwiou", 0.0)), 2),
        },
    }

    output_path.write_text(json.dumps(submission, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

