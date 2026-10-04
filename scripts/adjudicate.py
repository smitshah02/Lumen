"""Blinded human adjudication of claim support. Local only; calls no model.

    ./scripts/lumen research adjudicate sample                  # from the scorecard runs on disk
    ./scripts/lumen research adjudicate label  <sample.jsonl>   # you decide; the verifier's view is hidden
    ./scripts/lumen research adjudicate report <sample.jsonl>   # afterwards: human vs verifier

While labelling you see only the claim and the source it cites. The verifier's
verdict, whether the claim was flagged, and which test case it came from are
stored in the file but never displayed until `report`.

Sample files contain patient text. They are written under
~/Lumen_local_results, outside the repository, and are never sent anywhere.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evals import adjudication as adj  # noqa: E402

SCORECARD_DIR = Path.home() / "Lumen_local_results" / "scorecard"
KEYS = {"s": "SUPPORTED", "p": "PARTIALLY_SUPPORTED", "u": "UNSUPPORTED", "c": "UNCLEAR", "k": "SKIP"}
BAR = "=" * 78


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.chmod(0o600)


def cmd_sample(args) -> int:
    patterns = args.inputs or [str(SCORECARD_DIR / "claims-*.jsonl"), str(SCORECARD_DIR / "adjudication-*.jsonl")]
    files = sorted({f for pattern in patterns for f in glob.glob(str(Path(pattern).expanduser()))})
    files = [f for f in files if "sample" not in Path(f).name]
    if not files:
        print("no scorecard claim files found; run the scorecard first", file=sys.stderr)
        return 2
    rows = [dict(row, cohort=args.cohort) for f in files for row in read_jsonl(Path(f)) if "verifier_verdict" in row]
    sample = adj.build_sample(rows, target=args.target, seed=args.seed)
    out = Path(args.out).expanduser()
    if out.exists() and any(r.get("human_verdict") for r in read_jsonl(out)):
        print(f"refusing: {out} already holds human verdicts; choose another --out", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, sample)
    kinds: dict[str, int] = {}
    for row in sample:
        kinds[adj.source_kind(row)] = kinds.get(adj.source_kind(row), 0) + 1
    print(f"{len(files)} claim file(s), {len(rows)} claims read, {len(sample)} sampled (seed {args.seed})")
    print("by cited source type: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    if len(sample) < 20:
        print(f"note: only {len(sample)} distinct claims were available; fewer than the 20 intended")
    print(f"sample written to {out}\nnext: ./scripts/lumen research adjudicate label {out}")
    return 0


def show(row: dict, position: int, total: int) -> None:
    view = adj.blind_view(row)
    print("\n" + BAR)
    print(f"  CLAIM {position} of {total}")
    print(BAR)
    print(f"\n  CLAIM:\n    {view['claim']}\n")
    for src in view["sources"]:
        print(f"  CITED SOURCE [{src['label'] or 'none'}]  ({src['type']})")
        print("    " + (src["excerpt"] or "(the claim cites no source)").replace("\n", "\n    ") + "\n")
    print("  Does the cited source support the claim?")
    print("  [s] supported   [p] partially supported   [u] unsupported   [c] unclear   [k] skip   [q] quit")


def cmd_label(args, ask=input) -> int:
    path = Path(args.file).expanduser()
    rows = read_jsonl(path)
    todo = [r for r in rows if args.redo or not r.get("human_verdict")]
    print(f"{len(rows)} claims in the sample, {len(todo)} to label. Progress is saved after every answer.")
    for n, row in enumerate(todo, 1):
        show(row, n, len(todo))
        while True:
            key = ask("  > ").strip().lower()[:1]
            if key == "q":
                print(f"\nstopped. {sum(1 for r in rows if r.get('human_verdict'))} of {len(rows)} labelled.")
                return 0
            if key in KEYS:
                break
            print("  please enter s, p, u, c, k or q")
        row["human_verdict"] = KEYS[key]
        row["human_note"] = ask("  note (optional): ").strip()[:500]
        write_jsonl(path, rows)
    print(f"\nall {len(rows)} claims labelled.\nnext: ./scripts/lumen research adjudicate report {path}")
    return 0


def cmd_report(args) -> int:
    for name in args.files:
        path = Path(name).expanduser()
        rows = read_jsonl(path)
        summary = adj.summarize(rows)
        print(adj.render(summary, f"HUMAN ADJUDICATION vs LUMEN VERIFIER — {path.name}"))
        if summary["unlabelled"]:
            print(f"\nnote: {summary['unlabelled']} claim(s) are not labelled yet; the figures cover labelled claims only")
        path.with_suffix(".report.json").write_text(json.dumps(summary, indent=1))
        print(f"\nsaved: {path.with_suffix('.report.json')}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("inputs", nargs="*", help="scorecard claim files (default: every run under ~/Lumen_local_results/scorecard)")
    s.add_argument("--out", default=str(SCORECARD_DIR / "adjudication-sample.jsonl"))
    s.add_argument("--target", type=int, default=25)
    s.add_argument("--seed", type=int, default=adj.DEFAULT_SEED)
    s.add_argument("--cohort", default="fixture", help="label stored with the sample (fixture | holdout)")
    lab = sub.add_parser("label")
    lab.add_argument("file")
    lab.add_argument("--redo", action="store_true", help="ask again about claims that already have a verdict")
    r = sub.add_parser("report")
    r.add_argument("files", nargs="+")
    args = p.parse_args(argv)
    return {"sample": cmd_sample, "label": cmd_label, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
