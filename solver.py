#!/usr/bin/env python3
"""Sudoku step-by-step solver.

Usage:
    python solver.py <input_image>

Reads a Sudoku puzzle image, applies the next solving step, and outputs
a step_NNN.png image and step_NNN.txt explanation file.
"""

import sys
from pathlib import Path

from ingestion import ingest_image
from models import Game
from strategies import UniquenessStrategy, NakedSingleStrategy, HiddenSingleStrategy, StepResult
from validation import validate
from output import render_image, render_text


# Strategies are applied in this order; stop at the first one that makes
# a change. Add new strategies here as they are implemented.
STRATEGIES = [
    UniquenessStrategy(),
    NakedSingleStrategy(),
    HiddenSingleStrategy(),
]


def find_next_step_number(output_dir: Path) -> int:
    """Auto-detect the next step number by scanning existing step files."""
    max_num = 0
    for f in output_dir.glob("step_*.png"):
        try:
            num = int(f.stem.split("_")[1])
            max_num = max(max_num, num)
        except (IndexError, ValueError):
            continue
    return max_num + 1


def solve_step(game: Game) -> StepResult | None:
    """Apply strategies in order until one makes a single change."""
    for strategy in STRATEGIES:
        result = strategy.apply(game)
        if result is not None:
            return result
    return None


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python solver.py <input_image>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Ingest the image
    print(f"Reading puzzle from {input_path}...")
    game = ingest_image(input_path)

    # Validate after ingestion
    errors = validate(game)
    if errors:
        print("Validation errors found after ingestion:")
        for err in errors:
            print(f"  - {err.description}")

    # Apply one solving step
    step_result = solve_step(game)

    # Validate after strategy
    if step_result:
        post_errors = validate(game)
        if post_errors:
            errors.extend(post_errors)

    # Determine output file names
    output_dir = Path(".")
    step_num = find_next_step_number(output_dir)
    img_path = output_dir / f"step_{step_num:03d}.png"
    txt_path = output_dir / f"step_{step_num:03d}.txt"

    # Render outputs
    render_image(game, step_result, errors if errors else None, img_path)
    render_text(step_result, errors if errors else None, txt_path)

    if step_result:
        print(f"Step {step_num}: {step_result.description}")
    else:
        print(f"Step {step_num}: Unable to proceed.")

    print(f"Output: {img_path}, {txt_path}")


if __name__ == "__main__":
    main()
