# CS336 — Language Modeling from Scratch

Workspace for self-studying Stanford CS336.

## Layout

- `assignments/assignment1-basics/` — personal implementation for Assignment 1
- `assignments/assignment2-systems/` — official Assignment 2 Systems starter and tests
- `assignments/assignment3-scaling/` — reserved for the Assignment 3 starter repository
- `assignments/assignment4-data/` — reserved for the Assignment 4 starter repository
- `assignments/assignment5-alignment/` — reserved for the Assignment 5 starter repository
- `notes/` — personal lecture and study notes
- `scratch/` — experiments that should not become assignment submissions

The assignments live in this monorepo, while each assignment keeps its own `uv` environment and lockfile.

## Assignment workflow

```bash
cd /Users/ning/Documents/CS336/assignments/assignment1-basics
uv run pytest
uv run python -m <module>
```

For Assignment 2:

```bash
cd /Users/ning/Documents/CS336/assignments/assignment2-systems
uv run pytest
```

Keep downloaded datasets, checkpoints, and generated outputs local; do not commit them.
