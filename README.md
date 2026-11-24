# A Manifold Introduction to Robotics and Control

A textbook on coordinate-free modeling, control, estimation and optimization for robots whose configurations live on manifolds: $\mathrm{SO}(3)$, $S^2$, $\mathrm{SE}(3)$ and their products.

It consolidates a body of research on geometric mechanics and control into one exposition with one notation, written for advanced undergraduates through PhD students in robotics and control. No differential geometry is assumed.

## Contents

| | Chapter | Status |
| :-- | :-- | :-- |
| 1 | Why coordinate-free? | written |
| 2 | The manifolds robots live on | written |
| 3 | Variations, errors and transport | written |
| 4 | Dynamics without coordinates | written |
| 5 | Symbolic derivation | written |
| 6 | Tracking control and almost-global stability | written |
| 7 | Variation-based linearization and LQR | written |
| 8 | Robust, adaptive and safety-critical control | written |
| 9 | Symmetry, fiber bundles and connections | planned |
| 10 | Differential flatness | written |
| 11 | Hybrid zero dynamics on manifolds | planned |
| 12 | Variational integrators and collocation | written |
| 13 | Error-state MPC and on-manifold solvers | written |
| 14 | Filtering on Lie groups | written |
| 15 | Equivariant learning and residual control | planned |
| 16 | How it all comes together | written |
| A | Notation and glossary | written |

## What each chapter comes with

- **Text** in `chapters/`, with figures generated from code in `figures/src/`.
- **A verification script** in `verify/`. It checks every identity in the chapter numerically, re-derives every equation of motion and linearization with [`geomech`](https://github.com/vkotaru/geomech), and computes every number the text quotes.
- **A notebook** in `notebooks/` that reproduces the chapter's examples and starts the exercises. Runs on Colab.
- **A skill** in `skills/`: a short procedure for applying the chapter to your own system, written so that a person or a coding agent can follow it.

## Layout

```text
book.toml          title, chapter order, shared math macros
chapters/          chapter sources (Pandoc Markdown)
figures/src/       one script per chapter; writes figures/*.svg
verify/            one script per chapter; must exit 0
notebooks/src/     notebook sources; built to notebooks/*.ipynb
skills/            one skill per chapter
case_studies/      the robot model library (case_studies/robots) used by Chapter 16
tools/             build script, cross-reference filter, figure kit
docs/              the built site (GitHub Pages serves this directory)
AUTHORING.md       how chapters are written: voice, markup, figures
terms.toml         vocabulary and notation rules (python tools/check_terms.py)
```

## Build

Requires Python 3.11+, Pandoc 3, and Node.js.

```bash
pip install numpy scipy matplotlib nbformat nbclient ipykernel geomech
```

```bash
npm install --prefix tools
```

```bash
for f in figures/src/ch*.py; do python "$f"; done
```

```bash
for f in verify/ch*.py; do python "$f"; done
```

```bash
python tools/check_terms.py
```

```bash
python tools/build.py
```

The build fails on a broken cross-reference, a duplicate label, a missing figure or a LaTeX error. Open `docs/index.html` to read the result.

## Contributing a chapter

Read `AUTHORING.md`, then `chapters/ch01_why_coordinate_free.md` as the worked example.
