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
| 17 | Case study: a quadrotor, end to end | written |
| 18 | Case study: a quadrotor with a cable-suspended load | written |
| 19 | Case study: carrying a payload together | written |
| 20 | Case study: legged robots | written |
| A | Notation and glossary | written |

## What each chapter comes with

- **Text** in `chapters/`, with figures generated from code in `figures/src/`.
- **A verification script** in `verify/`. It checks every identity in the chapter numerically, re-derives every equation of motion and linearization with [`geomech`](https://github.com/vkotaru/geomech), and computes every number the text quotes.
- **A notebook** in `notebooks/` that reproduces the chapter's examples and starts the exercises. Runs on Colab.
- **A skill** in `skills/`: a short procedure for applying the chapter to your own system, written so that a person or a coding agent can follow it.

Chapter 16 is the hub of the case studies. It has no notebook and no skill of its own: it comes with the **robot model library** in `case_studies/robots/`, one module per robot with one interface, described in `case_studies/README.md` and checked by `verify/robots.py`.

Appendix A (`chapters/appA_notation.md`) fixes every symbol, convention, term and code name. `terms.toml` holds the same decisions as rules, and `python tools/check_terms.py` enforces them on the chapters, the figure scripts, the notebook sources, the skills and the robot model library.

## Layout

```text
book.toml          title, chapter order, shared math macros
chapters/          chapter sources (Pandoc Markdown)
figures/src/       one script per chapter; writes figures/*.svg
verify/            one script per chapter, and robots.py for the library; each must exit 0
notebooks/src/     notebook sources; built to notebooks/*.ipynb
skills/            one skill per chapter (all but Chapter 16)
case_studies/      the robot model library (case_studies/robots) and its README, read by Chapter 16
tools/             build script, terminology check, notebook builder, figure kit
docs/              the built site (GitHub Pages serves this directory)
AUTHORING.md       how chapters are written: voice, markup, figures
terms.toml         vocabulary and notation rules (python tools/check_terms.py)
```

## Build

Requires Python 3.11 or later (tested on 3.13), Pandoc 3 and Node.js.

```bash
pip install numpy scipy sympy matplotlib nbformat nbclient ipykernel geomech
```

```bash
npm install --prefix tools
```

```bash
npx --prefix tools playwright install chromium
```

Chromium is needed only for the PDF. `python tools/build.py --no-pdf` skips it and keeps the existing `docs/book.pdf`.

The next two steps are slow: about half an hour each. The figure scripts for Chapters 12 and 17 to 20 cache simulation results (Chapter 12 in `$CH12_CACHE`, default `/tmp/shots`; the others in `verify/`, which git ignores), so later runs are faster.

```bash
for f in figures/src/ch*.py; do python "$f"; done
```

```bash
for f in verify/ch*.py verify/robots.py; do python "$f"; done
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
