# Authoring guide

Read `chapters/ch01_why_coordinate_free.md` first. It is the reference for voice, structure and markup. This file states the rules it follows.

## Who we write for

Advanced undergraduates through senior PhD students in robotics and control. They know linear algebra, rigid-body dynamics and classical control. They do not know differential geometry. Every geometric idea is introduced at the moment a robot needs it.

## How a chapter is built

1. **Open with a puzzle.** A concrete question the reader thinks they can answer, and cannot. No roadmap paragraph, no learning objectives, no thesis statement.
2. **Concrete before general.** The circle before the sphere, the sphere before the manifold. A picture or a number before the definition. State the theorem only after the reader could have guessed it.
3. **Thought experiments carry the hard ideas.** Use a `thought` block for anything a reader can do in their head or with their hands. At least two per chapter.
4. **Worked examples have real numbers.** Every number quoted in the text is produced by `verify/chNN.py`.
5. **Each section ends by creating the need for the next one.** A question, an unexplained piece, a failure of what was just built.
6. **Be honest about limits.** Say what a result does not give you. Use a `warning` block for traps that cost people flight hardware.
7. **Close with exercises and sources.** 5 to 8 exercises that ask for a derivation, an experiment or a counterexample. Then "Notes and sources": who did what, in two or three sentences, and a reference list.

Target length is 3000 to 4500 words of prose, not counting display math or the reference list. Cut anything that does not help the reader understand or do something.

## Prose rules

- Short sentences. One idea each. If a sentence has three clauses, split it.
- No em-dashes. Use a full stop, a colon or parentheses.
- Plain verbs. Never "leverage", "delve", "crucial", "pivotal", "robust framework", "it is worth noting", "in this chapter we will".
- Do not announce what you are about to say. Say it.
- Address the reader as "you". Use "we" for shared derivation steps.
- Define a term once, in bold, at first use. Do not rename it later.
- American spelling throughout: linearization, modeling, stabilize, center, behavior, maneuver, favorite, neighbor. This holds in code identifiers and plot labels too.
- No claims you have not checked. No invented quotes, numbers or citations. Cite only works you can identify precisely. If a source PDF is available, check the claim against it.
- Never mention how the book was produced, which tools generated it, or what any layout was modelled on.

## Markup

Chapters are Pandoc Markdown. No level-1 headings: the title comes from the front matter.

```markdown
---
title: The manifolds robots live on
short: Manifolds
id: ch-manifolds
summary: One sentence for the contents page.
notebook: notebooks/ch02_name.ipynb
verify: verify/ch02.py
epigraph: A verified quotation that fits the chapter, two lines at most.
epigraph_source: Author, *Work* (year)
---

Opening puzzle, straight in.

## Section title {#sec2-name}

::: {.definition #def2-name name="Short title"}
Body. Display math goes on its own lines.
:::
```

Block types: `definition`, `theorem`, `proposition`, `lemma`, `corollary` share one counter. `example`, `exercise`, `thought` each have their own. `remark`, `warning` and `proof` are unnumbered. `name="..."` is optional and may contain inline math.

Numbering and cross-references are automatic.

| Thing | How to label it | How to refer to it | Renders as |
| :-- | :-- | :-- | :-- |
| Equation | `\label{eq2-name}` inside `$$...$$` | `[](#eq2-name)` | (2.4) |
| Block | `#thm2-name` in the div attributes | `[](#thm2-name)` | Theorem 2.3 |
| Figure | `![Caption.](figures/ch02-name.svg){#fig2-name}` | `[](#fig2-name)` | Figure 2.1 |
| Section | `{#sec2-name}` on the heading | `[](#sec2-name)` | Section 2.2 |
| Table | wrap in `::: {.table #tbl2-name}` | `[](#tbl2-name)` | Table 2.1 |
| Chapter | `id:` in front matter | `[](#ch-manifolds)` | Chapter 2 |

Rules that keep the build green:

- Prefix every label with the chapter number (`eq2-`, `thm2-`, `fig2-`, `sec2-`). Labels are global.
- A reference already carries its name: write `[](#ch-manifolds)`, not `Chapter [](#ch-manifolds)`.
- No math in section headings. They also appear in the sidebar.
- Code listings are at most 84 characters wide. Longer lines overflow the column.
- A `warning` block is headed "Pitfall". Refer to it that way in prose.
- Only label equations you refer to. Unlabelled equations are unnumbered.
- A figure is an image alone in its paragraph, with a caption that says what to look at.
- Sidenotes are footnotes: `^[Short aside.]`. Keep them to one or two sentences.
- Unnumbered closing sections: `## Exercises {#sec2-exercises .unnumbered}` and `## Notes and sources {#sec2-notes .unnumbered}`.
- Inside math write degrees as `^\circ`, never the `°` character. In prose `°` is fine.
- Math is KaTeX. A LaTeX error fails the build. Use `\begin{bmatrix}`, `\begin{aligned}`, `\tfrac`, `\operatorname`. One `\label` per display.
- Shared macros are in `book.toml`: `\R`, `\SO`, `\so`, `\SE`, `\se`, `\T` (transpose, as in `R\T`), `\tr`, `\quat`, `\Ad`, `\ad`, `\dd`, `\ddt`.

Chapter ids: `ch-why` (1), `ch-manifolds` (2), `ch-variations` (3), `ch-dynamics` (4), `ch-symbolic` (5), `ch-tracking` (6), `ch-linearization` (7), `ch-robust` (8), `ch-mpc` (13), `app-notation` (Appendix A). Labels in the appendix use the prefix `A` (`secA-`, `tblA-`).

## Notation and terminology

`chapters/appA_notation.md` is the single source of truth. It fixes every symbol, every convention (frames, signs, orderings, the direction of the cable vector) and every technical term, and it says which variants are not used. `terms.toml` holds the same decisions as machine-checkable rules, and `python tools/check_terms.py` enforces them on chapters, figure scripts, notebook sources and skills.

- Read the appendix before writing. Use its symbols and its terms. Do not introduce alternatives.
- A new symbol or term is added to the appendix, and a rule for its variants to `terms.toml`, before it is used in a chapter.
- A symbol means one thing in the whole book. If a letter you want is taken, choose another. Do not reuse it locally.
- When a later chapter restates an equation from an earlier one, it uses the same form and cites it with `[](#label)`.
- Point to a specific definition, equation, figure or section, not to a whole chapter, whenever one exists.
- Notebooks and skills are read outside the book, where the macros of `book.toml` do not exist. They write the same notation in plain LaTeX: `R_d^\top R`, `\mathrm{SO}(3)`, `\mathbb{R}`, `\operatorname{tr}`.
- Code uses the function names `hat`, `vee`, `expm_so3`, `log_so3`, `Psi_R`, `Psi_q`, `e_R`, `e_Omega`, `e_q`, `e_omega`, with the actual configuration before the desired one: `e_R(R, Rd)`. Desired quantities are `Rd`, `qd`, `Omegad`, `omegad`, and a time derivative in a code-style formula is `R_dot`, `Omega_dot`. Short local names for velocities inside a cell (`Om`, `om`) are tolerated.

## Figures

Each chapter has one script, `figures/src/chNN.py`, with one function per figure, writing `figures/chNN-name.svg`. Use `tools/figkit.py`. Read its docstrings and `figures/src/ch01.py` before drawing.

- **Compute, never fake.** Every point is the projection of real 3D geometry through `Scene`. Hidden parts are found by the kit (`sphere_curve`, `point_visible`, `ribbons`).
- **One question per figure.** If the caption needs "and also", make two figures.
- **Stroke tiers.** Silhouette thickest, vectors next, construction lines thin, hidden lines dashed. Use the kit's constants.
- **Palette.** Ink for structure. Indigo for the main object or the "good" path. Terracotta for the contrasting object, error or failure. Teal for a third role. Pastel fills only. A quantity keeps its color across chapters: the trace error is indigo and the logarithm error is teal wherever they are plotted.
- **The same object is drawn the same way in every chapter.** A reference next to a motion that tracks it is a thin dashed gray line with a hollow marker, and the motion is solid. The boundary of a constraint is ink and its forbidden side pale terracotta. A quadrotor is drawn with `figkit.quadrotor`. The appendix lists these conventions under "References, targets and constraints in figures".
- **Labels use the book's symbols and terms exactly.** A label is checked by `tools/check_terms.py` like any sentence.
- **Labels sit next to what they name**, with the halo the kit applies. Math in labels uses `$...$`. Words use `fontfamily="Source Sans 3"` at size 9 to 10.
- Mathtext in labels cannot draw a postfix `^\wedge` or `\mathcal` well. Keep hats and calligraphic letters out of figure labels.
- **Animate what moves.** A thought experiment or a concept about motion gets an animation: `figkit.animate(make_frame, gif, strip)` writes a looping GIF for the site and a filmstrip of key frames for the PDF from one function. Mark it up as `![Caption.](figures/ch02-name-strip.png){#fig2-name anim="figures/ch02-name.gif"}`. An animation must earn the reader's attention: one idea, a clear beginning and end, eased motion, a short caption drawn in each frame that says what is happening now, and the same colours and strokes as the still figures. Give every frame the same canvas (fixed `xlim`, `ylim`). Keep each GIF under about 2.5 MB.
- **Open with a picture.** The first figure of a chapter is an evocative illustration of its central puzzle, placed right after the opening paragraphs.
- **Look at every figure** before you keep it. For an animation, look at the filmstrip and at least two frames of the GIF. Rasterise it and check for overlaps, clipped text, edge-on planes and labels that landed on a line.
- Aim for 4 to 6 figures per chapter. At least half should be geometric scenes, the rest data plots produced by simulation.

## Verification

`verify/chNN.py` checks the chapter. It uses `verify/common.py` (`Checker`, `hat`, `vee`, `expm_so3`, rotations).

- Every identity stated in the chapter gets a numerical check at random points.
- Every derivative formula is checked against finite differences.
- Every equation of motion and every linearization is derived a second time with `geomech` and compared numerically. Run `check_eom` and require an error below `1e-5`.
- Every number quoted in the text is computed there and stored with `ck.record(...)`.
- The script must exit 0. If a claim in an earlier draft fails, the text changes, not the check.

## Notebooks and skills

Each chapter has a notebook and a skill.

- **Notebook.** Source in `notebooks/src/chNN.py` (percent format, `# %%` and `# %% [markdown]` cells). Build and execute with `python tools/make_notebook.py notebooks/src/chNN.py notebooks/chNN_name.ipynb`. It must run top to bottom on a clean Colab runtime: install anything beyond NumPy, SciPy and Matplotlib in its first cell. It lets the reader reproduce the chapter's examples and start the exercises. Its title is `# Chapter N: Title`, its section titles follow the chapter's, and its last section is "Starting points for the exercises".
- **Skill.** `skills/chNN-name/SKILL.md` with valid front matter (`name`, `description`). It is a procedure a researcher or an agent follows to apply the chapter to their own system: when to use it, the steps, the checks that must pass, the failure modes. Formulas in a skill must match the chapter exactly. No narrative.

## Build

```bash
python figures/src/ch03.py
python verify/ch03.py
python tools/check_terms.py
python tools/build.py --only 3 --out /tmp/draft-ch3
python tools/build.py
```

The full build fails on any broken reference, duplicate label, missing figure or LaTeX error. `tools/check_terms.py` must report 0 violations before a chapter is merged.
