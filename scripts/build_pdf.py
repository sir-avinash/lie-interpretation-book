"""Pure-Python PDF builder for 'A Lie-Interpretation of Robotics and Control'.

Generates a crisp, multi-page review PDF directly from the repository Markdown
chapters without external TeX/browser dependencies.
"""

from __future__ import annotations

import re
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PDF = ROOT / "A_Lie_Interpretation_of_Robotics_and_Control.pdf"

SOURCES = [
    ("Front Matter: Overview", ROOT / "README.md"),
    ("Master Notation Standard", ROOT / "NOTATION_STANDARD.md"),
    ("Table of Contents", ROOT / "TOC.md"),
    ("Chapter 1", ROOT / "chapters" / "ch01_why_coordinate_free.md"),
    ("Chapter 2", ROOT / "chapters" / "ch02_core_manifolds.md"),
    ("Chapter 3", ROOT / "chapters" / "ch03_variations_errors_transport.md"),
    ("Chapter 4", ROOT / "chapters" / "ch04_coordinate_free_dynamics.md"),
    ("Chapter 5", ROOT / "chapters" / "ch05_symbolic_derivation_geomech.md"),
    ("Chapter 7", ROOT / "chapters" / "ch07_variation_linearization_lqr.md"),
]

# Clean WinAnsi-safe replacements for mathematical symbols in Type1 PDF fonts
LATEX_REPLACEMENTS = [
    (r"\\mathbb\{R\}", "R"),
    (r"\\mathbb\{Z\}", "Z"),
    (r"\\mathbb\{N\}", "N"),
    (r"\\mathfrak\{so\}", "so"),
    (r"\\mathfrak\{se\}", "se"),
    (r"\\mathfrak\{g\}", "g"),
    (r"\\mathfrak\{q\}", "q"),
    (r"\\operatorname\{tr\}|\\operatorname\{Tr\}|\\mathrm\{Tr\}|\\mathrm\{tr\}", "Tr"),
    (r"\\operatorname\{det\}|\\det", "det"),
    (r"\\operatorname\{exp\}|\\exp", "exp"),
    (r"\\operatorname\{log\}|\\log", "log"),
    (r"\\operatorname\{Ad\}|\\mathbf\{Ad\}", "Ad"),
    (r"\\operatorname\{ad\}|\\mathbf\{ad\}", "ad"),
    (r"\\mathcal\{([A-Za-z0-9]+)\}", r"\1"),
    (r"\\mathbf\{([A-Za-z0-9_\\]+)\}", r"\1"),
    (r"\\text\{([^}]*)\}", r"\1"),
    (r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)"),
    (r"\\tfrac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)"),
    (r"\\dot\{([^}]*)\}", r"\1_dot"),
    (r"\\ddot\{([^}]*)\}", r"\1_ddot"),
    (r"\\hat\{([^}]*)\}", r"\1^"),
    (r"\\tilde\{([^}]*)\}", r"~\1"),
    (r"\\bar\{([^}]*)\}", r"\1_bar"),
    (r"\\big\(|\\Big\(", "("),
    (r"\\big\)|\\Big\)", ")"),
    (r"\\left|\\right", ""),
    (r"\\qquad|\\quad|\\,|\\;|\\!", " "),
    (r"\\tag\{([^}]*)\}", r"   (\1)"),
    (r"\\cdot", " . "),
    (r"\\times", " x "),
    (r"\\wedge", "^"),
    (r"\\vee", "v"),
    (r"\\otimes", "(x)"),
    (r"\\in\b", "in"),
    (r"\\notin\b", "not in"),
    (r"\\subset\b", "subset"),
    (r"\\subseteq\b", "subseteq"),
    (r"\\forall\b", "for all"),
    (r"\\exists\b", "exists"),
    (r"\\to\b|\\rightarrow\b", "->"),
    (r"\\mapsto\b", "|->"),
    (r"\\xrightarrow\{([^}]*)\}", r"--[\1]-->"),
    (r"\\Longrightarrow\b|\\implies\b", "=>"),
    (r"\\iff\b", "<=>"),
    (r"\\approx\b", "~="),
    (r"\\cong\b", "~="),
    (r"\\equiv\b", "==="),
    (r"\\leq\b|\\le\b", "<="),
    (r"\\geq\b|\\ge\b", ">="),
    (r"\\neq\b|\\ne\b", "!="),
    (r"\\pm\b", "+/-"),
    (r"\\infty\b", "inf"),
    (r"\\partial\b", "d"),
    (r"\\nabla\b", "nabla"),
    (r"\\sum\b", "Sum"),
    (r"\\prod\b", "Prod"),
    (r"\\int\b", "Int"),
    (r"\\perp\b", "_|_"),
    (r"\\|([^|]+)\\|", r"||\1||"),
    (r"\\Psi\b", "Psi"),
    (r"\\Omega\b", "Omega"),
    (r"\\omega\b", "omega"),
    (r"\\eta\b", "eta"),
    (r"\\xi\b", "xi"),
    (r"\\gamma\b", "gamma"),
    (r"\\theta\b", "theta"),
    (r"\\phi\b", "phi"),
    (r"\\psi\b", "psi"),
    (r"\\rho\b", "rho"),
    (r"\\tau\b", "tau"),
    (r"\\epsilon\b|\\varepsilon\b", "eps"),
    (r"\\delta\b", "delta"),
    (r"\\Delta\b", "Delta"),
    (r"\\Sigma\b", "Sigma"),
    (r"\\alpha\b", "alpha"),
    (r"\\beta\b", "beta"),
    (r"\\lambda\b", "lambda"),
    (r"\\begin\{aligned\}|\\end\{aligned\}|\\begin\{bmatrix\}|\\end\{bmatrix\}", ""),
    (r"\\\\", " ; "),
]

UNICODE_TO_ASCII = {
    "—": "--",
    "–": "-",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "ℝ": "R",
    "ℤ": "Z",
    "ℕ": "N",
    "∈": "in",
    "∉": "not in",
    "⊂": "subset",
    "≅": "~=",
    "≈": "~=",
    "≡": "===",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "±": "+/-",
    "×": " x ",
    "·": " . ",
    "∧": "^",
    "∨": "v",
    "→": "->",
    "↦": "|->",
    "⇒": "=>",
    "⇔": "<=>",
    "∀": "for all ",
    "∃": "exists ",
    "∂": "d",
    "∇": "nabla",
    "∑": "Sum",
    "∏": "Prod",
    "∫": "Int",
    "⊥": "_|_",
    "‖": "||",
    "⟨": "<",
    "⟩": ">",
    "⟪": "<<",
    "⟫": ">>",
    "Ψ": "Psi",
    "Ω": "Omega",
    "ω": "omega",
    "η": "eta",
    "ξ": "xi",
    "γ": "gamma",
    "θ": "theta",
    "φ": "phi",
    "ψ": "psi",
    "ρ": "rho",
    "τ": "tau",
    "ε": "eps",
    "δ": "delta",
    "Δ": "Delta",
    "Σ": "Sigma",
    "α": "alpha",
    "β": "beta",
    "λ": "lambda",
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "ⁿ": "^n",
    "ᵀ": "^T",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "½": "1/2",
    "✓": "[ok]",
    "−": "-",
    "": "",
    "": "",
}


def clean_text(s: str) -> str:
    # Strip markdown links [label](url) -> label
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # Strip inline bold/italic markers
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    # Convert LaTeX commands
    for pat, rep in LATEX_REPLACEMENTS:
        s = re.sub(pat, rep, s)
    # Remove remaining $ delimiters
    s = s.replace("$$", "").replace("$", "")
    # Clean stray braces from simple subscripts/superscripts like R_{eq} -> R_eq
    s = re.sub(r"_\{([^}]+)\}", r"_\1", s)
    s = re.sub(r"\^\{([^}]+)\}", r"^\1", s)
    for k, v in UNICODE_TO_ASCII.items():
        s = s.replace(k, v)
    # Collapse multiple spaces
    s = re.sub(r"[ \t]+", " ", s)
    # Keep printable ASCII only for Type1 PDF safety
    s = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in s)
    return s.strip()


def escape_pdf(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pages() -> list[list[tuple[str, float, float, str, tuple[float, float, float]]]]:
    """Return list of pages, each a list of (font, size, x, y, text, rgb) draw ops."""
    PAGE_W, PAGE_H = 612.0, 792.0
    MARGIN_L, MARGIN_R = 64.0, 64.0
    TOP_Y, BOT_Y = 724.0, 64.0
    MAX_W = PAGE_W - MARGIN_L - MARGIN_R  # 484 pt

    pages: list[list[tuple[str, float, float, float, str, tuple[float, float, float]]]] = []
    cur: list[tuple[str, float, float, float, str, tuple[float, float, float]]] = []
    y = TOP_Y
    cur_section = "A Lie-Interpretation of Robotics and Control"

    def new_page():
        nonlocal cur, y
        if cur:
            pages.append(cur)
        cur = []
        y = TOP_Y

    def add_line(text: str, font="F1", size=10.5, indent=0.0, leading=14.5, rgb=(0.12, 0.14, 0.17)):
        nonlocal y
        if y - leading < BOT_Y:
            new_page()
        cur.append((font, size, MARGIN_L + indent, y, text, rgb))
        y -= leading

    def add_wrapped(text: str, font="F1", size=10.5, indent=0.0, leading=14.5, rgb=(0.12, 0.14, 0.17)):
        approx_char_w = size * (0.56 if font == "F4" else 0.48)
        width_chars = max(30, int((MAX_W - indent) / approx_char_w))
        for line in textwrap.wrap(text, width=width_chars, break_long_words=True) or [""]:
            add_line(line, font=font, size=size, indent=indent, leading=leading, rgb=rgb)

    # Title Page
    y = 560.0
    add_line("A Lie-Interpretation of Robotics and Control", font="F3", size=22.0, leading=30.0, rgb=(0.11, 0.12, 0.54))
    add_line("Coordinate-Free Modeling, Geometric Control, and Physical AI on Manifolds", font="F2", size=13.0, leading=22.0, rgb=(0.32, 0.36, 0.42))
    y -= 18.0
    add_line("Review PDF Edition (Front Matter & Verified Chapters 1, 2, 3, 4, 5, 7)", font="F1", size=11.0, leading=18.0)
    add_line("Master Notation Standard: SO(3), S^2, SE(3), and geomech Symbolic Pipelines", font="F1", size=11.0, leading=18.0)
    new_page()

    for label, path in SOURCES:
        if not path.exists():
            continue
        cur_section = label
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        in_code = False

        if y < TOP_Y - 20:
            new_page()

        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                y -= 4.0
                continue
            if in_code:
                code_line = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in raw_line)
                add_wrapped(code_line, font="F4", size=8.5, indent=14.0, leading=11.5, rgb=(0.18, 0.20, 0.24))
                continue

            if not stripped or set(stripped) <= {"_", "-", "="}:
                y -= 5.0
                continue

            if stripped.startswith("# "):
                y -= 6.0
                add_wrapped(clean_text(stripped[2:]), font="F3", size=16.0, leading=21.0, rgb=(0.11, 0.12, 0.54))
                y -= 4.0
            elif stripped.startswith("## "):
                y -= 6.0
                add_wrapped(clean_text(stripped[3:]), font="F3", size=13.0, leading=17.5, rgb=(0.11, 0.12, 0.54))
                y -= 2.0
            elif stripped.startswith("### "):
                y -= 4.0
                add_wrapped(clean_text(stripped[4:]), font="F2", size=11.5, leading=15.5, rgb=(0.12, 0.14, 0.17))
            elif stripped.startswith("|") and set(stripped.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
                continue
            elif stripped.startswith("|"):
                cells = [clean_text(c) for c in stripped.strip("|").split("|")]
                add_wrapped(" | ".join(cells), font="F1", size=9.5, indent=8.0, leading=13.0, rgb=(0.20, 0.22, 0.26))
            elif stripped.startswith("* ") or stripped.startswith("- "):
                add_wrapped("* " + clean_text(stripped[2:]), font="F1", size=10.5, indent=12.0, leading=14.5)
            else:
                add_wrapped(clean_text(stripped), font="F1", size=10.5, indent=0.0, leading=14.5)

    if cur:
        pages.append(cur)

    # Stamp running headers & page numbers
    total = len(pages)
    for idx, ops in enumerate(pages, start=1):
        if idx > 1:
            ops.append(("F1", 8.5, 64.0, 758.0, "A Lie-Interpretation of Robotics and Control", (0.45, 0.48, 0.53)))
            ops.append(("F1", 8.5, 490.0, 36.0, f"Page {idx} of {total}", (0.45, 0.48, 0.53)))
    return pages


def write_pdf(pages: list[list[tuple]], out_path: Path) -> None:
    objects: list[bytes] = []

    def add_obj(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    # 1..4: Standard Type1 fonts
    f1 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>")
    f2 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>")
    f3 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    f4 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    font_dict = f"<< /F1 {f1} 0 R /F2 {f2} 0 R /F3 {f3} 0 R /F4 {f4} 0 R >>".encode("ascii")
    pages_placeholder_idx = add_obj(b"")  # 5

    page_ids: list[int] = []
    for ops in pages:
        stream_lines = ["BT"]
        for font, size, x, y, text, (r, g, b) in ops:
            stream_lines.append(f"{r:.2f} {g:.2f} {b:.2f} rg")
            stream_lines.append(f"/{font} {size:.1f} Tf")
            stream_lines.append(f"1 0 0 1 {x:.1f} {y:.1f} Tm")
            stream_lines.append(f"({escape_pdf(text)}) Tj")
        stream_lines.append("ET")
        raw_stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        comp = zlib.compress(raw_stream)
        c_id = add_obj(
            f"<< /Length {len(comp)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
            + comp
            + b"\nendstream"
        )
        p_id = add_obj(
            f"<< /Type /Page /Parent {pages_placeholder_idx} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font {font_dict.decode('ascii')} >> /Contents {c_id} 0 R >>".encode("ascii")
        )
        page_ids.append(p_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_placeholder_idx - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_placeholder_idx} 0 R >>".encode("ascii"))

    # Assemble PDF bytes with xref
    buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf.extend(f"{i} 0 obj\n".encode("ascii"))
        buf.extend(obj)
        buf.extend(b"\nendobj\n")

    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    out_path.write_bytes(bytes(buf))


if __name__ == "__main__":
    pages = build_pages()
    write_pdf(pages, OUT_PDF)
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size} bytes, {len(pages)} pages)")
