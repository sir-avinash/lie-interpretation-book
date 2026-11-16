import json
from pathlib import Path

ROOT = Path("/tmp/lie_textbook_repo")
DOCS = ROOT / "docs"

def read_text(rel_path: str) -> str:
    p = ROOT / rel_path
    return p.read_text(encoding="utf-8") if p.exists() else ""

sections = [
    {
        "id": "overview",
        "part": "Front Matter",
        "label": "0. Overview & Notation",
        "verified": True,
        "markdown": read_text("README.md") + "\n\n---\n\n" + read_text("NOTATION_STANDARD.md") + "\n\n---\n\n" + read_text("TOC.md") + "\n\n```mermaid\nflowchart LR\n  A[Configuration Manifold M] --> B[Coordinate-Free Dynamics]\n  B --> C[Symmetry & Flatness Reduction]\n  C --> D[Geometric Control & LieIPM]\n  D --> E[InEKF & Physical AI]\n```\n"
    },
    {
        "id": "ch01",
        "part": "Part I: Foundations",
        "label": "Ch 1. Why Coordinate-Free?",
        "verified": True,
        "chapterFile": "ch01_why_coordinate_free.md",
        "notebook": "ch01_singularities_unwinding.ipynb",
        "skill": "ch01-manifold-diagnostics",
        "diagramKey": "s3_double_cover",
        "markdown": read_text("chapters/ch01_why_coordinate_free.md")
    },
    {
        "id": "ch02",
        "part": "Part I: Foundations",
        "label": "Ch 2. Core Manifolds",
        "verified": True,
        "chapterFile": "ch02_core_manifolds.md",
        "notebook": "ch02_lie_groups_product_manifolds.ipynb",
        "skill": "ch02-manifold-kinematics",
        "markdown": read_text("chapters/ch02_core_manifolds.md")
    },
    {
        "id": "ch03",
        "part": "Part I: Foundations",
        "label": "Ch 3. Variations & Transport",
        "verified": True,
        "chapterFile": "ch03_variations_errors_transport.md",
        "notebook": "ch03_variations_transport_maps.ipynb",
        "skill": "ch03-geometric-error-design",
        "diagramKey": "transport_map",
        "markdown": read_text("chapters/ch03_variations_errors_transport.md")
    },
    {
        "id": "ch04",
        "part": "Part II: Modeling",
        "label": "Ch 4. Coordinate-Free Dynamics",
        "verified": True,
        "chapterFile": "ch04_coordinate_free_dynamics.md",
        "notebook": "ch04_lagrange_dalembert_sim.ipynb",
        "skill": "ch04-manifold-dynamics-modeling",
        "markdown": read_text("chapters/ch04_coordinate_free_dynamics.md") + "\n\n```mermaid\nstateDiagram-v2\n  [*] --> TautMode: \"T > 0 on SE(3) x S^2\"\n  TautMode --> SlackMode: \"Guard S_z: T = 0\"\n  SlackMode --> TautMode: \"Guard S_n: ||x_Q - x_L|| = l\"\n```\n"
    },
    {
        "id": "ch07",
        "part": "Part III: Geometric Control",
        "label": "Ch 7. Variation Linearization & LQR",
        "verified": True,
        "chapterFile": "ch07_variation_linearization_lqr.md",
        "notebook": "ch07_variation_lqr_sim.ipynb",
        "skill": "ch07-variation-lqr-design",
        "markdown": read_text("chapters/ch07_variation_linearization_lqr.md")
    },
]

out_js = DOCS / "book_data.js"
out_js.write_text("window.BOOK_SECTIONS = " + json.dumps(sections, ensure_ascii=False) + ";\n", encoding="utf-8")
print(f"Wrote {out_js} ({out_js.stat().st_size} bytes) with {len(sections)} sections.")
