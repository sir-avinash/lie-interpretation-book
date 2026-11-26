"""Shared helpers for the chapter verification scripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def hat(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def expm_so3(w):
    """exp(hat(w)) by Rodrigues' formula."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + hat(w)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def random_rotation(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Checker:
    """Collects named checks, prints a table, and exits non-zero on failure."""

    def __init__(self, chapter: str):
        self.chapter = chapter
        self.rows = []
        self.values = {}

    def close(self, name, got, want, tol=1e-9):
        err = float(np.max(np.abs(np.asarray(got, float) - np.asarray(want, float))))
        self.rows.append((name, err <= tol, f"err {err:.1e}"))

    def true(self, name, cond, note=""):
        self.rows.append((name, bool(cond), note))

    def record(self, key, value):
        """Store a number that the chapter text quotes."""
        self.values[key] = value

    def finish(self):
        width = max(len(r[0]) for r in self.rows)
        for name, ok, note in self.rows:
            print(f"  {'ok  ' if ok else 'FAIL'} {name.ljust(width)}  {note}")
        if self.values:
            out = ROOT / "verify" / f"{self.chapter}_values.json"
            out.write_text(json.dumps(self.values, indent=2) + "\n")
            print(f"  quoted values -> {out.relative_to(ROOT)}")
        bad = [r for r in self.rows if not r[1]]
        print(f"{self.chapter}: {len(self.rows) - len(bad)}/{len(self.rows)} checks passed")
        raise SystemExit(1 if bad else 0)
