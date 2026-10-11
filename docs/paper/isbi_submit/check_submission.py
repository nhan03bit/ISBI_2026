#!/usr/bin/env python3
"""Pre-submission checks for the ISBI 2027 paper in this directory.

Usage (system python3, needs PyMuPDF):  python3 -I check_submission.py [--no-build] [--allow-author]
Exit code 0 = ready to upload; 1 = at least one hard failure (all failures are listed).

ISBI 2027 rules checked: technical content on pages 1-4; an optional page 5 may hold only the
references, "Compliance with Ethical Standards" and the acknowledgments; US Letter; fonts embedded.

  build    latexmk succeeds; no undefined reference/citation; no overfull box wider than 1pt
  results  generated/entropy_ablation_table.tex exists (written by analysis/entropy_ablation.py
           only when no run is missing), so the table holds real numbers
  TBD      no \\TBD left in the source
  AUTHOR   each \\AUTHOR{...} (a statement the authors must confirm) is listed; a failure
           unless --allow-author
  pages    at most 5 pages, US Letter
  layout   every \\label (sections, equations, figures, tables) and the end-of-technical marker
           at the end of the Conclusion is on page <= 4, except the ethics/acknowledgment sections
  fonts    every font embedded, no Type 3 font
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "paper.tex"
MAX_PAGES, LAST_TECHNICAL_PAGE = 5, 4
AFTER_TECHNICAL = {"sec:ethics", "sec:acknowledgments"}
END_MARKER = "end-of-technical"
LETTER = (612.0, 792.0)


def strip_comments(src):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in src.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true", help="check the existing paper.pdf/aux/log")
    ap.add_argument("--allow-author", action="store_true", help="do not fail on \\AUTHOR items")
    args = ap.parse_args()
    fail, warn = [], []

    if not args.no_build:
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", TEX.name],
                           cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            fail.append("build: latexmk failed (see paper.log)")

    src = strip_comments(TEX.read_text())
    n_tbd = len(re.findall(r"\\TBD\b", src)) - len(re.findall(r"\\newcommand\{\\TBD\}", src))
    if n_tbd:
        fail.append(f"TBD: {n_tbd} \\TBD placeholder(s) left in paper.tex")
    authors = [m.group(1)[:90] for m in re.finditer(r"\\AUTHOR\{(.*)", src)]  # the definition is \AUTHOR}
    authors += [m.group(1)[:90] for m in re.finditer(r"^%\s*CONFIRM:\s*(.*)", TEX.read_text(), re.M)]
    for a in authors:
        (warn if args.allow_author else fail).append(f"AUTHOR: unconfirmed statement: {a}...")
    if not (HERE / "generated" / "entropy_ablation_table.tex").exists():
        fail.append("results: generated/entropy_ablation_table.tex missing (run analysis/entropy_ablation.py)")

    log = (HERE / "paper.log").read_text(errors="replace")
    for m in re.finditer(r"LaTeX Warning: (Reference|Citation) `([^']*)' .*undefined", log):
        fail.append(f"build: undefined {m.group(1).lower()} {m.group(2)}")
    for m in re.finditer(r"Overfull \\([hv])box \((\d+(?:\.\d+)?)pt too (?:wide|high)\)([^\n]*)", log):
        if float(m.group(2)) > 1.0:
            where = re.search(r"lines? (\d+)", m.group(3))
            fail.append(f"build: overfull \\{m.group(1)}box {m.group(2)}pt"
                        + (f" at paper.tex line {where.group(1)}" if where else ""))
    if "Rerun to get" in log:
        warn.append("build: LaTeX asks for a rerun")
    blg = HERE / "paper.blg"
    if blg.exists():
        for line in blg.read_text(errors="replace").splitlines():
            if line.startswith("Warning--"):
                warn.append(f"bibtex: {line}")

    aux = (HERE / "paper.aux").read_text(errors="replace")
    labels = {m.group(1): m.group(2) for m in
              re.finditer(r"\\newlabel\{([^}]*)\}\{\{[^}]*\}\{([^}]*)\}", aux)}
    if END_MARKER not in labels:
        fail.append(f"layout: \\label{{{END_MARKER}}} missing at the end of the Conclusion")
    for name, page in sorted(labels.items(), key=lambda kv: kv[0]):
        if name in AFTER_TECHNICAL:
            continue
        if not page.isdigit() or int(page) > LAST_TECHNICAL_PAGE:
            fail.append(f"layout: {name} is on page {page} (technical content must end on page "
                        f"{LAST_TECHNICAL_PAGE})")

    import pymupdf
    doc = pymupdf.open(HERE / "paper.pdf")
    if len(doc) > MAX_PAGES:
        fail.append(f"pages: {len(doc)} pages (max {MAX_PAGES})")
    for i, page in enumerate(doc, 1):
        w, h = page.rect.width, page.rect.height
        if abs(w - LETTER[0]) > 1 or abs(h - LETTER[1]) > 1:
            fail.append(f"pages: page {i} is {w:.0f}x{h:.0f} pt, not US Letter")
    fonts = {}
    for page in doc:
        for xref, ext, ftype, basefont, *_ in page.get_fonts(full=True):
            fonts[basefont] = (ext, ftype)
    for name, (ext, ftype) in sorted(fonts.items()):
        if ftype == "Type3":
            fail.append(f"fonts: Type 3 font {name}")
        elif ext == "n/a":
            fail.append(f"fonts: {name} ({ftype}) not embedded")

    size_mb = (HERE / "paper.pdf").stat().st_size / 2**20
    end_page = labels.get(END_MARKER, "?")
    print(f"paper.pdf: {len(doc)} pages, technical content ends on page {end_page}, "
          f"{len(fonts)} fonts, {size_mb:.2f} MB")
    for w in warn:
        print("WARN ", w)
    for f in fail:
        print("FAIL ", f)
    print("READY" if not fail else f"NOT READY ({len(fail)} failure(s))")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
