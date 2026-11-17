"""Render submission/TECHNICAL_DOC.md to submission/out/<team>.pdf for the upload form.

    pip install markdown==3.7          # only this script needs it; not part of the labelling pipeline
    python submission/build_doc_pdf.py
    python submission/build_doc_pdf.py --html-only     # just the HTML, to look at in a browser

The Markdown stays the source: edit it, rerun this. Printing goes through headless Chrome
(--chrome to point at another Chromium). The PDF lands in submission/out/, which is git-ignored,
as is *.pdf everywhere in this repo.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #1d2128; margin: 0; }
h1 { font-size: 21pt; line-height: 1.2; margin: 0 0 4pt; color: #0e1a2b; letter-spacing: -0.01em; }
h1 + p { color: #5a6270; font-size: 9.5pt; margin: 0 0 14pt; padding-bottom: 10pt;
         border-bottom: 2px solid #0e1a2b; }
h2 { font-size: 13.5pt; color: #0e1a2b; margin: 18pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1px solid #d5d9e0; break-after: avoid; }
h3 { font-size: 11pt; color: #0e1a2b; margin: 12pt 0 4pt; break-after: avoid; }
p, ul, ol { margin: 0 0 7pt; }
li { margin-bottom: 2pt; }
a { color: #1f5fbf; text-decoration: none; }
code { font-family: Menlo, "SF Mono", Consolas, monospace; font-size: 8.8pt;
       background: #f1f3f6; padding: 0.5pt 3pt; border-radius: 3px; }
pre { background: #f1f3f6; border: 1px solid #e1e5eb; border-radius: 4px; padding: 7pt 9pt;
      font-size: 8.6pt; line-height: 1.4; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; font-size: 9pt;
        break-inside: auto; }
th { background: #0e1a2b; color: #fff; text-align: left; font-weight: 600; padding: 4pt 6pt; }
td { padding: 4pt 6pt; border-bottom: 1px solid #e1e5eb; vertical-align: top; }
tr { break-inside: avoid; }
tr:nth-child(even) td { background: #f7f8fa; }
.todo { background: #fff4d6; border-left: 3px solid #e0a400; padding: 1pt 4pt; }
"""


def team_name():
    with open(os.path.join(HERE, "submission.json")) as handle:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", json.load(handle)["team_name"].strip()).strip("_")


def to_html(md_text, title):
    md_text = re.sub(r"<!--.*?-->", "", md_text, flags=re.S)  # drafting notes stay out of the PDF
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    # open TODOs stay visible, so a draft cannot pass for the final document
    body = re.sub(r"(TODO[^<]*)", r'<span class="todo">\1</span>', body)
    return ("<!doctype html><html><head><meta charset='utf-8'><title>{}</title><style>{}</style>"
            "</head><body>{}</body></html>").format(html.escape(title), CSS, body)


def find_chrome(explicit):
    for candidate in ([explicit] if explicit else CHROME_CANDIDATES):
        path = candidate if os.path.isabs(candidate) else shutil.which(candidate)
        if path and os.path.exists(path):
            return path
    sys.exit("no Chrome/Chromium found; pass --chrome PATH, or use --html-only and print from a browser")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=os.path.join(HERE, "TECHNICAL_DOC.md"))
    parser.add_argument("--chrome")
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    name = team_name()
    with open(args.source, encoding="utf-8") as handle:
        md_text = handle.read()
    title_match = re.search(r"^# (.+)$", md_text, flags=re.M)
    page = to_html(md_text, title_match.group(1) if title_match else name)

    os.makedirs(OUT_DIR, exist_ok=True)
    html_path = os.path.join(OUT_DIR, name + ".html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(page)
    if args.html_only:
        print("-> " + os.path.relpath(html_path))
        return

    pdf_path = os.path.join(OUT_DIR, name + ".pdf")
    cmd = [find_chrome(args.chrome), "--headless", "--disable-gpu", "--no-pdf-header-footer",
           "--print-to-pdf=" + pdf_path, "file://" + html_path]
    if os.path.exists(pdf_path):
        os.remove(pdf_path)  # so a run that writes nothing cannot pass off the last PDF as new
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        cmd.insert(1, "--no-sandbox")  # Chrome refuses to run as root (containers) without it
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode or not os.path.isfile(pdf_path):
        sys.exit("Chrome failed (exit {}):\n{}\nFallback: --html-only, then print the HTML from a "
                 "browser.".format(result.returncode, (result.stderr or result.stdout).strip()[-2000:]))
    todos = len(re.findall(r"\bTODO\b", re.sub(r"<!--.*?-->", "", md_text, flags=re.S)))
    print("-> " + os.path.relpath(pdf_path))
    if todos:
        print("NOTE  {} TODO(s) still in the document; they are highlighted in the PDF".format(todos))


if __name__ == "__main__":
    main()
