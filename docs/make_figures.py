"""Regenerate the README's paper-vs-code figures.

Paper side: equations cropped from arXiv:2610.33099 (downloaded here; needs `pdftoppm`).
Code side: the implementing functions rendered with Pygments; lines tagged `# Eq. (n)` are
highlighted, with the real file line numbers.

usage: python docs/make_figures.py        (needs: pip install pillow pygments)
"""
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
PDF_URL = "https://arxiv.org/pdf/2610.33099"

# (page, y0, y1) in PDF points, x range 100-508 pt: positions taken from `pdftotext -bbox-layout`
PAPER = {
    "paper_eq15_training.png": [(15, 80, 121)],
    "paper_eq3_6_8_guided_ddim.png": [(4, 384.6, 432), (5, 80, 254)],
    "paper_eq4_21_22_weights.png": [(4, 507, 566), (16, 80, 229)],
    "paper_eq18_wawl.png": [(15, 371, 487)],
    "paper_eq20_overlap.png": [(15, 671, 735)],
}

# (file, start regex, end regex, extra highlight substrings)
CODE = {
    "code_model_loss.png": ("eggpcb/model.py", r"def loss\(self", r"return \(\(pred - eps\)", []),
    "code_sample_reverse_step.png": ("eggpcb/sample.py", r"for t in range\(T, 0, -1\):", r"z = z0 \+ g", []),
    "code_guide_refine.png": ("eggpcb/sample.py", r"def refine\(self", r"return u.detach\(\), last_leg", []),
    "code_energies_wawl.png": ("eggpcb/energies.py", r"^def wawl\(", r"return total",
                               ["torch.exp(z - zmax", "sxe / se", "z = sign * x / gamma"]),
    "code_energies_overlap.png": ("eggpcb/energies.py", r"^def overlap\(", r"tot = \(ox \* oy \* same\)\.sum\(\)",
                                  ["ox = _sp(", "oy = _sp("]),
}


def paper_figures(dpi=200):
    S = dpi / 72
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "paper.pdf"
        req = urllib.request.Request(PDF_URL, headers={"User-Agent": "eggplacer-make-figures"})
        pdf.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        pages = {}
        for name, parts in PAPER.items():
            crops = []
            for page, y0, y1 in parts:
                if page not in pages:
                    subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png",
                                    str(pdf), str(Path(td) / f"p{page}")], check=True)
                    pages[page] = Image.open(next(Path(td).glob(f"p{page}-*.png")))
                crops.append(pages[page].crop((int(100 * S), int(y0 * S), int(508 * S), int(y1 * S))))
            W = max(c.width for c in crops); H = sum(c.height for c in crops) + 30 * (len(crops) - 1)
            out = Image.new("RGB", (W, H), "white"); y = 0
            for c in crops:
                out.paste(c, (0, y)); y += c.height + 30
            out.save(IMG / name)
            print("paper", name)


def code_figures():
    spans = {}
    for name, (path, sp, ep, extra) in CODE.items():
        lines = (ROOT / path).read_text().splitlines()
        a = next(i for i, l in enumerate(lines) if re.search(sp, l))
        b = next(i for i in range(a + 1, len(lines)) if re.search(ep, lines[i]))
        snippet = lines[a:b + 1]
        hl = [i + 1 for i, l in enumerate(snippet) if "Eq. (" in l or "App. D" in l or any(x in l for x in extra)]
        fmt = ImageFormatter(font_name="DejaVu Sans Mono", font_size=17, line_numbers=True, line_number_start=a + 1,
                             hl_lines=hl, hl_color="#fff1a8", style="default", line_number_bg="#f3f3f0",
                             line_number_fg="#898781", line_number_pad=10, image_pad=26, line_pad=4)
        (IMG / name).write_bytes(highlight("\n".join(snippet) + "\n", PythonLexer(), fmt))
        spans[name] = (path, a + 1, b + 1)
        print("code ", name, f"{path}#L{a + 1}-L{b + 1}")
    return spans


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    paper_figures()
    code_figures()
