"""
make_samples.py — Regenerates the binary sample documents in inbox_v2/.

  python make_samples.py

Creates:
  scanned_invoice.png         a picture of an invoice (needs OCR)
  scanned_invoice.pdf         a PDF containing only an image of an invoice (needs OCR)
  bundle_three_invoices.pdf   three invoices on three pages of one text PDF (needs splitting)

The files are already in the repository; run this only to rebuild them.
It also provides write_text_pdf(), used by the tests to build small PDFs without
extra libraries.

LINE BY LINE (non-obvious lines):
  write_text_pdf(): a PDF is a list of numbered objects, then a table
  ("xref") giving the byte position of each. We write the objects, remember
  each one's offset, then write that table. Strings go inside (...) with
  backslash, ( and ) escaped.

  ImageFont.truetype(DejaVuSansMono, 26)
      A monospace font keeps the table columns aligned, like a real printout.
      If the font is not installed we fall back to Pillow's built-in font.

  img.save(path, "PDF", resolution=200)
      Pillow can write a PDF whose single page is the image, with NO text
      layer: exactly what a scanner produces.
"""

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "inbox_v2"

SCANNED_PNG = """INVOICE

Invoice Number: INV-2024-0555
Invoice Date: 2024-06-10
Due Date: 2024-07-10

From: Northwind Supplies Ltd.
Bill To: Fabrikam Ltd

Description        Qty   Unit Price    Amount
Pallet racking       2       650.00  1,300.00
Installation         1       200.00    200.00

Subtotal: 1,500.00
Tax (8%): 120.00
Total Due: 1,620.00
Currency: USD
Pay to IBAN: GB29 NWBK 6016 1331 9268 19""".splitlines()

SCANNED_PDF = """INVOICE No. 5503
Date: 12 June 2024
Due: 12 July 2024

Vendor: Globex Consulting GmbH
Customer: Initech Ltd

Item                       Qty  Rate      Amount
Consulting hours             8  150.00  1,200.00
Report writing               1  300.00    300.00

Subtotal: EUR 1,500.00
VAT 19%: EUR 285.00
TOTAL: EUR 1,785.00""".splitlines()

BUNDLE = [
    """INVOICE

Invoice Number: B-2001
Invoice Date: 2024-06-14
Due Date: 2024-07-14

From: Northwind Supplies Ltd.
Bill To: Contoso Labs Inc.

Label printer    2    150.00    300.00

Subtotal: 300.00
Tax: 0.00
Total Due: 300.00
Currency: USD""".splitlines(),
    """INVOICE

Invoice Number: B-2002
Invoice Date: 2024-06-14
Due Date: 2024-07-14

Vendor: Globex Consulting GmbH
Customer: Initech Ltd

Consulting hours    5    150.00    750.00

Subtotal: EUR 750.00
VAT 19%: EUR 142.50
TOTAL: EUR 892.50""".splitlines(),
    """INVOICE

Invoice Number: B-2003
Invoice Date: 2024-06-15
Due Date: 2024-07-15

From: Northwind Supplies Ltd.
Bill To: Contoso Labs Inc.

Toner cartridges    10    40.00    400.00

Subtotal: 400.00
Tax: 0.00
Total Due: 400.00
Currency: USD""".splitlines(),
]


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def render_image(lines: List[str], size: int = 26) -> Image.Image:
    line_h = int(size * 1.45)
    width = max(1100, int(size * 0.62 * max(len(l) for l in lines)) + 120)
    img = Image.new("RGB", (width, 80 + line_h * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    font = _font(size)
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * line_h), line, fill="black", font=font)
    return img


def write_text_pdf(path, pages: List[List[str]]) -> None:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    n = len(pages)
    objects = ["<< /Type /Catalog /Pages 2 0 R >>",
               f"<< /Type /Pages /Kids [{' '.join(f'{3 + 2 * i} 0 R' for i in range(n))}] /Count {n} >>"]
    font_obj = 3 + 2 * n
    for i, lines in enumerate(pages):
        stream = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(f"({esc(l)}) Tj T*" for l in lines) + " ET"
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                       f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode()
    Path(path).write_bytes(out)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    render_image(SCANNED_PNG).save(OUT / "scanned_invoice.png")
    render_image(SCANNED_PDF).save(OUT / "scanned_invoice.pdf", "PDF", resolution=200)
    write_text_pdf(OUT / "bundle_three_invoices.pdf", BUNDLE)
    print("Wrote scanned_invoice.png, scanned_invoice.pdf, bundle_three_invoices.pdf to", OUT)


if __name__ == "__main__":
    main()
