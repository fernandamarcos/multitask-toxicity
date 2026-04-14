"""
Interactive Molecule Viewer
============================
Reads a CSV with columns: bit, mean_abs_attr, example_smiles, bit_smiles, svg
and launches a local web app to browse the molecules.

Requirements:
    pip install pandas rdkit flask

Usage:
    python molecule_viewer.py --csv your_file.csv
    python molecule_viewer.py --csv your_file.csv --port 5050
"""

import argparse
import io
import base64
import pandas as pd
from flask import Flask, render_template_string, jsonify

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ── HTML template ─────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Molecule Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f5f5f0; color: #1a1a1a; }

  header {
    background: #fff;
    border-bottom: 1px solid #e0e0d8;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  header h1 { font-size: 18px; font-weight: 500; }
  header span { font-size: 13px; color: #888; }

  .controls {
    padding: 12px 24px;
    background: #fff;
    border-bottom: 1px solid #e0e0d8;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }
  .controls input {
    border: 1px solid #d0d0c8;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    width: 240px;
    outline: none;
  }
  .controls input:focus { border-color: #7f77dd; }
  .controls label { font-size: 13px; color: #555; }
  .controls select {
    border: 1px solid #d0d0c8;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    outline: none;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    padding: 20px 24px;
  }

  .card {
    background: #fff;
    border: 1px solid #e0e0d8;
    border-radius: 10px;
    overflow: hidden;
    cursor: pointer;
    transition: box-shadow 0.15s, transform 0.15s;
  }
  .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); transform: translateY(-2px); }

  .card-img {
    background: #fafaf8;
    border-bottom: 1px solid #e0e0d8;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px;
    min-height: 180px;
  }
  .card-img img { max-width: 100%; height: auto; }
  .card-img svg { max-width: 160px; height: auto; }

  .card-body { padding: 12px 14px; }
  .badge {
    display: inline-block;
    background: #eeedfe;
    color: #3c3489;
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 8px;
  }
  .row { display: flex; justify-content: space-between; font-size: 12px; color: #666; margin-top: 4px; }
  .smiles {
    font-family: monospace;
    font-size: 11px;
    color: #555;
    word-break: break-all;
    margin-top: 6px;
    background: #f5f5f0;
    padding: 4px 6px;
    border-radius: 4px;
  }

  .overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.4);
    z-index: 100;
    align-items: center;
    justify-content: center;
  }
  .overlay.open { display: flex; }
  .modal {
    background: #fff;
    border-radius: 12px;
    max-width: 600px;
    width: 90%;
    padding: 24px;
    position: relative;
    max-height: 90vh;
    overflow-y: auto;
  }
  .modal-close {
    position: absolute; top: 14px; right: 14px;
    background: none; border: none; font-size: 20px; cursor: pointer; color: #888;
  }
  .modal h2 { font-size: 16px; font-weight: 500; margin-bottom: 16px; }
  .modal-imgs { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
  .modal-img-box {
    flex: 1; min-width: 160px;
    background: #fafaf8;
    border: 1px solid #e0e0d8;
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }
  .modal-img-box span { font-size: 11px; color: #888; }
  .modal-img-box img, .modal-img-box svg { max-width: 140px; }
  .detail-table { width: 100%; font-size: 13px; border-collapse: collapse; }
  .detail-table td { padding: 6px 8px; border-bottom: 1px solid #f0f0e8; }
  .detail-table td:first-child { color: #888; width: 140px; }
  .empty { text-align: center; padding: 60px; color: #888; font-size: 14px; }
</style>
</head>
<body>

<header>
  <h1>Molecule Viewer</h1>
  <span id="count-label">Loading...</span>
</header>

<div class="controls">
  <input id="search" placeholder="Search SMILES or bit..." oninput="filter()">
  <label>Sort by:
    <select id="sort" onchange="filter()">
      <option value="bit">Bit number</option>
      <option value="maa_asc">MAA ascending</option>
      <option value="maa_desc">MAA descending</option>
    </select>
  </label>
</div>

<div class="grid" id="grid"></div>

<div class="overlay" id="overlay" onclick="closeModal(event)">
  <div class="modal" id="modal">
    <button class="modal-close" onclick="closeOverlay()">x</button>
    <h2 id="modal-title">Bit -</h2>
    <div class="modal-imgs" id="modal-imgs"></div>
    <table class="detail-table" id="modal-table"></table>
  </div>
</div>

<script>
let molecules = [];

fetch('/api/molecules')
  .then(r => r.json())
  .then(data => { molecules = data; filter(); });

function filter() {
  const q = document.getElementById('search').value.toLowerCase();
  const sort = document.getElementById('sort').value;

  let filtered = molecules.filter(m =>
    String(m.bit).includes(q) ||
    (m.example_smiles || '').toLowerCase().includes(q) ||
    (m.bit_smiles || '').toLowerCase().includes(q)
  );

  if (sort === 'bit') filtered.sort((a, b) => a.bit - b.bit);
  else if (sort === 'maa_asc') filtered.sort((a, b) => a.mean_abs_attr - b.mean_abs_attr);
  else if (sort === 'maa_desc') filtered.sort((a, b) => b.mean_abs_attr - a.mean_abs_attr);

  render(filtered);
}

function render(data) {
  const grid = document.getElementById('grid');
  document.getElementById('count-label').textContent = data.length + ' molecule(s)';

  if (!data.length) {
    grid.innerHTML = '<div class="empty">No molecules match your search.</div>';
    return;
  }

  grid.innerHTML = data.map(m => `
    <div class="card" onclick="openModal(${m._idx})">
      <div class="card-img">${m.example_img_html}</div>
      <div class="card-body">
        <div class="badge">Bit ${m.bit}</div>
        <div class="row">
          <span>Mean abs. attribution</span>
          <strong>${parseFloat(m.mean_abs_attr).toFixed(4)}</strong>
        </div>
        <div class="smiles" title="Bit SMILES">${m.bit_smiles || '-'}</div>
      </div>
    </div>
  `).join('');
}

function openModal(idx) {
  const m = molecules.find(x => x._idx === idx);
  if (!m) return;

  document.getElementById('modal-title').textContent = 'Bit ' + m.bit;
  document.getElementById('modal-imgs').innerHTML = `
    <div class="modal-img-box">
      ${m.example_img_html}
      <span>Example molecule</span>
    </div>
    ${m.bit_img_html ? `
    <div class="modal-img-box">
      ${m.bit_img_html}
      <span>Bit fragment</span>
    </div>` : ''}
  `;
  document.getElementById('modal-table').innerHTML = `
    <tr><td>Bit</td><td>${m.bit}</td></tr>
    <tr><td>Mean abs. attr.</td><td>${parseFloat(m.mean_abs_attr).toFixed(6)}</td></tr>
    <tr><td>Example SMILES</td><td style="font-family:monospace;font-size:11px;word-break:break-all">${m.example_smiles || '-'}</td></tr>
    <tr><td>Bit SMILES</td><td style="font-family:monospace;font-size:11px;word-break:break-all">${m.bit_smiles || '-'}</td></tr>
  `;
  document.getElementById('overlay').classList.add('open');
}

function closeModal(e) {
  if (e.target === document.getElementById('overlay')) closeOverlay();
}
function closeOverlay() {
  document.getElementById('overlay').classList.remove('open');
}
</script>
</body>
</html>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def smiles_to_png_b64(smiles, size=(200, 200)):
    if not RDKIT_AVAILABLE or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def img_html(smiles, fallback_svg=None):
    b64 = smiles_to_png_b64(smiles)
    if b64:
        return f'<img src="data:image/png;base64,{b64}" alt="{smiles}">'
    if fallback_svg:
        svg = str(fallback_svg).strip()
        start = svg.find("<svg")
        if start != -1:
            return svg[start:]
    return f'<span style="font-size:11px;color:#888">{smiles or "-"}</span>'


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
_molecules = []


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/molecules")
def api_molecules():
    return jsonify(_molecules)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)

    required = {"bit", "mean_abs_attr", "example_smiles", "bit_smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    records = []
    for i, row in df.iterrows():
        svg_col = row.get("svg") if "svg" in df.columns else None

        records.append({
            "_idx": int(i),
            "bit": int(row["bit"]),
            "mean_abs_attr": float(row["mean_abs_attr"]),
            "example_smiles": str(row["example_smiles"]) if pd.notna(row["example_smiles"]) else "",
            "bit_smiles": str(row["bit_smiles"]) if pd.notna(row["bit_smiles"]) else "",
            "example_img_html": img_html(
                str(row["example_smiles"]) if pd.notna(row["example_smiles"]) else "",
                fallback_svg=svg_col if pd.notna(svg_col) else None,
            ),
            "bit_img_html": img_html(
                str(row["bit_smiles"]) if pd.notna(row["bit_smiles"]) else ""
            ),
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Interactive molecule viewer")
    parser.add_argument("--csv", required=True, help="Path to your CSV file")
    parser.add_argument("--port", type=int, default=5000, help="Port (default 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1)")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    global _molecules
    _molecules = load_csv(args.csv)
    print(f"Loaded {len(_molecules)} molecule(s).")

    if not RDKIT_AVAILABLE:
        print("Warning: RDKit not found - falling back to inline SVG from the CSV.")
        print("   To enable live rendering:  pip install rdkit")

    url = f"http://{args.host}:{args.port}"
    print(f"\n  Open your browser at: {url}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
