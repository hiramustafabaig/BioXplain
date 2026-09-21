"""Build configs/feature_universe.tsv (decision D5).

Universe = GPL570 probes that (a) map to exactly one gene, are not AFFX controls and have a symbol, and
(b) are present in BOTH series matrices.

Separation note: GSE65194 is opened here ONLY to read its list of probe IDs (structure). No expression
value of GSE65194 is read. The output file is tracked in git so that discovery runs never need to open the
external cohort at all.
"""
from __future__ import annotations

import json
import pathlib

from bioxplain.data.geo import read_gpl_annotation, read_probe_ids, usable_probe_table
from bioxplain.utils.provenance import verify_file

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "configs/data_manifest.json").read_text())["files"]
    for key in ("GSE42568", "GSE65194", "GPL570"):
        verify_file(ROOT / manifest[key]["path"], manifest[key]["sha256"])
    ann = read_gpl_annotation(ROOT / manifest["GPL570"]["path"])
    usable = usable_probe_table(ann)
    p_disc = set(read_probe_ids(ROOT / manifest["GSE42568"]["path"]))
    p_ext = set(read_probe_ids(ROOT / manifest["GSE65194"]["path"]))
    universe = usable[usable.probe_id.isin(p_disc & p_ext)].sort_values("probe_id").reset_index(drop=True)
    out = ROOT / "configs/feature_universe.tsv"
    universe.to_csv(out, sep="\t", index=False)
    print(f"usable probes: {len(usable)} | in both series: {len(universe)} | genes: {universe.gene_symbol.nunique()}"
          f" | entrez ids: {universe.entrez_id.nunique()}")


if __name__ == "__main__":
    main()
