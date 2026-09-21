"""python scripts/create_freeze.py <full_run_dir>

Writes docs/freeze/discovery_freeze.json (+ .md) and results/freeze/frozen_genes.csv. Requires a clean tracked working tree.
Uses discovery data only.
"""
from __future__ import annotations

import json
import pathlib
import sys

from bioxplain.freeze import create_freeze

ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    run = pathlib.Path(sys.argv[1])
    run = run if run.is_absolute() else ROOT / run
    from bioxplain.utils.provenance import sha256_file
    refs = {}
    for name, rel in {"observed_vs_null": run / "analysis/observed_vs_null.csv", "null_repeat_level": run / "analysis/null_repeat_level_stability.csv",
                      "sensitivity_summary": ROOT / "results/sensitivity_summary.json"}.items():
        if rel.exists():
            refs[name] = {"path": str(rel.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(rel)}
    refs["null_replicates"] = len(list((ROOT / "results/null/main").glob("rep_*/manifest.json")))
    fz = create_freeze(ROOT, run, extra_refs=refs)
    c = fz["consensus"]["lists"]
    md = [
        "# Discovery freeze", "",
        f"Frozen at {fz['frozen_utc']} on commit `{fz['git']['commit']}` (clean tree). Machine-readable: `docs/freeze/discovery_freeze.json`.", "",
        "After this file was committed no discovery decision may use GSE65194: no tuning, and no gene, k, threshold, preprocessing or endpoint change.", "",
        "## Discovery run",
        f"- run: `{fz['discovery_run']['dir']}` (config sha256 `{fz['discovery_run']['config_sha256'][:16]}`, code commit at run start/manifest `{fz['discovery_run']['code_commit_of_run'][:8]}`, "
        f"{fz['discovery_run']['n_folds']} folds, seed {fz['discovery_run']['seed']})",
        f"- samples: {fz['data']['sample_definition']}",
        f"- feature universe: {fz['data']['n_universe_probes']} probes / {fz['data']['n_universe_genes']} genes (sha256 `{fz['data']['feature_universe_sha256'][:16]}`); "
        f"eligible on the full cohort: {fz['data']['n_eligible_genes_full_cohort']} genes",
        "", "## Preprocessing", *[f"- {k}: {v}" for k, v in fz["preprocessing"].items()],
        "", "## Models (resolved, pinned)", *[f"- `{k}`: {v}" for k, v in fz["models"].items()],
        "", "## Explainers", *[f"- `{k}`: {v}" for k, v in fz["explainers"].items()],
        "", f"## Stability\n- {fz['stability']}", "",
        "## Consensus", f"- rule: {fz['consensus']['rule']}", f"- frozen list sizes: {c}", "",
        "## Frozen lists", *[f"### {k}\n{', '.join(v) if v else '(empty)'}\n" for k, v in fz["consensus"]["frozen_lists"].items()],
        "## External endpoints", f"- {fz['external_endpoints']}", "", f"## Software\n- {fz['software']}", "",
        f"Frozen gene table: `{fz['frozen_genes_file']}` sha256 `{fz['frozen_genes_sha256']}`.",
    ]
    (ROOT / "docs/freeze/discovery_freeze.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"commit": fz["git"]["commit"], "lists": c, "genes_sha256": fz["frozen_genes_sha256"]}, indent=1))
