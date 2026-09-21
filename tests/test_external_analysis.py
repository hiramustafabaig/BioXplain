"""End-to-end dry run of the external analysis on SYNTHETIC cohorts (the real GSE65194 is not opened before the freeze)."""
import json

import numpy as np
import pandas as pd
import pytest

import bioxplain.external_analysis as ea
from bioxplain.data.discovery import DiscoveryData
from bioxplain.external import ExternalData


@pytest.fixture()
def synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(120)] + ["ADIPOQ", "PLIN1"]
    p = len(genes)
    eff = rng.normal(0, 1, p); eff[:20] *= 3
    probes = [f"{g}_at" for g in genes]
    yd = np.r_[np.zeros(15, int), np.ones(60, int)]; ye = np.r_[np.zeros(11, int), np.ones(50, int)]
    Xd = pd.DataFrame(rng.normal(size=(75, p)) + yd[:, None] * eff, columns=probes, index=[f"D{i}" for i in range(75)])
    Xe = pd.DataFrame(2.0 * (rng.normal(size=(61, p)) + ye[:, None] * eff * 0.8) + 5, columns=probes, index=[f"E{i}" for i in range(61)])
    from bioxplain.replication import auc_effects
    e_disc = auc_effects(Xd.to_numpy(), yd) - 0.5
    s = np.abs(e_disc) / np.abs(e_disc).max()
    order = np.argsort(-s)
    frozen = pd.DataFrame({"gene": genes, "probe": probes, "e_disc": e_disc, "s_g_k25": s})
    for name, n in (("S_cons", 0), ("S_top25", 25), ("S_top50", 50)):
        frozen[f"in_{name}"] = False
        frozen.loc[order[:n], f"in_{name}"] = True
    (tmp_path / "results/freeze").mkdir(parents=True)
    frozen.to_csv(tmp_path / "results/freeze/frozen_genes.csv", index=False)
    subtype = np.where(ye == 0, "Healthy", np.array(["TNBC", "Her2", "Luminal A", "Luminal B", "TNBC"] * 14)[: len(ye)])
    units = pd.DataFrame({"subtype": subtype, "n_arrays": 1}, index=Xe.index)
    ext = ExternalData(X=Xe, y=ye, units=units, n_arrays_used=61)
    disc = DiscoveryData(X=Xd, y=pd.Series(yd, index=Xd.index), samples=pd.DataFrame(index=Xd.index), probe_to_gene=pd.Series(genes, index=probes), entrez=pd.Series(dtype=str))
    monkeypatch.setattr(ea, "verify_freeze", lambda root: {"frozen_genes_file": "results/freeze/frozen_genes.csv", "git": {"commit": "abc"}})
    monkeypatch.setattr(ea, "load_discovery", lambda root: disc)
    monkeypatch.setattr(ea, "load_external", lambda root, variant="mean": ext)
    return tmp_path


def test_end_to_end_on_synthetic_replicating_signal(synthetic):
    out = ea.run_external_analysis(synthetic, out_dir=synthetic / "out", n_boot=30, seed=1)
    for f in ["per_gene_external_effects.csv", "list_level_replication.csv", "summary.json", "manifest.json", "composition_panel.csv", "sensitivity_gene_level.json"]:
        assert (out / f).exists(), f
    s = json.loads((out / "summary.json").read_text())
    g = s["primary_gene_level"]
    assert g["c1_spearman"] > 0.3 and g["direction_all_genes"]["fraction"] > 0.6            # the synthetic signal replicates
    assert g["n_units"] == 61 and g["n_normal"] == 11 and len(g["c1_spearman_ci"]) == 2
    lst = pd.read_csv(out / "list_level_replication.csv").set_index("list")
    assert lst.loc["S_top25", "signature_auc_external"] > 0.9 and lst.loc["S_top50", "n_genes"] == 50
    assert "too few" in str(lst.loc["S_cons", "note"])                                       # empty consensus list is reported, not hidden
    assert s["sample_level_frozen_lr_S_top25"]["roc_auc"] > 0.9
    assert set(s["composition"]["panel_genes_present_in_eligible_universe"]) == {"ADIPOQ", "PLIN1"}
    man = json.loads((out / "manifest.json").read_text())
    assert man["external_units"]["n_normal"] == 11 and man["freeze_commit"] == "abc"
    subs = json.loads((out / "sensitivity_gene_level.json").read_text())
    assert {r["tag"] for r in subs} >= {"sensitivity_repA_only", "subtype_TNBC_vs_healthy"}


def test_null_signal_gives_no_replication(synthetic, monkeypatch):
    rng = np.random.default_rng(9)
    ext_noise = ExternalData(X=pd.DataFrame(rng.normal(size=(61, 122)), columns=pd.read_csv(synthetic / "results/freeze/frozen_genes.csv").probe, index=[f"E{i}" for i in range(61)]),
                             y=np.r_[np.zeros(11, int), np.ones(50, int)], units=pd.DataFrame({"subtype": ["TNBC"] * 61, "n_arrays": 1}, index=[f"E{i}" for i in range(61)]), n_arrays_used=61)
    monkeypatch.setattr(ea, "load_external", lambda root, variant="mean": ext_noise)
    out = ea.run_external_analysis(synthetic, out_dir=synthetic / "out2", n_boot=30, seed=1, variants=False)
    g = json.loads((out / "summary.json").read_text())["primary_gene_level"]
    assert abs(g["c1_spearman"]) < 0.3 and abs(g["direction_all_genes"]["fraction"] - 0.5) < 0.15
