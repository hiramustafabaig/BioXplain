import pandas as pd
import pytest
from toy import write_toy_annotation, write_toy_series_matrix

from bioxplain.data.geo import read_gpl_annotation, read_probe_ids, read_series_matrix, sample_table, usable_probe_table
from bioxplain.data.labels import LabelError, derive_cancer_normal_labels
from bioxplain.utils.provenance import DataIntegrityError, config_hash, sha256_file, verify_file


def test_series_matrix_shape_and_header(tmp_path):
    header, expr = read_series_matrix(write_toy_series_matrix(tmp_path / "s.txt.gz"))
    assert expr.shape == (4, 6)
    assert list(expr.columns) == [f"GSM{i}" for i in range(1, 7)]
    assert header["!Series_geo_accession"] == [["GSETOY"]]
    assert expr.index.tolist()[0] == "1000_at"


def test_probe_ids_match_full_parse(tmp_path):
    p = write_toy_series_matrix(tmp_path / "s.txt.gz")
    assert read_probe_ids(p) == read_series_matrix(p)[1].index.tolist()


def test_sample_table_turns_characteristics_into_columns(tmp_path):
    header, expr = read_series_matrix(write_toy_series_matrix(tmp_path / "s.txt.gz"))
    st = sample_table(header)
    assert {"title", "source_name", "description", "tissue", "age"} <= set(st.columns)
    assert list(st.index) == list(expr.columns)
    assert st.loc["GSM4", "age"] == "50"


def test_labels_cancer_is_one(tmp_path):
    header, _ = read_series_matrix(write_toy_series_matrix(tmp_path / "s.txt.gz"))
    y = derive_cancer_normal_labels(sample_table(header))
    assert y.tolist() == [0, 0, 0, 1, 1, 1]
    assert y.dtype.kind == "i"


def test_labels_disagreement_between_fields_is_an_error(tmp_path):
    header, _ = read_series_matrix(write_toy_series_matrix(tmp_path / "s.txt.gz", mislabel_description=True))
    with pytest.raises(LabelError, match="description"):
        derive_cancer_normal_labels(sample_table(header))


def test_labels_unknown_tissue_is_an_error(tmp_path):
    header, _ = read_series_matrix(
        write_toy_series_matrix(tmp_path / "s.txt.gz", tissue_values=["normal breast"] * 3 + ["breast cancer", "breast cancer", "lung"])
    )
    with pytest.raises(LabelError, match="unrecognised"):
        derive_cancer_normal_labels(sample_table(header))


def test_gpl_usable_probe_rules(tmp_path):
    ann = read_gpl_annotation(write_toy_annotation(tmp_path / "a.annot.gz"))
    assert len(ann) == 5
    usable = usable_probe_table(ann)
    assert set(usable.probe_id) == {"1000_at", "1003_at"}      # multi-gene, '---' and AFFX removed
    assert set(usable.gene_symbol) == {"GENEA"}
    assert usable.entrez_id.tolist() == ["1", "1"]


def test_file_hash_verification(tmp_path):
    p = write_toy_series_matrix(tmp_path / "s.txt.gz")
    verify_file(p, sha256_file(p))
    with pytest.raises(DataIntegrityError):
        verify_file(p, "0" * 64)


def test_config_hash_is_order_independent_and_sensitive():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


@pytest.mark.integration
def test_real_discovery_data_matches_reconnaissance():
    import pathlib

    from bioxplain.data.discovery import load_discovery
    root = pathlib.Path(__file__).resolve().parents[1]
    if not (root / "data/raw/GSE42568/GSE42568_series_matrix.txt.gz").exists():
        pytest.skip("raw data not present")
    d = load_discovery(root)                                   # also verifies SHA-256
    assert d.X.shape == (121, 42892)
    assert int(d.y.sum()) == 104 and int((d.y == 0).sum()) == 17
    assert d.probe_to_gene.nunique() == 20848
    assert d.y.iloc[:17].eq(0).all() and d.y.iloc[17:].eq(1).all()   # class-blocked GEO order (recon finding F4)
    assert d.X.notna().all().all()
