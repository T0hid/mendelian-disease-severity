"""Counts that link the corpus to the pairs classified in the earlier study (Online Methods, Data and severity labels)."""

import zipfile

import pandas as pd
from conftest import ROOT

from severity.mapping import corpus_reconciliation

SAVED = ROOT / "published_run" / "benchmark" / "results" / "corpus_reconciliation.csv"


def _mapping(tmp_path):
    with zipfile.ZipFile(ROOT / "data" / "training_tables.zip") as bundle:
        data = bundle.read("gene_disease_mapping_HPO_rebuilt.csv")
    path = tmp_path / "gene_disease_mapping_HPO_rebuilt.csv"
    path.write_bytes(data)
    return str(path)


def test_reconciliation_matches_saved_table(tmp_path):
    table = corpus_reconciliation(_mapping(tmp_path))
    saved = pd.read_csv(SAVED)
    pd.testing.assert_frame_equal(table.reset_index(drop=True), saved, check_dtype=False)


def test_reconciliation_adds_up(tmp_path):
    counts = dict(corpus_reconciliation(_mapping(tmp_path)).itertuples(index=False))
    assert counts["candidate gene-disease pairs"] == 8467
    assert counts["pairs kept in this corpus"] == 6298
    assert counts["kept pairs without a severity class in ref. 16"] == 0
    excluded = counts["pairs with a severity class in ref. 16 excluded here"]
    assert counts["pairs with a severity class in ref. 16"] == counts["pairs kept in this corpus"] + excluded
    groups = [key for key in counts if key.startswith("excluded here:") and key.count(":") == 1]
    assert sum(counts[key] for key in groups) == excluded
