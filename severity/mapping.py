"""Gene-disease mapping filter (Data and severity labels).

``gene_disease_mapping_HPO_rebuilt.csv`` lists every candidate gene-disease pair (8,467) with
its decision (``status`` KEPT or DROP, ``drop_reason`` and ``use_in_rebuilt_corpus``). Only the
6,298 kept pairs are used; every other row of the training table is removed before any feature
is built. The benchmark and the scorer apply the same filter. ``corpus_reconciliation`` counts how
the kept pairs relate to the pairs with a severity class in the earlier study of the labelling agent.
"""

import os
from typing import Any, Callable, Optional, Set, Tuple

import pandas as pd


def _normalise(value: Any) -> str:
    return str(value).strip().upper()


# Groups of drop_reason values in the mapping, in the wording of the Online Methods.
EXCLUSION_GROUPS = (
    ("no gene-level features", ("no gene-level features",)),
    ("inclusion rules", ("rule (",)),
    (
        "manual review",
        (
            "generic tumour type",
            "low-penetrance risk entry",
            "susceptibility entry not resolved",
            "not a germline Mendelian disease",
        ),
    ),
)


def corpus_reconciliation(mapping_path: str) -> pd.DataFrame:
    """Link the corpus to the gene-disease pairs that have a severity class in the earlier study (ref. 16).

    A candidate pair has a severity class there when it has a (frequency-filtered) severity label and
    its GenCC support is not Limited only. The table counts those pairs, the pairs kept here and the
    reasons the others were excluded (Data and severity labels).
    """
    mapping = pd.read_csv(mapping_path, low_memory=False)
    for column in ("severity", "classification_title", "status", "drop_reason"):
        if column not in mapping.columns:
            raise ValueError(f"{mapping_path} has no column {column!r}.")
    classified = mapping["severity"].notna() & mapping["classification_title"].astype(str).str.strip().ne("Limited")
    kept = mapping["status"].eq("KEPT")
    reasons = mapping.loc[classified & ~kept, "drop_reason"].astype(str)

    rows = [
        ("candidate gene-disease pairs", len(mapping)),
        ("pairs with a severity class in ref. 16", int(classified.sum())),
        ("pairs kept in this corpus", int((classified & kept).sum())),
        ("kept pairs without a severity class in ref. 16", int((kept & ~classified).sum())),
        ("pairs with a severity class in ref. 16 excluded here", len(reasons)),
    ]
    grouped = pd.Series(False, index=reasons.index)
    for group, prefixes in EXCLUSION_GROUPS:
        in_group = reasons.apply(lambda reason: reason.startswith(prefixes))
        grouped |= in_group
        rows.append((f"excluded here: {group}", int(in_group.sum())))
        if group == "manual review":
            for reason, count in reasons.loc[in_group].value_counts().sort_index().items():
                rows.append((f"excluded here: manual review: {reason}", int(count)))
    rows.append(("excluded here: other reasons", int((~grouped).sum())))
    return pd.DataFrame(rows, columns=["quantity", "n"])


def load_allowed_pairs(mapping_path: str) -> Tuple[Set[Tuple[str, str]], dict]:
    """Return the (gene, MONDO) pairs kept in the mapping, with a short summary."""
    mapping = pd.read_csv(mapping_path, low_memory=False)
    for column in ("gene_symbol", "mondo_id", "status", "use_in_rebuilt_corpus"):
        if column not in mapping.columns:
            raise ValueError(f"{mapping_path} has no column {column!r}.")
    keep = mapping["use_in_rebuilt_corpus"].astype(str).str.lower().isin(("true", "1", "yes"))
    keep &= mapping["status"].eq("KEPT")
    selected = mapping.loc[keep]
    allowed = {(_normalise(g), _normalise(m)) for g, m in zip(selected.gene_symbol, selected.mondo_id)}
    summary = {"mapping_path": mapping_path, "allowed_pairs": len(allowed)}
    return allowed, summary


def apply_rebuilt_mapping(
    df: pd.DataFrame,
    gene_col: str,
    mondo_col: str,
    mapping_path: str,
    log: Optional[Callable[..., None]] = None,
    audit_dir: Optional[str] = None,
    audit_name: str = "gene_disease_mapping_filter_audit.csv",
) -> pd.DataFrame:
    """Keep only the rows of ``df`` whose gene-disease pair is kept in the mapping."""

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Gene-disease mapping not found: {mapping_path}")
    allowed, summary = load_allowed_pairs(mapping_path)
    key = pd.Series(
        list(zip(df[gene_col].map(_normalise), df[mondo_col].map(_normalise))),
        index=df.index,
    )
    mask = key.isin(allowed)
    filtered = df.loc[mask].copy()
    before_pairs, after_pairs = int(key.nunique()), int(key.loc[mask].nunique())
    emit(f"  HPO-verified mapping: {summary['mapping_path']} ({summary['allowed_pairs']} pairs)")
    emit(f"  Gene-disease pairs {before_pairs} -> {after_pairs}; source rows {len(df)} -> {len(filtered)}")
    if after_pairs == 0:
        raise ValueError(
            "The gene-disease mapping removed every pair; check gene symbols and MONDO identifiers."
        )
    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
        removed = df.loc[~mask, [gene_col, mondo_col]].drop_duplicates()
        removed = removed.rename(columns={gene_col: "gene_symbol", mondo_col: "mondo_id"})
        removed["removed_reason"] = "not verified by HPO gene-to-disease mapping"
        removed.to_csv(os.path.join(audit_dir, audit_name), index=False)
    return filtered
