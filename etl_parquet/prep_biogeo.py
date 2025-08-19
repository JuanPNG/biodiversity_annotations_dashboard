from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import pyarrow.parquet as pq
import pandas as pd


# PARQUET_PATH = '../data/integ_genome_features_20250812.parquet'

def _safe_col(table: pq.Table, name: str) -> List[Any]:
    return table[name].to_pylist() if name in table.column_names else [None] * len(table)

def _flat_str_list(v: Any) -> List[str]:
    """biogeo values are list<array:string>; flatten nested lists -> list[str]."""
    out: List[str] = []
    if isinstance(v, list):
        for x in v:
            if isinstance(x, list):
                out.extend([str(s) for s in x if s is not None])
            elif x is not None:
                out.append(str(x))
    return out

def build_biogeo_long(parquet_path: str) -> pd.DataFrame:
    """
    Create a tidy biogeography table with one level per row:

        accession | level | value
        level ∈ {"realm","biome","ecoregion"}

    The source schema has independent lists per level; we do NOT fabricate triplets.
    """
    wanted = ["accession", "biogeo_Ecoregion"]
    schema = pq.read_schema(parquet_path)
    cols = [c for c in wanted if c in schema.names]
    table = pq.read_table(parquet_path, columns=cols)

    acc = _safe_col(table, "accession")
    bioe = _safe_col(table, "biogeo_Ecoregion")

    rows: List[Dict[str, str]] = []

    for i in range(len(table)):
        a = acc[i]
        b = bioe[i]
        if not isinstance(b, dict):
            continue

        # Each block has: {"count": int, "values": list<array<string>>}
        r_vals = _flat_str_list(b.get("realm", {}).get("values") if isinstance(b.get("realm"), dict) else None)
        m_vals = _flat_str_list(b.get("biome", {}).get("values") if isinstance(b.get("biome"), dict) else None)
        e_vals = _flat_str_list(b.get("ecoregion", {}).get("values") if isinstance(b.get("ecoregion"), dict) else None)

        for r in r_vals:
            rows.append({"accession": str(a), "level": "realm", "value": r})
        for m in m_vals:
            rows.append({"accession": str(a), "level": "biome", "value": m})
        for e in e_vals:
            rows.append({"accession": str(a), "level": "ecoregion", "value": e})

    df = pd.DataFrame(rows, columns=["accession", "level", "value"]).drop_duplicates()
    # optional: enforce categories for convenience
    df["level"] = pd.Categorical(df["level"], categories=["realm","biome","ecoregion"], ordered=False)
    return df.reset_index(drop=True)


# df = build_biogeo_long(parquet_path=PARQUET_PATH)
#
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.head())
#
# ja = df[df['accession'] == 'GCA_905220365.2'].to_dict('records')
# print(json.dumps(ja, indent=4))