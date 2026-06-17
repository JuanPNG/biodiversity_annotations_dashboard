"""Build the dashboard's GBIF occurrence Parquet table.

The source integrated Parquet stores GBIF occurrences as a nested list per
accession. This module explodes that list into one row per occurrence while
preserving occurrence-level fields used by maps and downstream summaries.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# PARQUET_PATH = '../data/integ_genome_features_20250812.parquet'

def _safe_col(table: pa.Table, name: str) -> List[Any]:
    """Return column values as a list, or None placeholders if the column is absent."""
    return table[name].to_pylist() if name in table.column_names else [None] * len(table)


def build_gbif_occurrences(parquet_path: str) -> pd.DataFrame:
    """Explode GBIF occurrence records into one row per occurrence.

    Input source column:
        gbif_occs: list<struct>

    Output schema:
        accession
        occurrenceID
        geo_coordinate
        geodeticDatum
        coordinateUncertaintyInMeters
        eventDate
        elevation
        countryCode
        iucnRedListCategory
        gadm_level0_name
        gadm_level1_name
        gadm_level2_name
        institutionCode
        collectionCode
        catalogNumber

    Nested GADM values are flattened to explicit level-name columns. Other
    occurrence fields are preserved as provided by the source.
    """
    wanted = ["accession", "gbif_occs"]
    schema = pq.read_schema(parquet_path)
    cols = [c for c in wanted if c in schema.names]
    table = pq.read_table(parquet_path, columns=cols)

    acc_col = _safe_col(table, "accession")
    occs_col = _safe_col(table, "gbif_occs")

    rows: List[Dict[str, Any]] = []

    for i in range(len(table)):
        a = acc_col[i]
        occs = occs_col[i]
        if not isinstance(occs, list):
            continue
        for o in occs:
            if not isinstance(o, dict):
                continue
            # Flatten top-level fields verbatim
            row: Dict[str, Any] = {
                "accession": a,
                "occurrenceID": o.get("occurrenceID"),
                "geo_coordinate": o.get("geo_coordinate"),
                "geodeticDatum": o.get("geodeticDatum"),
                "coordinateUncertaintyInMeters": o.get("coordinateUncertaintyInMeters"),
                "eventDate": o.get("eventDate"),
                "elevation": o.get("elevation"),
                "countryCode": o.get("countryCode"),
                "iucnRedListCategory": o.get("iucnRedListCategory"),
                "institutionCode": o.get("institutionCode"),
                "collectionCode": o.get("collectionCode"),
                "catalogNumber": o.get("catalogNumber"),
            }

            # Expand GADM level structs into explicit name columns.
            gadm = o.get("gadm")
            if isinstance(gadm, dict):
                for level in ("level0", "level1", "level2"):
                    lv = gadm.get(level)
                    if isinstance(lv, dict):
                        row[f"gadm_{level}_name"] = lv.get("name")
                    else:
                        row[f"gadm_{level}_name"] = None
            else:
                for level in ("level0", "level1", "level2"):
                    row[f"gadm_{level}_name"] = None

            rows.append(row)

    # Build DataFrame with a stable column order
    cols_out = [
        "accession",
        "occurrenceID", "geo_coordinate", "geodeticDatum", "coordinateUncertaintyInMeters",
        "eventDate", "elevation", "countryCode", "iucnRedListCategory",
        "gadm_level0_name", "gadm_level1_name", "gadm_level2_name",
        "institutionCode", "collectionCode", "catalogNumber",
    ]
    df = pd.DataFrame(rows, columns=cols_out).drop_duplicates(ignore_index=True)
    return df
#
# df = build_gbif_occurrences(parquet_path=PARQUET_PATH)
#
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.head())
#
# ja = df[df['accession'] == 'GCA_905220365.2'].to_dict('records')
# print(json.dumps(ja, indent=4))
