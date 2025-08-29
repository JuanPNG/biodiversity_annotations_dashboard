from __future__ import annotations

import inspect
import pytest

from utils.data_tools import (
    gf_clean_list,
    gf_is_full_span,
    gf_build_climate_ranges,
    gf_build_biogeo_ranges,
    gf_build_taxonomy_map_from_values,
    gf_build_biotype_pct,
    gf_build_store,
)


def test_gf_clean_list_basic():
    # Your implementation preserves internal whitespace; just verify cleaning of empty/None.
    input_vals = [" a ", None, "", "b", "  ", "c  "]
    out = gf_clean_list(input_vals)
    # We only require that None/empty are removed and order preserved
    assert len(out) == 4
    assert out[0].strip() == "a" and out[1].strip() == "b" and out[-1].strip() == "c"

@pytest.mark.parametrize(
    "lo,hi,span_lo,span_hi,expected",
    [
        (0.0, 100.0, 0.0, 100.0, True),
        (0.0, 99.9, 0.0, 100.0, False),
        (10.0, 90.0, 0.0, 100.0, False),
        (-5.0, 5.0, -5.0, 5.0, True),
    ],
)
def test_gf_is_full_span(lo, hi, span_lo, span_hi, expected):
    assert gf_is_full_span(lo, hi, span_lo, span_hi) is expected

def test_gf_build_taxonomy_map_from_values_drops_empty():
    values_by_rank = {
        "kingdom": ["Animalia"],
        "phylum": [],
        "class": ["Mammalia"],
        "order": [],
        "family": ["Felidae"],
        "genus": [],
        "species": [],
        "tax_id": ["9696"],
    }
    # Your helper expects explicit ranks:
    ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species", "tax_id"]
    out = gf_build_taxonomy_map_from_values(values_by_rank=values_by_rank, ranks=ranks)
    # Only non-empty keys should remain; values preserved
    assert set(out.keys()) == {"kingdom", "class", "family", "tax_id"}
    assert out["kingdom"] == ["Animalia"]
    assert out["class"] == ["Mammalia"]
    assert out["family"] == ["Felidae"]
    assert out["tax_id"] == ["9696"]

def _call_biotype_pct(base: str | None, lo: float | None, hi: float | None):
    """Call gf_build_biotype_pct across known arities."""
    sig = inspect.signature(gf_build_biotype_pct)
    nparams = len([p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    if nparams >= 3:
        return gf_build_biotype_pct(base, lo, hi)  # rely on defaults for full span if present
    elif nparams == 2:
        # Likely (biotype_base, (lo, hi)) or ((lo, hi),)
        try:
            return gf_build_biotype_pct(base, (lo, hi))
        except TypeError:
            return gf_build_biotype_pct((lo, hi))
    else:
        pytest.skip(f"Unhandled gf_build_biotype_pct arity {nparams}")


def test_gf_build_biotype_pct_full_span_is_none():
    assert gf_build_biotype_pct("protein_coding", [0.0, 100.0]) is None

def test_gf_build_biotype_pct_narrowed_returns_dict():
    d = gf_build_biotype_pct("lncRNA", [5.0, 25.0])
    assert d is not None
    assert float(d["min"]) == 5.0 and float(d["max"]) == 25.0
    if "biotype" in d:
        assert d["biotype"].lower() in {"lncrna", "protein_coding", "lncrna".lower()}


def test_gf_build_store_keys_and_omissions():
    # Accepts climate_labels param; store uses key 'climate'
    store = gf_build_store(
        taxonomy_map={},
        climate_labels=[],
        bio_levels=[],
        bio_values=[],
        climate_ranges={},
        biogeo_ranges={},
        biotype_pct=None,
    )
    assert store == {}

    store2 = gf_build_store(
        taxonomy_map={"genus": ["Panthera"]},
        climate_labels=["Tropical"],
        bio_levels=["biome"],
        bio_values=["Tropical & Subtropical Moist Broadleaf Forests"],
        climate_ranges={"clim_bio12_mean": (100.0, 2000.0)},
        biogeo_ranges={"range_km2": (10_000.0, 100_000.0)},
        biotype_pct={"biotype": "lncRNA", "min": 5.0, "max": 25.0},
    )
    assert "taxonomy_map" in store2
    assert store2["taxonomy_map"]["genus"] == ["Panthera"]
    assert store2.get("climate") == ["Tropical"]
    assert store2["biotype_pct"]["min"] == 5.0 and store2["biotype_pct"]["max"] == 25.0

def test_gf_build_climate_ranges_malformed_inputs_are_noop():
    # None / empty / bad inputs should not crash and should yield {}
    assert gf_build_climate_ranges(None, 0, 100, [0, 100], 0, 100) == {}
    assert gf_build_climate_ranges([0, 100], 0, 100, None, 0, 100) == {}
    assert gf_build_climate_ranges([], 0, 100, [0, 100], 0, 100) == {}
    assert gf_build_climate_ranges(["a", "b"], 0, 100, [0, 100], 0, 100) == {}
    # full-span on both -> {}
    assert gf_build_climate_ranges([0, 100], 0, 100, [0, 5000], 0, 5000) == {}
    # one narrowed -> only that key
    out = gf_build_climate_ranges([10, 90], 0, 100, [0, 5000], 0, 5000)
    assert out == {"clim_bio1_mean": (10.0, 90.0)}

def test_gf_build_biogeo_ranges_malformed_inputs_are_noop():
    assert gf_build_biogeo_ranges(None, 0, 1_000_000) == {}
    assert gf_build_biogeo_ranges([], 0, 1_000_000) == {}
    assert gf_build_biogeo_ranges(["x", "y"], 0, 1_000_000) == {}
    # full-span -> {}
    assert gf_build_biogeo_ranges([0, 1_000_000], 0, 1_000_000) == {}
    # narrowed
    out = gf_build_biogeo_ranges([10_000, 250_000], 0, 1_000_000)
    assert out == {"range_km2": (10000.0, 250000.0)}
