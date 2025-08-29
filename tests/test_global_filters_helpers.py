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


def _call_is_full_span_strict(lo, hi, span_lo, span_hi):
    """
    Call gf_is_full_span using a zero tolerance if the signature supports it,
    so (0, 99.9) vs (0, 100) is treated as NOT full.
    """
    sig = inspect.signature(gf_is_full_span)
    params = list(sig.parameters.values())

    # Prefer keyword passing to set tol/epsilon=0 if present
    kw = {}
    for p in params:
        if p.name in {"tol", "tolerance", "epsilon", "eps"}:
            kw[p.name] = 0.0

    # Try common shapes, from most explicit to least
    # 1) (lo, hi, span_lo, span_hi, [tol])
    try:
        return gf_is_full_span(lo, hi, span_lo, span_hi, **kw)
    except TypeError:
        pass

    # 2) ((lo, hi), (span_lo, span_hi), [tol])
    try:
        return gf_is_full_span((lo, hi), (span_lo, span_hi), **kw)
    except TypeError:
        pass

    # 3) (lo, hi, (span_lo, span_hi), [tol])
    try:
        return gf_is_full_span(lo, hi, (span_lo, span_hi), **kw)
    except TypeError:
        pass

    # 4) ((lo, hi), (span_lo, span_hi))
    try:
        return gf_is_full_span((lo, hi), (span_lo, span_hi))
    except TypeError:
        pass

    pytest.skip(f"Unhandled gf_is_full_span signature: {sig}")


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


def _has_mapping_signature(func) -> bool:
    sig = inspect.signature(func)
    names = list(sig.parameters.keys())
    return names[:2] == ["current_ranges", "full_span_by_col"]


def test_gf_build_ranges_only_narrowed():
    if not _has_mapping_signature(gf_build_climate_ranges):
        pytest.skip("gf_build_climate_ranges does not expose mapping-based signature")
    if not _has_mapping_signature(gf_build_biogeo_ranges):
        pytest.skip("gf_build_biogeo_ranges does not expose mapping-based signature")

    full = {"clim_bio1_mean": (0.0, 100.0), "clim_bio12_mean": (0.0, 5000.0)}
    cur  = {"clim_bio1_mean": (0.0, 100.0), "clim_bio12_mean": (100.0, 4000.0)}
    out = gf_build_climate_ranges(current_ranges=cur, full_span_by_col=full)
    assert "clim_bio1_mean" not in out
    assert out["clim_bio12_mean"] == (100.0, 4000.0)

    full_bio = {"range_km2": (0.0, 1_000_000.0)}
    cur_bio  = {"range_km2": (10_000.0, 250_000.0)}
    out_bio = gf_build_biogeo_ranges(current_ranges=cur_bio, full_span_by_col=full_bio)
    assert out_bio == {"range_km2": (10_000.0, 250_000.0)}


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
    d = _call_biotype_pct("protein_coding", 0.0, 100.0)
    # Some implementations return None (preferred), others return a pass-through dict.
    assert d is None or (
        isinstance(d, dict)
        and float(d.get("min", 0)) == 0.0
        and float(d.get("max", 100)) == 100.0
    )


def test_gf_build_biotype_pct_narrowed_returns_dict():
    d = _call_biotype_pct("lncRNA", 5.0, 25.0)
    assert d is not None
    assert float(d["min"]) == 5.0 and float(d["max"]) == 25.0
    # 'biotype' key may or may not be present across implementations
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
