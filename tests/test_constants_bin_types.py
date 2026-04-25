from __future__ import annotations

from app.constants import BIN_TYPES


def test_bin_type_bucket_variants_intentionally_share_bin_class_id():
    assert BIN_TYPES["recyclable"]["classes"] == [0]
    assert BIN_TYPES["hazardous"]["classes"] == [0]
    assert BIN_TYPES["other"]["classes"] == [0]
    assert BIN_TYPES["kitchen"]["classes"] == [0]
    assert BIN_TYPES["overflow"]["classes"] == [1]
