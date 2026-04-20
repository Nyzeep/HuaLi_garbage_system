from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.exceptions import FileParseError, ValidationError
from app.core.validators import (
    validate_pagination,
    validate_skip_frames,
    validate_upload_size,
)


# ---------------------------------------------------------------------------
# validate_upload_size
# ---------------------------------------------------------------------------

def test_upload_size_ok():
    validate_upload_size(b"x" * 1024, max_mb=1)


def test_upload_size_exact_limit():
    validate_upload_size(b"x" * (1 * 1024 * 1024), max_mb=1)


def test_upload_size_over_limit():
    with pytest.raises(ValidationError):
        validate_upload_size(b"x" * (1 * 1024 * 1024 + 1), max_mb=1)


@given(st.integers(min_value=1, max_value=50))
def test_upload_size_property(max_mb):
    limit = max_mb * 1024 * 1024
    validate_upload_size(b"x" * limit, max_mb=max_mb)
    with pytest.raises(ValidationError):
        validate_upload_size(b"x" * (limit + 1), max_mb=max_mb)


# ---------------------------------------------------------------------------
# validate_skip_frames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (0, 1),
    (1, 1),
    (30, 30),
    (60, 60),
    (61, 60),
    (-5, 1),
])
def test_skip_frames_clamp(value, expected):
    assert validate_skip_frames(value) == expected


@given(st.integers(min_value=-1000, max_value=1000))
def test_skip_frames_always_in_range(value):
    result = validate_skip_frames(value)
    assert 1 <= result <= 60


# ---------------------------------------------------------------------------
# validate_pagination
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("page,per_page,exp_page,exp_per", [
    (1, 20, 1, 20),
    (0, 20, 1, 20),
    (-1, 20, 1, 20),
    (5, 0, 5, 1),
    (5, 101, 5, 100),
    (5, 100, 5, 100),
])
def test_pagination_clamp(page, per_page, exp_page, exp_per):
    assert validate_pagination(page, per_page) == (exp_page, exp_per)


@given(st.integers(min_value=-100, max_value=200), st.integers(min_value=-100, max_value=200))
def test_pagination_always_valid(page, per_page):
    p, pp = validate_pagination(page, per_page)
    assert p >= 1
    assert 1 <= pp <= 100
