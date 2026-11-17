"""Unit tests for dataset ingestion, validation, and sample data generation."""

import pandas as pd
import pytest

from datasleuth.ingestion import (
    format_bytes,
    get_dataset_metadata,
    get_sample_datasets,
    load_csv_from_bytes,
    load_sample_dataset,
    validate_and_deduplicate_columns,
)


def test_format_bytes():
    assert format_bytes(500) == "500.0 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(1024 * 1024 * 5) == "5.0 MB"


def test_validate_and_deduplicate_columns():
    raw_cols = ["age", "gender", "age", "  income  ", "", "age"]
    clean_cols, warnings = validate_and_deduplicate_columns(raw_cols)
    assert clean_cols == ["age", "gender", "age_1", "income", "unnamed_column_5", "age_2"]
    assert len(warnings) == 3


def test_load_csv_valid():
    csv_bytes = b"id,feature1,feature2,target\n1,10.5,A,0\n2,20.1,B,1\n3,15.2,A,1\n"
    res = load_csv_from_bytes(csv_bytes, filename="test.csv")
    assert res.success is True
    assert res.df is not None
    assert len(res.df) == 3
    assert list(res.df.columns) == ["id", "feature1", "feature2", "target"]
    assert res.metadata["row_count"] == 3


def test_load_csv_invalid_extension():
    res = load_csv_from_bytes(b"some content", filename="test.json")
    assert res.success is False
    assert "Unsupported file format" in res.error_message


def test_load_csv_empty_file():
    res = load_csv_from_bytes(b"", filename="empty.csv")
    assert res.success is False
    assert "completely empty" in res.error_message


def test_load_csv_size_limit():
    large_bytes = b"a,b\n" * 1000
    res = load_csv_from_bytes(large_bytes, filename="large.csv", max_size_mb=0)
    assert res.success is False
    assert "exceeds the maximum allowed limit" in res.error_message


def test_load_csv_semicolon_delimited():
    csv_bytes = b"col1;col2;col3\n1;2;3\n4;5;6\n"
    res = load_csv_from_bytes(csv_bytes, filename="data.csv")
    assert res.success is True
    assert res.df.shape == (2, 3)
    assert list(res.df.columns) == ["col1", "col2", "col3"]


def test_sample_datasets_catalog():
    catalog = get_sample_datasets()
    assert "heart_disease" in catalog
    assert "housing_pricing" in catalog
    assert "customer_churn" in catalog


@pytest.mark.parametrize("key", ["heart_disease", "housing_pricing", "customer_churn"])
def test_load_sample_datasets(key):
    df = load_sample_dataset(key)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    meta = get_dataset_metadata(df)
    assert meta["row_count"] == 1000
    assert meta["column_count"] > 5
