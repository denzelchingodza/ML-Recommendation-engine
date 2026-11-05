"""Tests for load_data: every bad input must fail loudly with a clear message."""

import pytest

from recommender.data.load_data import DataLoadError, file_fingerprint, load_csv


def write_csv(path, text):
    path.write_bytes(text.encode("mac_roman"))
    return path


def test_loads_a_good_file(tmp_path):
    path = write_csv(tmp_path / "good.csv", "a,b\n1,MiaSolé\n")
    frame = load_csv(path, file_fingerprint(path), ["a", "b"])
    assert len(frame) == 1
    assert frame.loc[0, "b"] == "MiaSolé"  # Mac Roman decoded correctly


def test_missing_file_says_could_not_find(tmp_path):
    with pytest.raises(DataLoadError, match="Could not find"):
        load_csv(tmp_path / "nope.csv", "abc", ["a"])


def test_changed_file_is_refused(tmp_path):
    path = write_csv(tmp_path / "data.csv", "a\n1\n")
    expected = file_fingerprint(path)
    write_csv(path, "a\n2\n")  # one byte changed
    with pytest.raises(DataLoadError, match="has changed"):
        load_csv(path, expected, ["a"])


def test_missing_column_is_named(tmp_path):
    path = write_csv(tmp_path / "data.csv", "a\n1\n")
    with pytest.raises(DataLoadError, match=r"missing required columns: \['b'\]"):
        load_csv(path, file_fingerprint(path), ["a", "b"])


def test_empty_file_is_refused(tmp_path):
    path = write_csv(tmp_path / "data.csv", "a,b\n")
    with pytest.raises(DataLoadError, match="no rows"):
        load_csv(path, file_fingerprint(path), ["a", "b"])
