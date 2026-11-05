"""Smoke test.

The only job of this test is to prove the pipeline runs: the package
imports and CI can execute tests. Real tests arrive with each phase.
"""

import re

import recommender


def test_package_imports_and_has_a_version():
    version = recommender.__version__
    assert isinstance(version, str)
    # Version must look like MAJOR.MINOR.PATCH, for example 0.0.1
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), f"Bad version: {version!r}"
