import re
from importlib.metadata import version
from pathlib import Path

import toolscosmo


def test_version_attribute_matches_metadata():
    assert toolscosmo.__version__ == version("toolscosmo")


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():  # tests run from an installed wheel without the repo
        return
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M).group(1)
    assert toolscosmo.__version__ == declared


def test_version_helper_not_exported():
    # importlib.metadata.version must not leak into the toolscosmo namespace.
    assert not hasattr(toolscosmo, "version")
