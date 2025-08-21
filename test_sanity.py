"""Basic sanity tests for the project."""

import pytest
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from postprocess_trt import main as postprocess_main


def test_postprocess_script_missing_file(tmp_path):
    """Test that postprocess script handles missing file gracefully."""
    # Change working directory to tmp_path
    original_cwd = Path.cwd()
    import os
    os.chdir(tmp_path)
    
    try:
        # Should return 0 when file doesn't exist
        result = postprocess_main()
        assert result == 0
    finally:
        os.chdir(original_cwd)


def test_requirements_file_exists():
    """Test that requirements.txt exists and has the right spacy version."""
    req_file = Path("requirements.txt")
    assert req_file.exists(), "requirements.txt should exist"
    
    content = req_file.read_text()
    assert "spacy==3.7.4" in content, "spacy should be pinned to version 3.7.4"


def test_postprocess_script_exists():
    """Test that the postprocess script exists."""
    script_file = Path("scripts/postprocess_trt.py")
    assert script_file.exists(), "postprocess_trt.py should exist"


def test_workflow_file_exists():
    """Test that the new workflow file exists."""
    workflow_file = Path(".github/workflows/notebook-ci.yml")
    assert workflow_file.exists(), "notebook-ci.yml workflow should exist"
    
    content = workflow_file.read_text()
    assert "en_core_web_sm==3.7.1" in content, "workflow should pin spaCy model to 3.7.1"
    assert "scripts/postprocess_trt.py" in content, "workflow should call postprocess script"