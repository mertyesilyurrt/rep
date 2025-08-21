"""
Unit tests for syntax_utils module.

Tests the improved alignment and dependency calculation functions.
"""

import pytest
import spacy
from scripts.syntax_utils import (
    align_aoi_to_spacy_windowed,
    dep_distance,
    dep_depth,
    is_punctuation_token,
    normalize_token_pass1,
    normalize_token_pass2
)


@pytest.fixture
def nlp():
    """Load spaCy model for testing."""
    return spacy.load("en_core_web_sm")


def test_normalization_functions():
    """Test the two-pass normalization functions."""
    # Test pass1 normalization (keeps apostrophes and hyphens)
    assert normalize_token_pass1("won't") == "won't"
    assert normalize_token_pass1("well-known") == "well-known"
    assert normalize_token_pass1("Hello,") == "hello"
    assert normalize_token_pass1("can't!") == "can't"
    
    # Test pass2 normalization (strips apostrophes and hyphens)
    assert normalize_token_pass2("won't") == "wont"
    assert normalize_token_pass2("well-known") == "wellknown"
    assert normalize_token_pass2("Hello,") == "hello"
    assert normalize_token_pass2("can't!") == "cant"


def test_alignment_contractions_and_hyphens(nlp):
    """Test alignment with contractions and hyphenated words."""
    # Test sentence with contractions and hyphens
    text = "He won't re-enter the well-known room."
    doc = nlp(text)
    doc_tokens = [t.text for t in doc if not t.is_space]
    
    # AOI tokens that should align
    aoi_tokens = ["He", "won't", "re-enter", "the", "well-known", "room", "."]
    
    alignment = align_aoi_to_spacy_windowed(aoi_tokens, doc_tokens, max_window=4)
    
    # Check that all tokens aligned (no None values)
    assert all(idx is not None for idx in alignment), f"Some tokens didn't align: {alignment}"
    
    # Check specific alignments
    assert alignment[0] is not None  # "He"
    assert alignment[1] is not None  # "won't"
    assert alignment[2] is not None  # "re-enter" 
    assert alignment[3] is not None  # "the"
    assert alignment[4] is not None  # "well-known"
    assert alignment[5] is not None  # "room"
    assert alignment[6] is not None  # "."


def test_alignment_window_concatenation(nlp):
    """Test that alignment works with multi-token windows."""
    text = "The well-known method works."
    doc = nlp(text)
    doc_tokens = [t.text for t in doc if not t.is_space]
    
    # AOI with hyphenated compound that might be split
    aoi_tokens = ["The", "well-known", "method", "works", "."]
    
    alignment = align_aoi_to_spacy_windowed(aoi_tokens, doc_tokens, max_window=4)
    
    # Should successfully align the compound term
    assert alignment[1] is not None, f"Failed to align hyphenated compound. Alignment: {alignment}, Doc tokens: {doc_tokens}"


def test_dep_distance_calculations(nlp):
    """Test dependency distance calculations."""
    text = "The quick brown fox jumps."
    doc = nlp(text)
    
    # Find ROOT token
    root_token = None
    for token in doc:
        if token.head == token:
            root_token = token
            break
    
    assert root_token is not None, "No ROOT token found"
    
    # ROOT should have distance 0
    assert dep_distance(root_token) == 0
    
    # Other tokens should have non-zero distance (except maybe some edge cases)
    non_root_tokens = [t for t in doc if t.head != t and not t.is_space]
    for token in non_root_tokens:
        distance = dep_distance(token)
        assert distance >= 0, f"Negative distance for token {token.text}"


def test_dep_depth_calculations(nlp):
    """Test dependency depth calculations.""" 
    text = "The quick brown fox jumps."
    doc = nlp(text)
    
    # Find ROOT token
    root_token = None
    for token in doc:
        if token.head == token:
            root_token = token
            break
    
    assert root_token is not None, "No ROOT token found"
    
    # ROOT should have depth 0
    assert dep_depth(root_token) == 0
    
    # Children of ROOT should have depth 1, etc.
    for token in doc:
        if not token.is_space:
            depth = dep_depth(token)
            assert depth >= 0, f"Negative depth for token {token.text}"
            assert depth <= 10, f"Unusually high depth {depth} for token {token.text}"


def test_cross_sentence_dependencies(nlp):
    """Test handling of cross-sentence dependencies."""
    text = "First sentence. Second sentence has more words."
    doc = nlp(text)
    
    # Test that all tokens get reasonable distance/depth values
    for token in doc:
        if not token.is_space:
            distance = dep_distance(token)
            depth = dep_depth(token)
            
            assert distance >= 0, f"Negative distance for {token.text}"
            assert depth >= 0, f"Negative depth for {token.text}"


def test_punctuation_detection(nlp):
    """Test punctuation detection function."""
    text = "Hello, world!"
    doc = nlp(text)
    
    punct_count = 0
    for token in doc:
        if is_punctuation_token(token):
            punct_count += 1
            # Punctuation should have PUNCT pos tag
            assert token.pos_ == "PUNCT" or token.is_punct
    
    # Should find some punctuation in the sentence
    assert punct_count > 0, "No punctuation detected"


def test_alignment_with_punctuation(nlp):
    """Test alignment handles punctuation correctly."""
    text = "Hello, world!"
    doc = nlp(text)
    doc_tokens = [t.text for t in doc if not t.is_space]
    
    aoi_tokens = ["Hello", ",", "world", "!"]
    
    alignment = align_aoi_to_spacy_windowed(aoi_tokens, doc_tokens, max_window=4)
    
    # All tokens should align including punctuation
    assert all(idx is not None for idx in alignment), f"Failed alignment: {alignment}"


def test_empty_and_edge_cases():
    """Test edge cases and empty inputs."""
    # Empty inputs
    assert align_aoi_to_spacy_windowed([], [], max_window=4) == []
    assert align_aoi_to_spacy_windowed(["test"], [], max_window=4) == [None]
    assert align_aoi_to_spacy_windowed([], ["test"], max_window=4) == []
    
    # Normalization edge cases
    assert normalize_token_pass1("") == ""
    assert normalize_token_pass2("") == ""
    assert normalize_token_pass1("123") == "123"
    assert normalize_token_pass2("!@#") == ""


if __name__ == "__main__":
    pytest.main([__file__])