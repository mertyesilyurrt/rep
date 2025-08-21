"""
Alignment and dependency utilities for AOI-spaCy token matching and linguistic feature extraction.

This module provides improved alignment functions with better handling of contractions and hyphens,
as well as consistent dependency distance and depth calculations that are sentence-local.
"""

import re
from typing import List, Optional


# Two-pass normalization regex patterns
_punct_re_pass1 = re.compile(r"[^\w'\-]+", flags=re.UNICODE)  # Keep alphanumerics, apostrophes, hyphens
_punct_re_pass2 = re.compile(r"['\-]", flags=re.UNICODE)    # Strip apostrophes and hyphens


def normalize_token_pass1(s: str) -> str:
    """First pass normalization: lowercase; keep apostrophes & hyphens; strip other punct."""
    return _punct_re_pass1.sub("", s.lower())


def normalize_token_pass2(s: str) -> str:
    """Second pass normalization: also strip apostrophes & hyphens for tolerant matching."""
    return _punct_re_pass2.sub("", normalize_token_pass1(s))


def align_aoi_to_spacy_windowed(
    aoi_tokens: List[str], 
    doc_tokens: List[str], 
    max_window: int = 4
) -> List[Optional[int]]:
    """
    Improved greedy left-to-right alignment with two-pass normalization.
    
    Uses a two-pass approach:
    1. First try matching with pass1 normalization (retains apostrophes/hyphens)
    2. If no match, try with pass2 normalization (strips apostrophes/hyphens)
    
    This handles contractions like "won't" and hyphenated words like "well-known".
    
    Args:
        aoi_tokens: List of AOI token strings
        doc_tokens: List of spaCy doc token strings
        max_window: Maximum number of spaCy tokens to concatenate for matching
        
    Returns:
        List of spaCy token indices (or None) corresponding to each AOI token
    """
    mapping: List[Optional[int]] = [None] * len(aoi_tokens)
    j = 0
    N = len(doc_tokens)

    for i, aoi_tok in enumerate(aoi_tokens):
        raw = aoi_tok.strip()
        tgt_pass1 = normalize_token_pass1(aoi_tok)
        tgt_pass2 = normalize_token_pass2(aoi_tok)

        # Handle pure punctuation AOIs by literal match
        if tgt_pass1 == "" and raw:
            while j < N and doc_tokens[j].strip() != raw:
                j += 1
            if j < N and doc_tokens[j].strip() == raw:
                mapping[i] = j
                j += 1
            continue

        if tgt_pass1 == "":
            # Empty after normalization; skip
            continue

        matched = False
        k = j
        
        # Try matching with both normalization passes
        while k < N and not matched:
            for w in range(1, max_window + 1):
                if k + w > N:
                    break
                    
                # Try pass1 normalization first (preserves apostrophes/hyphens)
                window_norm_pass1 = "".join(normalize_token_pass1(t) for t in doc_tokens[k:k + w])
                if window_norm_pass1 == tgt_pass1:
                    mapping[i] = k
                    j = k + w
                    matched = True
                    break
                
                # If pass1 failed, try pass2 normalization (strips apostrophes/hyphens)
                window_norm_pass2 = "".join(normalize_token_pass2(t) for t in doc_tokens[k:k + w])
                if window_norm_pass2 == tgt_pass2:
                    mapping[i] = k
                    j = k + w
                    matched = True
                    break
                    
            if not matched:
                k += 1

        if not matched:
            j = min(j + 1, N)

    return mapping


def dep_distance(token) -> int:
    """
    Calculate dependency distance consistently (sentence-local, ROOT=0).
    
    Returns 0 for ROOT tokens or cross-sentence heads (for robustness).
    Otherwise returns |token.i - token.head.i|.
    
    Args:
        token: spaCy Token object
        
    Returns:
        Dependency distance as integer
    """
    # ROOT token has no dependency distance
    if token.head == token:
        return 0
    
    # Check if head is in the same sentence (for robustness)
    # Get sentence boundaries
    token_sent = None
    head_sent = None
    
    for sent in token.doc.sents:
        if sent.start <= token.i < sent.end:
            token_sent = sent
        if sent.start <= token.head.i < sent.end:
            head_sent = sent
            
    # If cross-sentence dependency, return 0 for consistency
    if token_sent != head_sent:
        return 0
        
    # Otherwise return linear distance
    return abs(token.i - token.head.i)


def dep_depth(token) -> int:
    """
    Calculate dependency depth consistently (sentence-local, ROOT=0).
    
    Returns the number of steps to ROOT within the sentence.
    ROOT tokens have depth 0.
    
    Args:
        token: spaCy Token object
        
    Returns:
        Dependency depth as integer
    """
    depth = 0
    current = token
    
    # Get sentence boundaries for this token
    token_sent = None
    for sent in token.doc.sents:
        if sent.start <= token.i < sent.end:
            token_sent = sent
            break
    
    # Traverse up the dependency tree
    while current.head != current:  # Until we reach ROOT
        depth += 1
        current = current.head
        
        # Check if we've moved outside the sentence
        if token_sent:
            if not (token_sent.start <= current.i < token_sent.end):
                # Head is outside sentence, treat as sentence root
                break
        
        # Prevent infinite loops
        if depth > 50:
            break
            
    return depth


def is_punctuation_token(token) -> bool:
    """
    Check if a token should be considered punctuation for feature exclusion.
    
    Args:
        token: spaCy Token object
        
    Returns:
        True if token is punctuation and should be excluded from predictors
    """
    return token.pos_ == "PUNCT" or token.is_punct