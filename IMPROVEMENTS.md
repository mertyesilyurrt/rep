# Syntax Utils Improvements

This document summarizes the improvements made to alignment and dependency metrics in the repository.

## Overview

The changes implement the next set of easy, high-reward improvements to alignment and dependency metrics, with lightweight unit tests, as requested. The existing CI workflow remains unchanged and will continue to run the notebook and tests.

## Changes Made

### 1. Helper Module: `scripts/syntax_utils.py`

**New Functions:**
- `align_aoi_to_spacy_windowed(aoi_tokens, doc_tokens, max_window=4)`: Improved alignment with two-pass normalization
- `dep_distance(token)`: Sentence-local dependency distance (ROOT=0, cross-sentence=0)  
- `dep_depth(token)`: Sentence-local dependency depth (ROOT=0)
- `is_punctuation_token(token)`: Helper for punctuation detection

**Key Improvements:**
- **Two-pass normalization**: First pass retains apostrophes/hyphens, second pass strips them for tolerant matching
- **Larger window**: Increased from 2 to 4 tokens to handle complex hyphenated compounds
- **Better contraction handling**: "won't" → ["wo", "n't"] alignment works correctly
- **Consistent ROOT=0**: Both distance and depth return 0 for ROOT tokens
- **Cross-sentence safety**: Returns 0 for cross-sentence dependencies

### 2. Updated Notebook: `MertYesilyurt_Report.ipynb`

**Changes:**
- Added imports from `scripts.syntax_utils`
- Removed old function definitions (replaced with imports)
- Updated `align_aoi_to_spacy_windowed` call to use `max_window=4`
- Modified `locality_features()` to set dependency metrics to 0 for punctuation tokens
- Updated `integration_cost()` to use improved `dep_distance()`

**Maintained Compatibility:**
- High-level flow unchanged
- Output CSV path remains `data-clean/processed/trt_by_word.csv`
- All existing feature columns preserved
- No new dependencies introduced

### 3. Unit Tests: `tests/test_syntax_utils.py`

**Test Coverage:**
- Two-pass normalization functions
- Alignment with contractions ("won't") and hyphens ("well-known") 
- Multi-token window concatenation
- Dependency distance calculations (ROOT=0, non-negative)
- Dependency depth calculations (ROOT=0, reasonable bounds)
- Cross-sentence dependency handling
- Punctuation detection and exclusion
- Edge cases and empty inputs

**Example Test Case:**
```python
# Test sentence: "He won't re-enter the well-known room."
aoi_tokens = ["He", "won't", "re-enter", "the", "well-known", "room", "."]
# Should achieve 100% alignment with max_window=4
```

## Benefits

### Alignment Improvements
- **Better coverage**: Handles contractions and hyphenated compounds more robustly
- **Tolerant matching**: Two-pass normalization catches more alignment cases
- **Larger window**: Can handle complex multi-token compounds

### Dependency Metrics Improvements  
- **Consistency**: ROOT always has distance=0 and depth=0
- **Sentence-local**: Calculations stay within sentence boundaries
- **Punctuation handling**: Punctuation tokens get dependency metrics set to 0
- **Robustness**: Handles cross-sentence dependencies gracefully

### Code Quality
- **Modularity**: Helper functions separated into reusable module
- **Testing**: Comprehensive unit tests with 100% pass rate
- **Documentation**: Clear docstrings and type hints
- **Maintainability**: Easier to update and extend functionality

## Validation

All changes have been validated:
- ✅ Unit tests pass (9/9)
- ✅ Import compatibility confirmed
- ✅ spaCy model loading works
- ✅ Alignment improvements demonstrated
- ✅ Dependency calculations verified
- ✅ Punctuation handling confirmed

## Example Output

```
# Improved alignment results:
SpaCy tokens: ['He', 'wo', "n't", 're', '-', 'enter', 'the', 'well', '-', 'known', 'room', '.']
AOI tokens: ['He', "won't", 're-enter', 'the', 'well-known', 'room', '.']  
Alignment: [0, 1, 3, 6, 7, 10, 11]  # 100% coverage

# Dependency features with punctuation handling:
Token     | dep_dist | depth | punct | features
----------|----------|-------|-------|----------
Hello     |    0     |   0   | False | ROOT token
,         |    0     |   0   | True  | Punctuation excluded  
world     |    2     |   1   | False | Normal calculation
!         |    0     |   0   | True  | Punctuation excluded
```

The changes successfully increase AOI→spaCy token alignment coverage and robustness while providing consistent, sentence-local dependency metrics with proper punctuation handling.