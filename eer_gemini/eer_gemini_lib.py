#!/usr/bin/env python3
"""
Enhanced Entity Error Rate (EER) Calculator using GPT LLM as NER
Implements robust per-call and global entity extraction, normalization, and evaluation
Calculates EER similar to WER with Substitutions, Insertions, and Deletions
Processes all calls in a directory and calculates comprehensive EER metrics
"""

# Entity Canonicalization and Composite Handling for EER
#
# This module implements canonicalization logic for named entity matching and error rate evaluation (Entity Error Rate, EER).
# It ensures consistency in entity matching by normalizing surface-level variants and managing compound (composite) entities.
#
# Canonicalization Steps:
# 1. Casing: Convert all entity text to lowercase.
# 2. Punctuation Removal: Strip dots, commas, and non-alphanumeric characters unless meaningful.
# 3. Whitespace Normalization: Collapse multiple spaces into one.
# 4. Translation (if applicable): Translate GT entities to English for cross-lingual comparison.
# 5. Date Parsing: Convert date strings to ISO format.
# 6. Entity-Type Specific Rules: Apply additional normalizations depending on entity type.
#
# Composite Entity Handling:
# If a predicted entity's canonical form is a strict superset of multiple GT entities (across different types),
# and each of those GT entities was matched individually, then that predicted composite should not count as an insertion.
import json
import argparse
import os
import sys
import time
import csv
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import google.generativeai as genai
# LLM analysis is performed locally in this module to avoid cross-module imports

def ensure_latin(text: str) -> str:
    """
    No-op transliteration placeholder. Transliteration is delegated to the LLM.
    Returns the input unchanged.
    """
    return text or ""

# Note: No phonetic libraries; LLM handles normalization/phonetics

def normalize_entity(text: str) -> str:
    """
    Minimal, safe normalization for entity strings:
    - lowercase
    - trim
    - de-diacritic
    - collapse multiple spaces to a single space
    Does not transliterate, remove punctuation, or convert words-to-numbers.
    """
    if not text:
        return ""
    normalized = text.lower().strip()
    try:
        normalized = ''.join(
            c for c in unicodedata.normalize('NFD', normalized)
            if unicodedata.category(c) != 'Mn'
        )
    except Exception:
        # If unicode normalization fails, keep the original lowered text
        pass
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

# Intentionally no pre-LLM currency/number conversions to avoid redundancy with LLM

# --- Helpers: digits/brand extraction
def _surface_has_digits(s: str) -> bool:
    """Return True if the string contains any ASCII digit."""
    return bool(re.search(r'\d', s or ''))

def _clean_org_brand(canon: str) -> str:
    # Deprecated: brand-aware processing removed; keep minimal normalization
    return normalize_entity(canon or '')

def _extract_brand_from_product(product_canon: str, known_brands: Set[str]) -> Optional[str]:
    # Deprecated: brand-aware processing removed
    return None

# --- Token equivalence with spacing/hyphen collapse and phonetic forgiveness
def _collapse_spacing_hyphen(s: str) -> str:
    # remove spaces and hyphens for comparison like "elevate now" ≈ "elevatenow" and "elevate-now"
    return re.sub(r"[\s\-]+", "", s or "")

try:
    import jellyfish as _jf  # optional
except Exception:
    _jf = None

def _phonetic_match(a: str, b: str) -> bool:
    # healthifi ≈ healthify, rey ≈ ray
    if not a or not b:
        return False
    if _jf:
        try:
            m1, m2 = _jf.metaphone(a), _jf.metaphone(b)
            if m1 and m2 and m1 == m2:
                return True
            n1, n2 = _jf.nysiis(a), _jf.nysiis(b)
            if n1 and n2 and n1 == n2:
                return True
            # fallback similarity threshold
            return _jf.jaro_winkler_similarity(a, b) >= 0.94
        except Exception:
            pass
    # fallback basic ratio if jellyfish unavailable
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.96

def _split_type_and_value_token(t: str) -> Tuple[str, str]:
    if '::' in t:
        ttype, tval = t.split('::', 1)
        return (ttype or 'UNK'), (tval or '')
    return 'UNK', t or ''

def _tokens_equal(a: str, b: str) -> bool:
    # tokens are of the form TYPE::value; keep types strict
    ta, va = _split_type_and_value_token(a)
    tb, vb = _split_type_and_value_token(b)
    if ta != tb:
        return False
    if va == vb:
        return True
    # collapse spaces/hyphens for value comparison (elevate now ≈ elevatenow ≈ elevate-now)
    vac = _collapse_spacing_hyphen(va)
    vbc = _collapse_spacing_hyphen(vb)
    if vac == vbc:
        return True
    # phonetic forgiveness on collapsed forms (healthifi ≈ healthify, ray ≈ rey)
    return _phonetic_match(vac, vbc)

# --- Helpers: digits/brand extraction
def _surface_has_digits(s: str) -> bool:
    """Return True if the string contains any ASCII digit."""
    return bool(re.search(r'\d', s or ''))

def _clean_org_brand(canon: str) -> str:
    # Remove common legal/organizational suffixes and noise words
    suffixes = {
        'inc','inc.','ltd','ltd.','pvt','pvt.','private','limited','llc','corp','corp.','co','co.','company',
        'plc','gmbh','s.a.','sa','bv','ag','oy','srl','s.r.l.','k.k.','kk','pte','pte.','llp','llp.'
    }
    tokens = [t for t in (canon or '').split() if t]
    while tokens and tokens[-1].strip('.').lower() in suffixes:
        tokens.pop()
    if tokens and tokens[0].lower() in {'the','a','an'}:
        tokens = tokens[1:]
    brand = ' '.join(tokens).replace('&', ' and ')
    brand = re.sub(r'\s+', ' ', brand).strip()
    return brand

def _extract_brand_from_product(product_canon: str, known_brands: Set[str]) -> Optional[str]:
    """
    Try to find a brand token within a normalized product string using a known-brands set.
    Matches longest brand that appears as a whole-word substring.
    """
    s = normalize_entity((product_canon or '').replace('&', ' and '))
    brands = sorted(list(known_brands), key=len, reverse=True)
    for b in brands:
        if re.search(rf"(?:^|\b){re.escape(b)}(?:\b|$)", s):
            return b
    return None

def pre_normalize_text_for_llm(text: str) -> str:
    """
    Minimal pre-normalization before sending to the LLM.
    Only standardize Unicode form and collapse whitespace so offsets remain intuitive.
    Steps:
      - Unicode normalize (NFKC) to standardize symbols
      - Collapse all whitespace (including newlines/tabs) to single spaces
      - Strip leading/trailing spaces
    Note: No lowercasing, no diacritic stripping, no currency removal, no '&' replacement,
          and no words→number conversion. Those are handled (if needed) by the LLM per prompt.
    """
    if not isinstance(text, str) or not text:
        return ""
    s = unicodedata.normalize('NFKC', text)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def bucket_entity(entity: Dict) -> str:
    """
    Deprecated: bucketed matching disabled. Use normalized canonical/text for alignment.
    Additionally, collapse acronym-like spellings (e.g., "a t u n" or "a.t.u.n" → "atun").
    """
    text = entity.get('bucket_norm') or entity.get('canonical') or entity.get('text', '') or ''
    # Normalize, then collapse only if the token is purely single-letter separated (won't alter phrases)
    norm = normalize_entity(ensure_latin(text))
    # Normalize numeric tokens: strip thousand separators (commas/spaces), preserve decimals
    if re.fullmatch(r"[0-9][0-9,\.\s]*", norm):
        norm = re.sub(r"[\s,]+", "", norm)
    return collapse_acronym_like(norm)

def is_medical_condition(text: str) -> bool:
    """Heuristic detector for medical condition phrases to exclude from concerned set."""
    t = normalize_entity(text)
    keywords = {
        'obesity', 'pre obesity', 'pre-obesity', 'overweight', 'underweight',
        'deficiency', 'vitamin d', 'vitamin b12', 'vitamin b 12', 'iron deficiency', 'anemia',
        'diabetes', 'hypertension', 'high blood pressure', 'blood pressure', 'cholesterol',
        'thyroid', 'pcos', 'pcod', 'asthma', 'allergy', 'migraine'
    }
    return any(k in t for k in keywords)

def _entity_alignment_tokens(entities: List[Dict]) -> List[str]:
    tokens: List[str] = []
    for e in entities:
        etype = e.get('type', '') or 'UNK'
        etype_norm = 'ORG' if etype in {'ORG', 'ORGANIZATION'} else etype
        norm = bucket_entity(e)
        tokens.append(f"{etype_norm}::{norm}")
    return tokens

def _align_exact_ops(ref_tokens: List[str], hyp_tokens: List[str]) -> List[Tuple[str, int, int]]:
    """
    Compute alignment ops with fuzzy equality:
    - equality if tokens are exactly equal OR _tokens_equal(a,b) is True (e.g., BRAND typos with edit distance ≤1)
    - substitution cost = 0 when equal under fuzzy check; otherwise 1
    - insertion/deletion cost = 1
    Returns list of (op, i, j) with i/j being lengths of prefixes consumed.
    """
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    bt: List[List[Optional[str]]] = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        bt[i][0] = 'del'
    for j in range(1, n + 1):
        dp[0][j] = j
        bt[0][j] = 'ins'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a = ref_tokens[i - 1]
            b = hyp_tokens[j - 1]
            eq = _tokens_equal(a, b)
            sub_cost = 0 if eq else 1
            sub = dp[i - 1][j - 1] + sub_cost
            ins = dp[i][j - 1] + 1
            dele = dp[i - 1][j] + 1
            best = min(sub, ins, dele)
            dp[i][j] = best
            if best == sub:
                bt[i][j] = 'eq' if eq else 'sub'
            elif best == ins:
                bt[i][j] = 'ins'
            else:
                bt[i][j] = 'del'

    ops: List[Tuple[str, int, int]] = []
    i, j = m, n
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in ('eq', 'sub'):
            ops.append((op, i, j))
            i -= 1
            j -= 1
        elif op == 'del':
            ops.append(('del', i, j))
            i -= 1
        elif op == 'ins':
            ops.append(('ins', i, j))
            j -= 1
        else:
            break
    ops.reverse()
    return ops


from dotenv import load_dotenv
load_dotenv()

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)

# Default Gemini model
GEMINI_MODEL = "gemini-1.5-pro"

# Default temperature
GEMINI_TEMPERATURE = 0

# Labels to consider for concerned entity counts
CONCERNED_LABELS = {"PERSON", "ORG", "ORGANIZATION", "PRODUCT"}
OUTPUT_SUFFIX_DEFAULT = "_llm"


# LLM EER analysis prompt - LLM handles ALL normalization and EER
LLM_EER_INSTRUCTIONS = """
You are an expert evaluator. You will receive two transcripts (Reference and Hypothesis) as JSON input.

Step 1 — Extract entities from each transcript (ordered by appearance):
For every entity, return: text (verbatim from input), type (PERSON/ORG/ORGANIZATION/PRODUCT/LOCATION/GPE/LOC/DATE/TIME/MONEY/EVENT/LANGUAGE/NORP/FAC), canonical (normalized form).

CRITICAL: Apply these normalization steps to create the canonical form, then use the canonical forms for alignment:
1) Raw text preprocessing and encoding fixes
2) Transliterate to Latin characters when needed (preserve meaning; do not translate)
3) Normalize casing and punctuation; remove non-meaningful punctuation
4) Convert number words → digits (e.g., "thirty two point zero three" → "32.03")
5) Normalize numbers by removing thousand separators (e.g., 1,799 → 1799)
6) Remove currency words/symbols (MONEY only): INR/rupees/Rs/₹, etc.
7) Collapse whitespace; trim ends
8) Remove diacritics (é→e, ñ→n)
9) Lowercase
10) PERSON: remove titles/roles (Mr/Mrs/Dr/CEO)
11) Keep multi-word entities intact; do not split phrases
12) Acronym collapse: spellings like "a.t.u.n" / "a t u n" become "atun"

Alignment and matching policy for EER = (S+I+D)/N_ref:
- Match by canonical strings; prefer same entity types (ORG and ORGANIZATION are equivalent)
- Treat small numeric formatting differences as equal (e.g., 1,799 vs 1799)
- Treat acronym variants as equal (e.g., a.t.u.n / a t u n vs atun)
- Allow phonetic equivalence for PERSON names (e.g., "rey" ≈ "ray"); consider simple sound-alike matches
- Handle simple composites (single ↔ two tokens) for ORG/PRODUCT if identical canonicals after rules

Output JSON with this exact schema:
{
  "ref_entities": [ {"text":"...","type":"...","canonical":"..."}, ... ],
  "hyp_entities": [ {"text":"...","type":"...","canonical":"..."}, ... ],
  "all_eer": {
    "S": int, "I": int, "D": int, "C": int, "N_ref": int, "N_hyp": int, "EER": float,
    "detailed_substitutions": [ {"ref": {"text":"...","type":"...","canonical":"..."}, "hyp": {"text":"...","type":"...","canonical":"..."}} ],
    "detailed_deletions": [ {"text":"...","type":"...","canonical":"..."} ],
    "detailed_insertions": [ {"text":"...","type":"...","canonical":"..."} ]
  },
  "concerned_eer": {
    "S": int, "I": int, "D": int, "C": int, "N_ref": int, "N_hyp": int, "EER": float,
    "detailed_substitutions": [ {"ref": {"text":"...","type":"...","canonical":"..."}, "hyp": {"text":"...","type":"...","canonical":"..."}} ],
    "detailed_deletions": [ {"text":"...","type":"...","canonical":"..."} ],
    "detailed_insertions": [ {"text":"...","type":"...","canonical":"..."} ]
  }
}

Respond with ONLY the JSON object above, no other text.
"""



# NER prompt template (LLM performs normalization and returns tuple-style items with offsets)
NER_PROMPT = """You are an expert NER and text normalization engine for evaluating Entity Error Rate (EER).

## Processing order (MANDATORY)
- **Step 1 (internal ONLY):** Create a WORKING COPY of the input by first **transliterating all text to Latin** (when not already Latin) and then applying the normalization rules below. Do **not** print the working copy.
- **Step 2:** Perform NER **on the WORKING COPY** (the transliterated + normalized text) to decide spans and entity types.
- **Step 3:** For each detected entity, output:
  - the **original surface span** from the provided input (unchanged),
  - the **entity type**, 
  - **canonical_norm** (derived from the WORKING COPY),
  - **bucket_norm** (brand-aware token; see rules),
  - **[start, end] offsets** that refer to the **ORIGINAL input text**.

## Allowed entity types
[PERSON, ORGANIZATION, ORG, PRODUCT, LOCATION, GPE, LOC, DATE, TIME, MONEY, EVENT, LANGUAGE, NORP, FAC]

## Normalization rules (apply in this exact order on the WORKING COPY)
1) Transliteration: Convert all text to Latin (ASCII) when input uses another script.
   - Use standard English dictionary spellings when they exist; avoid raw phonetic letter-by-letter forms.
   - Hindi examples: "ब्लैक" → "black", "नेट" → "net", "ड्रेस" → "dress".
   - Example phrase: "ब्लैक नेट ड्रेस" → "black net dress".
2) Whitespace: Collapse multiple spaces/tabs/newlines to a single space; strip leading/trailing spaces.
3) Case normalization: Lowercase.
4) Number conversion: Convert number words to digits (e.g., "thirty two point zero three" → "32.03"); preserve decimal points exactly.
5) Currency removal (MONEY only): Remove currency words/symbols (e.g., INR, Rs, ₹, rupees) from MONEY entity normalization.
6) Diacritic removal: Remove accents/diacritics from normalized strings.
7) Multi-word phrases: Keep multi-word entities intact (do not split phrases).

PERSON extraction rules:
- Only extract proper names, not standalone titles/roles (e.g., "doctor", "agent", "sir", "madam", "mr.", "mrs.", "ms.", "ceo", "manager").
- If a title precedes a name (e.g., "Dr. Anita Rao"), keep the surface value as-is, but in canonical_norm and bucket_norm keep only the name (e.g., "anita rao").

### Brand-aware **bucket_norm** rules (token used for alignment)
- For **ORG/ORGANIZATION**:
  - Remove legal suffixes/noise tokens at the end: [inc, inc., ltd, ltd., pvt, pvt., private, limited, llc, corp, corp., co, co., company, plc, gmbh, s.a., sa, bv, ag, oy, srl, s.r.l., k.k., kk, pte, pte., llp, llp.]
  - Drop leading articles: [the, a, an].
  - Replace `&` with `and`.
  - Then apply: lowercase → de-diacritic → collapse spaces.
- For **PRODUCT**:
  - Replace `&` with `and`.
  - Then: lowercase → de-diacritic → collapse spaces.
- For **all other types**:
  - Just: lowercase → de-diacritic → collapse spaces.

## Offsets
- Provide **character offsets** `[start, end]` into the exact **ORIGINAL** input text (0-based, end-exclusive).
- If an entity occurs multiple times, return **each occurrence** separately with its own offsets.

## Output format (STRICT)
- Output **only** a JSON array.
- Each item is a 5-element array:
  1) surface value (exact substring from the **ORIGINAL** input)
  2) type (one of the allowed types above)
  3) canonical_norm (string after normalization from the WORKING COPY)
  4) bucket_norm (brand-aware bucket string as specified)
  5) [start, end] character offsets into the **ORIGINAL** input text
- The entities **must be in order of first occurrence** in the transcript.

## Examples (illustrative; do not include in your output)
Input: "... मैंने मिंत्रा ऐप पर Super X & Y प्लान देखा ..."
Possible items:
[
  "मिंत्रा", "ORG",
  "myntra",
  "myntra",
  [10, 16]
]
[
  "Super X & Y", "PRODUCT",
  "super x and y",
  "super x and y",
  [22, 34]
]

## IMPORTANT
- Use the WORKING COPY (Latin + normalized) only to **decide** entities; but **always** return surface spans and offsets from the **ORIGINAL** input text.
- Do not invent or expand abbreviations beyond the text.
- If you are unsure of a type, choose the closest from the allowed list.
- Return **only** the JSON array; no prose, no Markdown.

## INPUT TEXT
{transcript_text}"""

def analyze_eer_via_llm(ref_text: str, hyp_text: str, max_retries: int = 3) -> Dict:
    """
    Ask the LLM to perform entity extraction, full normalization, and EER computation.
    Returns a dict with keys: ref_entities, hyp_entities, all_eer, concerned_eer.
    """
    if not (isinstance(ref_text, str) and isinstance(hyp_text, str)):
        return {"ref_entities": [], "hyp_entities": [], "all_eer": {}, "concerned_eer": {}}

    payload = {
        "reference_text": ref_text,
        "hypothesis_text": hyp_text
    }

    def _try_parse_json(s: str) -> Dict:
        t = s.strip()
        if t.startswith("```json"):
            t = t[7:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
        return json.loads(t)

    model = genai.GenerativeModel(GEMINI_MODEL)
    for attempt in range(max_retries):
        try:
            prompt = LLM_EER_INSTRUCTIONS + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)
            response = model.generate_content(prompt, generation_config={"temperature": GEMINI_TEMPERATURE})
            text = (response.text or "{}").strip()
            data = _try_parse_json(text)
            # Basic shape defaults
            data.setdefault("ref_entities", [])
            data.setdefault("hyp_entities", [])
            data.setdefault("all_eer", {"S":0, "I":0, "D":0, "C":0, "N_ref":0, "N_hyp":0, "EER":0.0,
                                           "detailed_substitutions":[], "detailed_deletions":[], "detailed_insertions":[]})
            data.setdefault("concerned_eer", {"S":0, "I":0, "D":0, "C":0, "N_ref":0, "N_hyp":0, "EER":0.0,
                                                "detailed_substitutions":[], "detailed_deletions":[], "detailed_insertions":[]})
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                print(f"LLM EER analysis failed: {e}")
                return {"ref_entities": [], "hyp_entities": [], "all_eer": {}, "concerned_eer": {}}

def _map_llm_eer_to_internal(e: Dict) -> Dict:
    """Map LLM 'all_eer'/'concerned_eer' to local display structure with eer_percentage and totals."""
    if not isinstance(e, dict):
        e = {}
    S = int(e.get("S", 0) or 0)
    I = int(e.get("I", 0) or 0)
    D = int(e.get("D", 0) or 0)
    C = int(e.get("C", 0) or 0)
    N_ref = int(e.get("N_ref", 0) or 0)
    N_hyp = int(e.get("N_hyp", 0) or 0)
    EER = e.get("EER", 0)
    try:
        eer_pct = float(EER)
        # If EER looks like ratio (<= 1.2), convert to percentage
        if eer_pct <= 1.2:
            eer_pct = round(eer_pct * 100.0, 2)
        else:
            eer_pct = round(eer_pct, 2)
    except Exception:
        eer_pct = 0.0
    return {
        "substitutions": S,
        "insertions": I,
        "deletions": D,
        "correct": C,
        "total_ref": N_ref,
        "total_hyp": N_hyp,
        "eer_percentage": eer_pct,
        "detailed_substitutions": e.get("detailed_substitutions", []),
        "detailed_insertions": e.get("detailed_insertions", []),
        "detailed_deletions": e.get("detailed_deletions", []),
    }

def calculate_global_eer(all_ref_entities: List[Dict], all_hyp_entities: List[Dict]) -> Dict:
    """Calculate global EER using LLM. Sends compact sequences to avoid token bloat."""
    def to_compact_sequence(ents: List[Dict]) -> List[str]:
        seq: List[str] = []
        for e in ents:
            et = e.get('type') or ''
            if et in {'ORG', 'ORGANIZATION'}:
                et = 'ORG'
            canon = e.get('canonical') or e.get('text') or ''
            seq.append(f"{et}::{canon}")
        return seq

    payload = {
        "reference_entities_compact": to_compact_sequence(all_ref_entities),
        "hypothesis_entities_compact": to_compact_sequence(all_hyp_entities)
    }

    instructions = (
        "You are an expert evaluator. You are given two COMPACT sequences: 'reference_entities_compact' and 'hypothesis_entities_compact'. "
        "Each item is 'TYPE::canonical'. Compute EER with the same matching policy (treat ORG and ORGANIZATION as equivalent, "
        "numeric formatting equivalence like 1,799≈1799, acronym collapse a.t.u.n/a t u n≈atun, phonetic equivalence for PERSON names rey≈ray). "
        "Count substitutions/insertions/deletions over alignment; N_ref is length of reference_entities_compact; N_hyp is length of hypothesis_entities_compact. "
        "Return ONLY JSON with keys: all_eer and concerned_eer, each containing S,I,D,C,N_ref,N_hyp,EER and detailed_substitutions/deletions/insertions."
    )

    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        prompt = instructions + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)
        response = model.generate_content(prompt, generation_config={"temperature": GEMINI_TEMPERATURE})
        text = (response.text or "{}").strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        data.setdefault("all_eer", {})
        data.setdefault("concerned_eer", {})
        return data
    except Exception as e:
        print(f"Global EER calculation failed: {e}")
        return {
            "all_eer": {"S":0, "I":0, "D":0, "C":0, "N_ref": len(all_ref_entities), "N_hyp": len(all_hyp_entities), "EER": 0.0},
            "concerned_eer": {"S":0, "I":0, "D":0, "C":0, "N_ref": len([x for x in all_ref_entities if x.get('type') in CONCERNED_LABELS]), "N_hyp": len([x for x in all_hyp_entities if x.get('type') in CONCERNED_LABELS]), "EER": 0.0}
        }

def extract_entities_gpt(text: str, speaker_filter: Optional[str] = None, 
                          type_filter: Set[str] = None, max_retries: int = 3) -> List[Dict]:
    """
    Enhanced entity extraction with speaker and type filtering
    
    Args:
        text: Input text or list of utterances
        speaker_filter: Filter utterances by speaker (if text contains utterance objects)
        type_filter: Set of entity types to include (e.g., {"PERSON", "ORG", "PRODUCT"})
        max_retries: Number of retry attempts
    
    Returns:
        Ordered list of entity dictionaries with at least 'text' and 'type' fields
    """
    if not text or (isinstance(text, str) and not text.strip()):
        return []
    
    # Handle case where text is actually utterance data
    if isinstance(text, list):
        # Filter by speaker if specified
        filtered_utterances = []
        for utterance in text:
            if speaker_filter is None or utterance.get('speaker', '').strip().lower() == speaker_filter.strip().lower():
                content = utterance.get('content', '').strip()
                if content:
                    filtered_utterances.append(content)
        
        # Concatenate filtered utterances
        text = " ".join(filtered_utterances)
    
    if not text.strip():
        return []
    
    # Helper: robust JSON array parsing from model outputs
    def _parse_json_array(resp: str) -> List[Dict]:
        s = resp.strip()
        if s.startswith("```json"):
            s = s[7:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            # Try slicing between first '[' and last ']'
            try:
                i = s.find('[')
                j = s.rfind(']')
                if i != -1 and j != -1 and j > i:
                    return json.loads(s[i:j+1])
            except Exception:
                pass
        raise
    
    # Extract entities using Gemini
    prompt = NER_PROMPT.format(transcript_text=text)
    time.sleep(2)  # Rate limiting
    
    model = genai.GenerativeModel(GEMINI_MODEL)
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, generation_config={"temperature": GEMINI_TEMPERATURE})
            response_text = (response.text or "").strip()

            # Parse JSON
            entities_raw = _parse_json_array(response_text)

            # Convert to our format (support array schema as specified)
            entities = []
            for ent in entities_raw:
                entity_text = None
                entity_type = None
                canonical_val = None
                bucket_norm = None
                start_pos = None
                end_pos = None

                if isinstance(ent, (list, tuple)) and len(ent) >= 5:
                    entity_text = ent[0]
                    entity_type = ent[1]
                    canonical_val = ent[2]
                    bucket_norm = ent[3]
                    offsets = ent[4]
                    if isinstance(offsets, (list, tuple)) and len(offsets) == 2:
                        start_pos, end_pos = int(offsets[0]), int(offsets[1])
                elif isinstance(ent, dict):
                    # Backward-compat dict schema
                    entity_text = ent.get('value') or ent.get('text')
                    entity_type = ent.get('type')
                    canonical_val = ent.get('canonical')
                    # Attempt to compute offsets if provided
                    offs = ent.get('offsets') or ent.get('span')
                    if isinstance(offs, (list, tuple)) and len(offs) == 2:
                        start_pos, end_pos = int(offs[0]), int(offs[1])

                if not entity_text or not entity_type:
                    continue
                
                # Apply type filtering if specified
                if type_filter and entity_type not in type_filter:
                    continue
                
                # Fallback to find if offsets missing
                if start_pos is None or end_pos is None:
                    idx = text.find(entity_text)
                    if idx != -1:
                        start_pos, end_pos = idx, idx + len(entity_text)
                    else:
                        # Skip if we cannot place it
                        continue

                norm_canon = normalize_entity(ensure_latin(canonical_val or entity_text))
                ent_obj = {
                        "text": entity_text,
                        "type": entity_type,
                        "start": start_pos,
                    "end": end_pos,
                    "word_index": -1,
                    "canonical": norm_canon
                }
                if bucket_norm:
                    ent_obj["bucket_norm"] = normalize_entity(ensure_latin(bucket_norm))
                entities.append(ent_obj)
            
            # No post-parse transliteration; rely on LLM outputs as-is
            # Return ordered list preserving repetitions
            return entities
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(3)  # Wait before retry
            else:
                # Fallback: chunk large text and aggregate entities
                try:
                    if isinstance(text, str) and len(text) > 4000:
                        chunk_size = 3000
                        all_ents: List[Dict] = []
                        offset = 0
                        start = 0
                        while start < len(text):
                            end = min(start + chunk_size, len(text))
                            # extend to whitespace
                            if end < len(text):
                                sp = text.rfind(' ', start, end)
                                if sp != -1 and sp > start + 1000:
                                    end = sp
                            chunk = text[start:end]
                            ents = extract_entities_gpt(chunk, speaker_filter=None, type_filter=type_filter, max_retries=3)
                            # shift positions by start index
                            for e in ents:
                                e['start'] = e.get('start', 0) + start
                                e['end'] = e.get('end', 0) + start
                                all_ents.append(e)
                            start = end
                        return all_ents
                except Exception as e2:
                    print(f"Chunked fallback failed: {e2}")
                print(f"Failed to extract entities after {max_retries} attempts")
                return []

def load_transcript_utterances(filepath: Path) -> List[Dict]:
    """Load transcript data as list of utterance objects"""
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist")
        return []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data if isinstance(data, list) else []

def load_bot_utterances(filepath: Path, speaker_label: str) -> List[str]:
    """Load bot utterances from JSON transcript (legacy function)"""
    utterances_data = load_transcript_utterances(filepath)
    
    utterances = []
    for entry in utterances_data:
        if entry.get('speaker', '').strip().lower() == speaker_label.strip().lower():
            content = entry.get('content', '').strip()
            if content:
                utterances.append(content)
    
    return utterances

def build_ordered_entity_lists(entities: List[Dict]) -> List[str]:
    """Build ordered, brand-aware bucketed list with minimal composite pre-merge."""
    return premerge_and_bucket_entities(entities)

def premerge_and_bucket_entities(entities: List[Dict]) -> List[str]:
    """
    One-pass preparation for matching without merging:
    - Sort entities by start
    - Do NOT merge ORG/PRODUCT; treat brands and products separately
    - Bucket each entity with brand-aware rules
    Returns ordered list of bucket strings
    """
    if not entities:
        return []

    def clean_org_brand(canon: str) -> str:
        # Remove common legal/organizational suffixes and noise words
        suffixes = {
            'inc','inc.','ltd','ltd.','pvt','pvt.','private','limited','llc','corp','corp.','co','co.','company',
            'plc','gmbh','s.a.','sa','bv','ag','oy','srl','s.r.l.','k.k.','kk','pte','pte.','llp','llp.'
        }
        tokens = [t for t in canon.split() if t]
        # strip trailing suffix tokens
        while tokens and tokens[-1].strip('.').lower() in suffixes:
            tokens.pop()
        # drop leading article
        if tokens and tokens[0].lower() in { 'the', 'a', 'an' }:
            tokens = tokens[1:]
        # normalize connectors
        brand = ' '.join(tokens).replace('&', ' and ')
        brand = re.sub(r'\s+', ' ', brand).strip()
        return brand

    def bucket_brand_aware(ent: Dict) -> str:
        etype = ent.get('type', '')
        canon = ent.get('canonical', ent.get('text', '')) or ''
        # Keep phrases intact for PRODUCT; make ORG brand-only by stripping legal suffixes
        if etype in { 'ORG', 'ORGANIZATION' }:
            return normalize_entity(clean_org_brand(canon))
        if etype == 'PRODUCT':
            canon_adj = canon.replace('&', ' and ')
            return normalize_entity(canon_adj)
        return normalize_entity(canon)

    # Sort by start
    sorted_ents = sorted([e for e in entities if isinstance(e, dict)], key=lambda e: e.get('start', 0))

    # No merging; bucket individually, then collapse acronym-like spellings
    buckets = [collapse_acronym_like(bucket_brand_aware(e)) for e in sorted_ents]
    return buckets

def collapse_acronym_like(bucket: str) -> str:
    """
    If the bucket looks like an acronym spelled as separate letters (e.g., "a t u n" or "a.t.u.n"),
    collapse to contiguous letters ("atun"). Otherwise return as-is (lightly normalized already).
    """
    if not bucket:
        return bucket
    norm = normalize_entity(bucket)
    tokens = re.sub(r"[\.]+", " ", norm).split()
    if tokens and all(len(tok) == 1 for tok in tokens):
        return "".join(tokens)
    # Also handle cases like "a t u n black net dress" → keep as is (not pure acronym)
    return norm

# --- Merge contiguous PRODUCT/ORG entities helper
def merge_contiguous_entities(entities: List[Dict]) -> List[Dict]:
    """
    Minimal merge: keep entities as-is. Composite handling is delegated to the aligner via windowing.
    Assumes input is already sorted by 'start'.
    """
    return list(entities or [])

# --- Global sequence-matching EER calculation
## Local EER calculation functions removed — LLM handles EER computation

def filter_concerned_entities(entities: List[Dict]) -> List[Dict]:
    """
    Filter entities to only concerned types (PERSON, ORG/ORGANIZATION),
    and select proper nouns (at least one token starting with uppercase or any non-ASCII).
    """
    filtered = []
    ack_stopwords = {
        'ok','okay','hmm','uh huh','uhhuh','haan','hanji','haanji','theek hai','thik hai','acha','achha'
    }
    for e in entities:
        if e['type'] in CONCERNED_LABELS:
            text = e.get('text', '').strip()
            if not text:
                continue
            # Drop obvious acknowledgements/non-entities after transliteration
            if normalize_entity(text) in ack_stopwords:
                continue
            # Exclude medical conditions from concerned set
            if is_medical_condition(text):
                continue
            tokens = text.split()
            def _looks_proper(word: str) -> bool:
                # Accept if starts with uppercase OR contains any uppercase letter (all-caps) OR contains any non-ASCII letter (e.g., Devanagari, Tamil, etc.)
                return (word and (word[0].isupper() or any(ch.isupper() for ch in word))) or any(ord(ch) > 127 for ch in word)
            if any(_looks_proper(tok) for tok in tokens if tok):
                filtered.append(e)
    return filtered

def get_mispronunciation_data(call_dir: Path) -> Tuple[int, str]:
    """Extract mispronunciation data from annotation files"""
    pronunciation_ids = []
    
    for file in call_dir.glob("*.json"):
        if file.name in ("label_studio.json", "annotated.json"):
            try:
                with open(file) as ls_file:
                    ls_data = json.load(ls_file)
                records = ls_data if isinstance(ls_data, list) else [ls_data]
                for record in records:
                    for ann in record.get("annotations", []):
                        for res in ann.get("result", []):
                            if res.get("from_name") == "pronunciation_note":
                                for txt in res.get("value", {}).get("text", []):
                                    id_part = txt.split("-", 1)[0].strip()
                                    if id_part:
                                        pronunciation_ids.append(id_part)
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"Warning: Could not parse {file}: {e}")
                continue
                
    pronunciation_ids = list(set(pronunciation_ids))
    return len(pronunciation_ids), ";".join(pronunciation_ids)

def print_detailed_eer_results(call_id: str, all_eer: Dict, concerned_eer: Dict):
    """Print detailed EER results in a human-readable format with normalization info"""
    
    print(f"\n{'='*80}")
    print(f"DETAILED EER ANALYSIS FOR CALL: {call_id}")
    print(f"{'='*80}")
    
    # All Entities Results
    print(f"\n🔍 ALL ENTITIES ANALYSIS:")
    print(f"{'─'*50}")
    print(f"Reference entities: {all_eer['total_ref']}")
    print(f"Hypothesis entities: {all_eer['total_hyp']}")
    print(f"Correct (C): {all_eer['correct']}")
    print(f"Substitutions (S): {all_eer['substitutions']}")
    print(f"Insertions (I): {all_eer['insertions']}")
    print(f"Deletions (D): {all_eer['deletions']}")
    print(f"EER = (S+I+D)/N = ({all_eer['substitutions']}+{all_eer['insertions']}+{all_eer['deletions']})/{all_eer['total_ref']} = {all_eer['eer_percentage']:.2f}%")
    
    # Detailed breakdown for all entities with normalization
    if all_eer['detailed_substitutions']:
        print(f"\n📝 SUBSTITUTIONS ({len(all_eer['detailed_substitutions'])}):")
        for i, sub in enumerate(all_eer['detailed_substitutions'][:10]):  # Limit to 10
            ref_ent = sub['ref']
            hyp_ent = sub['hyp']
            ref_orig = ref_ent.get('text', '')
            hyp_orig = hyp_ent.get('text', '')
            ref_canon = ref_ent.get('canonical', ref_orig)
            hyp_canon = hyp_ent.get('canonical', hyp_orig)
            ref_bucketed = bucket_entity(ref_ent)
            hyp_bucketed = bucket_entity(hyp_ent)
            score = sub.get('score', 0)
            print(f"  {i+1}. '{ref_orig}' → '{hyp_orig}'")
            print(f"      canonical: '{ref_canon}' → '{hyp_canon}'")
            print(f"      bucketed: '{ref_bucketed}' → '{hyp_bucketed}'")
        if len(all_eer['detailed_substitutions']) > 10:
            print(f"  ... and {len(all_eer['detailed_substitutions']) - 10} more")
    
    if all_eer['detailed_deletions']:
        print(f"\n❌ DELETIONS ({len(all_eer['detailed_deletions'])}):")
        for i, deletion in enumerate(all_eer['detailed_deletions'][:10]):
            ref_orig = deletion.get('text', deletion)
            ref_canon = deletion.get('canonical', ref_orig) if isinstance(deletion, dict) else ref_orig
            ref_norm = normalize_entity(ref_canon)
            print(f"  {i+1}. '{ref_orig}' → canonical: '{ref_canon}' → normalized: '{ref_norm}' - MISSING in hypothesis")
        if len(all_eer['detailed_deletions']) > 10:
            print(f"  ... and {len(all_eer['detailed_deletions']) - 10} more")
    
    if all_eer['detailed_insertions']:
        print(f"\n➕ INSERTIONS ({len(all_eer['detailed_insertions'])}):")
        for i, insertion in enumerate(all_eer['detailed_insertions'][:10]):
            hyp_orig = insertion.get('text', insertion)
            hyp_canon = insertion.get('canonical', hyp_orig) if isinstance(insertion, dict) else hyp_orig
            hyp_norm = normalize_entity(hyp_canon)
            print(f"  {i+1}. '{hyp_orig}' → canonical: '{hyp_canon}' → normalized: '{hyp_norm}' - EXTRA in hypothesis")
        if len(all_eer['detailed_insertions']) > 10:
            print(f"  ... and {len(all_eer['detailed_insertions']) - 10} more")
    
    # Concerned Entities Results
    print(f"\n🎯 CONCERNED ENTITIES ANALYSIS (PERSON, ORG):")
    print(f"{'─'*50}")
    print(f"Reference concerned entities: {concerned_eer['total_ref']}")
    print(f"Hypothesis concerned entities: {concerned_eer['total_hyp']}")
    print(f"Correct (C): {concerned_eer['correct']}")
    print(f"Substitutions (S): {concerned_eer['substitutions']}")
    print(f"Insertions (I): {concerned_eer['insertions']}")
    print(f"Deletions (D): {concerned_eer['deletions']}")
    print(f"Concerned EER = (S+I+D)/N = ({concerned_eer['substitutions']}+{concerned_eer['insertions']}+{concerned_eer['deletions']})/{concerned_eer['total_ref']} = {concerned_eer['eer_percentage']:.2f}%")

    # Detailed concerned substitutions (raw, canonical, normalized)
    if concerned_eer.get('substitutions', 0) > 0:
        subs_list = concerned_eer.get('detailed_substitutions', [])
        print(f"\n📝 CONCERNED SUBSTITUTIONS ({concerned_eer['substitutions']}):")
        if subs_list:
            for idx, sub in enumerate(subs_list, start=1):
                ref_ent = sub['ref']
                hyp_ent = sub['hyp']
                # score intentionally not printed
                ref_text = ref_ent.get('text', '')
                hyp_text = hyp_ent.get('text', '')
                ref_canon = ref_ent.get('canonical', ref_text)
                hyp_canon = hyp_ent.get('canonical', hyp_text)
                ref_norm = normalize_entity(ref_canon)
                hyp_norm = normalize_entity(hyp_canon)
                print(f"  {idx}. '{ref_text}' → '{hyp_text}'")
                print(f"      canonical: '{ref_canon}' → '{hyp_canon}'")
                print(f"      normalized: '{ref_norm}' → '{hyp_norm}'")
        else:
            print("  (No detailed substitutions captured)")
    
    if concerned_eer['detailed_deletions']:
        print(f"\n❌ CONCERNED DELETIONS ({len(concerned_eer['detailed_deletions'])}):")
        for i, deletion in enumerate(concerned_eer['detailed_deletions']):
            ref_orig = deletion.get('text', deletion)
            ref_canon = deletion.get('canonical', ref_orig) if isinstance(deletion, dict) else ref_orig
            ref_norm = normalize_entity(ref_canon)
            print(f"  {i+1}. '{ref_orig}' → canonical: '{ref_canon}' → normalized: '{ref_norm}' - MISSING in hypothesis")
    
    if concerned_eer['detailed_insertions']:
        print(f"\n➕ CONCERNED INSERTIONS ({len(concerned_eer['detailed_insertions'])}):")
        for i, insertion in enumerate(concerned_eer['detailed_insertions']):
            hyp_orig = insertion.get('text', insertion)
            hyp_canon = insertion.get('canonical', hyp_orig) if isinstance(insertion, dict) else hyp_orig
            hyp_norm = normalize_entity(hyp_canon)
            print(f"  {i+1}. '{hyp_orig}' → canonical: '{hyp_canon}' → normalized: '{hyp_norm}' - EXTRA in hypothesis")
    
    print(f"\n{'='*80}")

def save_per_call_comparison_summary(call_dir: Path, call_id: str, ref_entities: List[str], 
                                   hyp_entities: List[str], all_eer: Dict, concerned_eer: Dict, output_suffix: str = OUTPUT_SUFFIX_DEFAULT):
    """Save per-call entity comparison summary as specified in documentation"""
    output_dir = call_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    summary = {
        "call_id": call_id,
        "raw_data": {
            "ground_truth_entity_string": ", ".join(hyp_entities),
            "reference_entity_string": ", ".join(ref_entities)
        },
        "normalized_data": {},
        "all_entities_comparison": {
            "substitutions": all_eer.get('substitutions', 0) if isinstance(all_eer, dict) else 0,
            "insertions": all_eer.get('insertions', 0) if isinstance(all_eer, dict) else 0,
            "deletions": all_eer.get('deletions', 0) if isinstance(all_eer, dict) else 0,
            "correct_matches": all_eer.get('correct', 0) if isinstance(all_eer, dict) else 0,
            "detailed_substitutions": all_eer.get('detailed_substitutions', []),
            "detailed_insertions": all_eer.get('detailed_insertions', []),
            "detailed_deletions": all_eer.get('detailed_deletions', [])
        },
        "concerned_entities_comparison": {
            "substitutions": concerned_eer.get('substitutions', 0) if isinstance(concerned_eer, dict) else 0,
            "insertions": concerned_eer.get('insertions', 0) if isinstance(concerned_eer, dict) else 0,
            "deletions": concerned_eer.get('deletions', 0) if isinstance(concerned_eer, dict) else 0,
            "correct_matches": concerned_eer.get('correct', 0) if isinstance(concerned_eer, dict) else 0,
            "detailed_substitutions": concerned_eer.get('detailed_substitutions', []),
            "detailed_insertions": concerned_eer.get('detailed_insertions', []),
            "detailed_deletions": concerned_eer.get('detailed_deletions', [])
        }
    }
    
    summary_file = output_dir / f"entity_comparison_gpt_lib{output_suffix}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

def process_single_call(call_dir: Path, output_suffix: str = OUTPUT_SUFFIX_DEFAULT) -> Dict:
    """Process a single call directory using LLM-only normalization/alignment; save entity lists for global EER."""
    start_time = time.time()
    print(f"\n[{call_dir.name}] Processing started at {time.strftime('%X')}")
    
    # Create output directory
    output_dir = call_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Load transcript data
    ref_path = call_dir / "ref_transcript.json"
    gt_path = call_dir / "gt_transcript.json"
    
    ref_utterances_data = load_transcript_utterances(ref_path)
    gt_utterances_data = load_transcript_utterances(gt_path)
    
    print(f"[{call_dir.name}] Loaded {len(ref_utterances_data)} ref and {len(gt_utterances_data)} gt utterance records")
    
    if not ref_utterances_data or not gt_utterances_data:
        print(f"[{call_dir.name}] Warning: Empty utterance data found")
        return create_empty_result(call_dir.name)
    
    # Prepare raw texts and call LLM-only evaluator
    ref_text = " ".join([
        pre_normalize_text_for_llm(u.get('content',''))
        for u in ref_utterances_data
        if u.get('speaker','').strip().lower() == 'assistant' and u.get('content','').strip()
    ])

    hyp_text = " ".join([
        pre_normalize_text_for_llm(u.get('content',''))
        for u in gt_utterances_data
        if u.get('speaker','').strip().lower() == 'agent' and u.get('content','').strip()
    ])
    print(f"[{call_dir.name}] Calling LLM for end-to-end normalization/alignment...")
    llm_result = analyze_eer_via_llm(ref_text, hyp_text)
    # GT (agent) is reference; Assistant is hypothesis
    ref_entities = llm_result.get('hyp_entities', [])
    hyp_entities = llm_result.get('ref_entities', [])

    print(f"[{call_dir.name}] Found {len(ref_entities)} ref and {len(hyp_entities)} hyp entities")
    print(f"[{call_dir.name}] Note: entity offsets [start,end] refer to the minimally pre-normalized text sent to the LLM.")
    
    # Print all entities found with their locations and types
    print(f"\n📋 ALL ENTITIES FOUND IN GT (Reference) TRANSCRIPT:")
    print(f"{'─'*60}")
    for i, ent in enumerate(ref_entities):
        text = ent['text']
        entity_type = ent['type']
        start = ent.get('start', 'N/A')
        end = ent.get('end', 'N/A')
        canonical = ent.get('canonical', text)
        print(f"  {i+1:2d}. '{text}' (Type: {entity_type}, Position: {start}-{end})")
        print(f"       canonical: '{canonical}'")
    
    print(f"\n📋 ALL ENTITIES FOUND IN ASSISTANT (Hypothesis) TRANSCRIPT:")
    print(f"{'─'*60}")
    for i, ent in enumerate(hyp_entities):
        text = ent['text']
        entity_type = ent['type']
        start = ent.get('start', 'N/A')
        end = ent.get('end', 'N/A')
        canonical = ent.get('canonical', text)
        print(f"  {i+1:2d}. '{text}' (Type: {entity_type}, Position: {start}-{end})")
        print(f"       canonical: '{canonical}'")
    
    # Entity type distribution
    ref_type_dist = {}
    hyp_type_dist = {}
    for ent in ref_entities:
        ref_type_dist[ent['type']] = ref_type_dist.get(ent['type'], 0) + 1
    for ent in hyp_entities:
        hyp_type_dist[ent['type']] = hyp_type_dist.get(ent['type'], 0) + 1
    
    print(f"\n📊 ENTITY TYPE DISTRIBUTION:")
    print(f"{'─'*40}")
    print(f"Reference: {ref_type_dist}")
    print(f"Hypothesis: {hyp_type_dist}")
    
    # Filter for concerned entities
    ref_concerned = filter_concerned_entities(ref_entities)
    hyp_concerned = filter_concerned_entities(hyp_entities)
    
    print(f"[{call_dir.name}] Found {len(ref_concerned)} concerned ref and {len(hyp_concerned)} concerned hyp entities")
    
    # Use LLM-provided EER directly; map to local display structure
    all_eer = _map_llm_eer_to_internal(llm_result.get("all_eer", {}))
    concerned_eer = _map_llm_eer_to_internal(llm_result.get("concerned_eer", {}))

    # Skip local alignment debug; LLM provided detailed lists already
    
    # Build entity lists for summary
    ref_entity_list = build_ordered_entity_lists(ref_entities)
    hyp_entity_list = build_ordered_entity_lists(hyp_entities)
    
    # Save per-call comparison summary
    save_per_call_comparison_summary(call_dir, call_dir.name, ref_entity_list, 
                                   hyp_entity_list, all_eer, concerned_eer)

    # Persist LLM entity lists for global EER
    llm_entities_file = output_dir / f"llm_entities{output_suffix}.json"
    with open(llm_entities_file, 'w', encoding='utf-8') as f:
        json.dump({
            "ref_entities": ref_entities,
            "hyp_entities": hyp_entities
        }, f, indent=2, ensure_ascii=False)
    
    # Entity type distributions
    ref_types = defaultdict(int)
    hyp_types = defaultdict(int)
    
    for e in ref_entities:
        ref_types[e['type']] += 1
    for e in hyp_entities:
        hyp_types[e['type']] += 1
    
    # Get mispronunciation data
    mispronunciation_count, pronunciation_ids = get_mispronunciation_data(call_dir)
    
    # Print detailed results
    print_detailed_eer_results(call_dir.name, all_eer, concerned_eer)
    
    # Create comprehensive results
    results = {
        "call_id": call_dir.name,
        "method": "Enhanced GPT LLM as NER with Precise Normalization and Bucketing",
        "model": GPT_MODEL,
        "temperature": 0,
        "processing_time": round(time.time() - start_time, 2),
        "all_entities_eer": all_eer,
        "concerned_entities_eer": concerned_eer,
        "entity_type_distribution": {
            "reference": dict(ref_types),
            "hypothesis": dict(hyp_types)
        },
        "mispronunciation": {
            "count": mispronunciation_count,
            "pronunciation_ids": pronunciation_ids
        },
        "entity_types_found": {
            "reference": list(set(e['type'] for e in ref_entities)),
            "hypothesis": list(set(e['type'] for e in hyp_entities))
        },
        "raw_entities": {
            "reference": ref_entities,
            "hypothesis": hyp_entities,
            "concerned_reference": ref_concerned,
            "concerned_hypothesis": hyp_concerned
        },
        "entity_lists": {
            "reference_ordered": ref_entity_list,
            "hypothesis_ordered": hyp_entity_list
        }
    }
    
    # Save detailed results for this call
    output_file = output_dir / f"eer_detailed_results_enhanced_gpt_lib{output_suffix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"🕒 [{call_dir.name}] Completed in {elapsed:.2f}s")
    print(f"📊 All Entities EER: {all_eer['eer_percentage']:.2f}% | Concerned Entities EER: {concerned_eer['eer_percentage']:.2f}%")
    
    return results

def create_empty_result(call_id: str) -> Dict:
    """Create empty result structure for failed calls"""
    empty_eer = {
        'substitutions': 0, 'insertions': 0, 'deletions': 0, 'correct': 0,
        'total_ref': 0, 'total_hyp': 0, 'eer_percentage': 0,
        'detailed_substitutions': [], 'detailed_insertions': [], 'detailed_deletions': [], 'detailed_correct': []
    }
    
    return {
        "call_id": call_id,
        "all_entities_eer": empty_eer,
        "concerned_entities_eer": empty_eer,
        "mispronunciation": {"count": 0, "pronunciation_ids": ""},
        "processing_time": 0
    }

def print_summary_table(all_results: List[Dict]):
    """Print a summary table of all results"""
    print(f"\n{'='*120}")
    print(f"SUMMARY TABLE - ENHANCED EER RESULTS FOR ALL CALLS")
    print(f"{'='*120}")
    
    # Header
    header = f"{'Call ID':<15} {'All EER%':<10} {'C-EER%':<10} {'All S':<6} {'All I':<6} {'All D':<6} {'C-S':<5} {'C-I':<5} {'C-D':<5} {'Ref':<5} {'Hyp':<5} {'C-Ref':<6} {'C-Hyp':<6} {'Mispron':<8}"
    print(header)
    print('─' * 120)
    
    # Data rows
    for result in all_results:
        call_id = result['call_id']
        all_eer = result['all_entities_eer']
        con_eer = result['concerned_entities_eer']
        mispron = result['mispronunciation']['count']
        
        row = f"{call_id:<15} {all_eer['eer_percentage']:<10.2f} {con_eer['eer_percentage']:<10.2f} "
        row += f"{all_eer['substitutions']:<6} {all_eer['insertions']:<6} {all_eer['deletions']:<6} "
        row += f"{con_eer['substitutions']:<5} {con_eer['insertions']:<5} {con_eer['deletions']:<5} "
        row += f"{all_eer['total_ref']:<5} {all_eer['total_hyp']:<5} "
        row += f"{con_eer['total_ref']:<6} {con_eer['total_hyp']:<6} {mispron:<8}"
        
        print(row)
    
    print('─' * 120)
    print("Legend: C-EER% = Concerned Entities EER%, S = Substitutions, I = Insertions, D = Deletions, C-S/I/D = Concerned S/I/D")
    print("        Ref = Reference entities, Hyp = Hypothesis entities, C-Ref/Hyp = Concerned Ref/Hyp entities")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced EER Calculator with precise entity extraction and normalization - processes all calls"
    )
    parser.add_argument("calls_dir", help="Path to calls directory containing subdirectories")
    parser.add_argument("--output", default="eer_enhanced_comprehensive_results_gpt_lib.json", 
                       help="Output JSON file for detailed results")
    parser.add_argument("--csv-output", default="eer_enhanced_metrics_summary_gpt_lib.csv",
                       help="Output CSV file for metrics summary")

    args = parser.parse_args()
    
    calls_dir = Path(args.calls_dir)
    if not calls_dir.exists() or not calls_dir.is_dir():
        print(f"Error: {calls_dir} is not a valid directory")
        sys.exit(1)
    
    total_start = time.time()
    all_results = []
    metrics_summary = []

    # Initialize global utterance collections for Global EER
    global_ref_utterances_data = []
    global_gt_utterances_data = []
    
    # Process each call directory
    call_dirs = sorted([d for d in calls_dir.glob("*") if d.is_dir() 
                       and (d / "ref_transcript.json").exists() 
                       and (d / "gt_transcript.json").exists()])
    
    print(f"Found {len(call_dirs)} valid call directories to process")
    
    for call_dir in call_dirs:
        try:
            # Load utterances for global aggregation
            ref_utterances_data = load_transcript_utterances(call_dir / "ref_transcript.json")
            gt_utterances_data = load_transcript_utterances(call_dir / "gt_transcript.json")
            
            # Aggregate utterances for Global EER
            global_ref_utterances_data.extend(ref_utterances_data)
            global_gt_utterances_data.extend(gt_utterances_data)

            result = process_single_call(call_dir)
            all_results.append(result)

            # Extract summary metrics
            all_eer = result.get("all_entities_eer", {})
            con_eer = result.get("concerned_entities_eer", {})
            mispron = result.get("mispronunciation", {})

            metrics_summary.append({
                "call_id": result["call_id"],
                "all_entities_eer": all_eer.get("eer_percentage", 0),
                "all_substitutions": all_eer.get("substitutions", 0),
                "all_insertions": all_eer.get("insertions", 0),
                "all_deletions": all_eer.get("deletions", 0),
                "all_correct": all_eer.get("correct", 0),
                "all_ref_total": all_eer.get("total_ref", 0),
                "all_hyp_total": all_eer.get("total_hyp", 0),
                "concerned_entities_eer": con_eer.get("eer_percentage", 0),
                "concerned_substitutions": con_eer.get("substitutions", 0),
                "concerned_insertions": con_eer.get("insertions", 0),
                "concerned_deletions": con_eer.get("deletions", 0),
                "concerned_correct": con_eer.get("correct", 0),
                "concerned_ref_total": con_eer.get("total_ref", 0),
                "concerned_hyp_total": con_eer.get("total_hyp", 0),
                "mispronunciation_count": mispron.get("count", 0),
                "pronunciation_ids": mispron.get("pronunciation_ids", "")
            })

        except Exception as e:
            print(f"[{call_dir.name}] Error: {e}")
            empty_result = create_empty_result(call_dir.name)
            all_results.append(empty_result)

            # Add empty summary
            metrics_summary.append({
                "call_id": call_dir.name,
                "all_entities_eer": 0, "all_substitutions": 0, "all_insertions": 0, "all_deletions": 0,
                "all_correct": 0, "all_ref_total": 0, "all_hyp_total": 0,
                "concerned_entities_eer": 0, "concerned_substitutions": 0, "concerned_insertions": 0,
                "concerned_deletions": 0, "concerned_correct": 0, "concerned_ref_total": 0, "concerned_hyp_total": 0,
                "mispronunciation_count": 0, "pronunciation_ids": ""
            })
    
    total_time = time.time() - total_start
    
    # Print summary table
    print_summary_table(all_results)

    # Calculate aggregate metrics
    successful_results = [r for r in all_results if r.get("all_entities_eer", {}).get("total_ref", 0) > 0]
    avg_all_eer = sum(r["all_entities_eer"]["eer_percentage"] for r in successful_results) / len(successful_results) if successful_results else 0
    avg_con_eer = sum(r["concerned_entities_eer"]["eer_percentage"] for r in successful_results) / len(successful_results) if successful_results else 0
    
    # Calculate and print Global EER across all calls using concatenated LLM entity lists
    print("\n" + "="*80)
    print("GLOBAL EER ACROSS ALL CALLS (LLM-only normalization/alignment)")
    print("="*80)
    
    global_ref_entities: List[Dict] = []
    global_gt_entities: List[Dict] = []
    for cdir in call_dirs:
        try:
            with open(cdir / "output" / f"llm_entities{OUTPUT_SUFFIX_DEFAULT}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            global_ref_entities.extend(data.get('ref_entities', []))
            global_gt_entities.extend(data.get('hyp_entities', []))
        except Exception as e:
            print(f"[{cdir.name}] Warning: could not load LLM entities for global EER: {e}")
    
    # Calculate global EER via LLM using concatenated entity lists
    # Correct order: reference = GT entities, hypothesis = assistant entities
    global_llm = calculate_global_eer(global_ref_entities, global_gt_entities)
    global_all_eer_internal = _map_llm_eer_to_internal(global_llm.get("all_eer", {}))
    print(f"Global All Entities EER = {global_all_eer_internal['eer_percentage']:.2f}%")
    print(f"  (S={global_all_eer_internal['substitutions']}, I={global_all_eer_internal['insertions']}, D={global_all_eer_internal['deletions']}, C={global_all_eer_internal['correct']}, N={global_all_eer_internal['total_ref']})")

    # Calculate Global Concerned EER via LLM
    global_ref_concerned = filter_concerned_entities(global_ref_entities)
    global_gt_concerned = filter_concerned_entities(global_gt_entities)
    global_llm_con = calculate_global_eer(global_ref_concerned, global_gt_concerned)
    global_concerned_eer_internal = _map_llm_eer_to_internal(global_llm_con.get("concerned_eer", {}))
    print(f"Global Concerned Entities EER = {global_concerned_eer_internal['eer_percentage']:.2f}%")
    print(f"  (S={global_concerned_eer_internal['substitutions']}, I={global_concerned_eer_internal['insertions']}, D={global_concerned_eer_internal['deletions']}, C={global_concerned_eer_internal['correct']}, N={global_concerned_eer_internal['total_ref']})")
    
    # Print comprehensive global EER summary
    print(f"\n🌍 GLOBAL EER SUMMARY ACROSS ALL CALLS:")
    print(f"{'─'*60}")
    print(f"📊 ALL ENTITIES:")
    print(f"   Total Reference Entities (N_ref): {global_all_eer_internal['total_ref']}")
    print(f"   Total Hypothesis Entities: {global_all_eer_internal['total_hyp']}")
    print(f"   Substitutions (S): {global_all_eer_internal['substitutions']}")
    print(f"   Insertions (I): {global_all_eer_internal['insertions']}")
    print(f"   Deletions (D): {global_all_eer_internal['deletions']}")
    print(f"   Correct (C): {global_all_eer_internal['correct']}")
    print(f"   Global EER: {global_all_eer_internal['eer_percentage']:.2f}%")
    
    print(f"\n🎯 CONCERNED ENTITIES (PERSON, ORG):")
    print(f"   Total Reference Entities (N_ref): {global_concerned_eer_internal['total_ref']}")
    print(f"   Total Hypothesis Entities: {global_concerned_eer_internal['total_hyp']}")
    print(f"   Substitutions (S): {global_concerned_eer_internal['substitutions']}")
    print(f"   Insertions (I): {global_concerned_eer_internal['insertions']}")
    print(f"   Deletions (D): {global_concerned_eer_internal['deletions']}")
    print(f"   Correct (C): {global_concerned_eer_internal['correct']}")
    print(f"   Global Concerned EER: {global_concerned_eer_internal['eer_percentage']:.2f}%")
    
    # Save comprehensive results
    comprehensive_results = {
        "processing_summary": {
            "total_calls": len(call_dirs),
            "successful_calls": len(successful_results),
            "total_processing_time": round(total_time, 2),
            "average_time_per_call": round(total_time / len(call_dirs), 2) if call_dirs else 0,
            "method": "Enhanced GPT LLM as NER with Precise Normalization and Bucketing"
        },
        "aggregate_metrics": {
            "average_all_entities_eer": round(avg_all_eer, 2),
            "average_concerned_entities_eer": round(avg_con_eer, 2),
            "total_ref_entities": sum(r["all_entities_eer"]["total_ref"] for r in successful_results),
            "total_hyp_entities": sum(r["all_entities_eer"]["total_hyp"] for r in successful_results),
            "total_concerned_ref": sum(r["concerned_entities_eer"]["total_ref"] for r in successful_results),
            "total_concerned_hyp": sum(r["concerned_entities_eer"]["total_hyp"] for r in successful_results)
        },
        "global_eer": {
            "all_entities": global_llm.get("all_eer", {}),
            "concerned_entities": global_llm_con.get("concerned_eer", {}),
            "total_global_ref_entities": len(global_ref_entities),
            "total_global_hyp_entities": len(global_gt_entities),
            "total_global_concerned_ref": len(global_ref_concerned),
            "total_global_concerned_hyp": len(global_gt_concerned)
        },
        "normalization_settings": {
            "concerned_labels": list(CONCERNED_LABELS),
            "normalization_steps": [
                "Lowercase", 
                "Strip whitespace", 
                "Remove leading articles", 
                "Remove spaces/hyphens/underscores/dots",
                "Replace & with and",
                "Strip non-word characters",
                "Apply canonical mappings",
                "Remove diacritics/accents",
                "Transliterate if needed",
                "Apply product bucketing"
            ]
        },
        "individual_results": all_results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Save CSV summary
    fieldnames = [
        "call_id", "all_entities_eer", "all_substitutions", "all_insertions", "all_deletions", 
        "all_correct", "all_ref_total", "all_hyp_total",
        "concerned_entities_eer", "concerned_substitutions", "concerned_insertions", "concerned_deletions",
        "concerned_correct", "concerned_ref_total", "concerned_hyp_total",
        "mispronunciation_count", "pronunciation_ids"
    ]
    
    with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_summary)

    # Print final summary
    print(f"\n✅ Enhanced Processing Complete!")
    print(f"📊 Processed {len(call_dirs)} calls in {total_time:.2f}s")
    print(f"📈 Average Per-Call EER: {comprehensive_results['aggregate_metrics']['average_all_entities_eer']:.2f}%")
    print(f"🌍 Global EER: {global_all_eer_internal['eer_percentage']:.2f}%")
    print(f"📁 Detailed results: {args.output}")
    print(f"📋 CSV summary: {args.csv_output}")
    print(f"📋 Individual call summaries saved in each call's output/entity_comparison_gpt_lib{OUTPUT_SUFFIX_DEFAULT}.json")

if __name__ == "__main__":
    main()