### Entity Error Rate (EER) evaluation – project overview

This repository evaluates Entity Error Rate (EER), analogous to Word Error Rate (WER), across calls. It supports three families of approaches and produces both per‑call artifacts and global summaries.

- **Library NER only**: extract entities using libraries (SpaCy for English, Stanza for Hindi), normalize lightly, then compute EER via set alignment. Script: `eer_spacy/eer_spacy.py`.
- **LLM as NER with precise normalization and alignment (OpenAI)**: ask the model to return entities, perform strong canonicalization, and compute EER (S/I/D). Scripts: `eer_gpt/eer_gpt_lib.py`, `eer_gpt/eer_gpt_llm.py`.
- **LLM as NER with precise normalization and alignment (Gemini)**: same idea using Gemini. Scripts: `eer_gemini/eer_gemini_lib.py`, `eer_gemini/eer_gemini_llm.py`.

Throughout, EER mirrors WER: EER = (S + I + D) / N_ref × 100, where S=Substitutions, I=Insertions, D=Deletions, N_ref=number of reference entities.


## Repository layout and outputs

Each method lives in its own top‑level directory with a consistent structure:

- `eer_spacy/`, `eer_gpt/`, `eer_gemini/`
  - `calls/` – one subfolder per call id
    - `annotated.json` – manual annotations (optional, may include mispronunciation notes)
    - `gt_transcript.json` – ground truth transcript (speaker label typically `Agent`)
    - `ref_transcript.json` – hypothesis/reference model transcript (speaker label typically `assistant`)
    - `output/` – per‑call outputs written by the script(s); e.g.:
      - SpaCy/Stanza: `eer_spacy.json`
      - GPT (library variant): `entity_comparison_gpt_lib.json`, `entities_lib.json`
      - GPT (llm variant): `entity_comparison_gpt_llm.json`, `entities_llm.json`
      - Gemini (library variant): `entity_comparison_gemini_lib.json`, `entities_gemini_lib.json`
      - Gemini (llm variant): `entity_comparison_gemini_llm.json`, `entities_gemini_llm.json`, `eer_detailed_results_enhanced_gemini_llm.json`
  - Method driver script(s): one or two Python files (documented below)
  - Global outputs (aggregated across all calls):
    - SpaCy/Stanza: CSV at calls root `entity_summary_spacy.csv`; terminal log `eer_spacy/output_spacy.txt`.
    - GPT (library variant): JSON `eer_gpt_lib.json`, CSV `eer_summary_gpt_lib.csv`; terminal log `eer_gpt/output_lib.txt`.
    - GPT (llm variant): JSON `global_eer_gpt_llm.json`, CSV `global_eer_summary_gpt_llm.csv`; terminal log `eer_gpt/output_llm.txt`.
    - Gemini (library variant): JSON `global_eer_gemini_lib.json`, CSV `eer_summary_gemini_lib.csv`; terminal log `eer_gemini/output_lib.txt`.
    - Gemini (llm variant): JSON `global_eer_gemini_llm.json`, CSV `eer_summary_gemini_llm.csv`; terminal log `eer_gemini/output_llm.txt`.

Notes:
- Filenames above reflect the current fixed naming in each script.
- Environment configuration is read from `.env` when present and standard environment variables (`OPENAI_API_KEY` for GPT; `GOOGLE_API_KEY` for Gemini). SpaCy/Stanza are local libraries and require models (e.g., `en_core_web_sm`).


## Normalization and EER/WER logic

All approaches share the same high‑level pipeline:
1) Load and filter utterances by speaker.
2) Extract entities (method‑specific) and normalize/canonicalize them.
3) Align entity sequences and compute EER.

Entity normalization differs by method:
- Library NER only (SpaCy/Stanza): normalize by lowercasing and trimming during comparison. Relevant labels are limited via `RELEVANT_LABELS = {"PERSON", "ORG", "PRODUCT"}`.
- LLM methods (GPT/Gemini): the prompt instructs the model to internally transliterate to Latin, normalize numbers (words→digits, remove thousands separators), strip diacritics, lowercase, remove titles for PERSON, collapse punctuation/whitespace, collapse acronym spellings (e.g., `a.t.u.n`→`atun`), and apply brand‑aware bucketing for ORG/PRODUCT during matching.

Phonetic equality (all methods):
- Before assigning S/I/D, tokens are checked for “sounds‑alike” equality using a fast, deterministic, cached pipeline:
  - Double Metaphone (primary+alternate) with an optional Jaro‑Winkler guard on codes.
  - If inconclusive, G2P (grapheme→phoneme) to ARPABET and phoneme‑level similarity.
  - If tokens sound alike, they are treated as exact matches (cost 0) in alignment to avoid inflating EER.
  - All phonetic steps degrade gracefully if optional deps are unavailable.

EER definition (WER‑style):
- EER = (S + I + D) / N_ref × 100.
- Concerned EER is computed on a subset of labels, typically `{"PERSON", "ORG", "ORGANIZATION", "PRODUCT"}`.

Alignment policy:
- Library NER only: greedy phonetic‑aware matching over normalized tokens; sounds‑alike are equal; remaining unmatched count as deletions/insertions.
- LLM methods: the LLM returns S/I/D counts and detailed lists; additionally, local helpers treat sounds‑alike tokens as equal in any auxiliary alignment to avoid over‑counting.


## Scripts and technical details

### eer_spacy/eer_spacy.py (Library NER only)

- Models and setup
  - Loads SpaCy `en_core_web_sm` as `nlp_en` and Stanza Hindi pipeline `tokenize,ner` as `stanza_hi`.
  - Relevant labels restricted via `RELEVANT_LABELS = {"PERSON","ORG","PRODUCT"}`.

- Key functions
  - `extract_entities(text)` – runs SpaCy on English text and Stanza on Hindi text; returns a unified list of entities with fields: `text`, `type`, `[start,end]`, `script`.
  - `load_bot_utterances(filepath, speaker_label)` – loads utterances from a transcript JSON filtered by `speaker`.
  - `normalize_entity(text)` – lowercases and trims for comparison.
  - `calculate_alignment_based_eer(ref_entities, gt_entities)` – greedy phonetic‑aware matching on normalized tokens (Double Metaphone → JW guard; fallback G2P ARPABET). Sounds‑alike are treated as equal; remaining unmatched form deletions/insertions. `eer_percentage = min(((missed + extra)/|ref|)×100, 100)` and accuracy as `|correct|/|ref|×100`.
  - `find_entity_errors_in_utterances(ref_utts, gt_utts)` – pairs utterances by index, extracts entities per utt, and returns first 10 pairs where entity sets differ.
  - `process_call(call_dir)` – loads transcripts, concatenates utterances, extracts entities, computes EER, writes per‑call `output/eer_spacy.json` and updates a global `stats` row.
  - `main()` – iterates all call folders under the provided root, writes `entity_summary_spacy.csv`, prints global aggregates and entity‑type histograms. Console is mirrored to `eer_spacy/output_spacy.txt`.

- Outputs (per call)
  - `output/eer_spacy.json` – detailed per‑call metrics, entity counts, type distributions, and sample error pairs.

- Global outputs
  - `entity_summary_spacy.csv` at the calls root – one row per call with entity counts and EER/accuracy.
  - Console summary mirrored to `eer_spacy/output_spacy.txt` includes global EER and Accuracy across all calls.


### eer_gpt/eer_gpt_lib.py (LLM as NER + LLM EER; OpenAI)

- Configuration
  - Requires `OPENAI_API_KEY` in the environment. Default model: `gpt-4o`, temperature 0.
  - Concerned labels: `CONCERNED_LABELS = {"PERSON","ORG","ORGANIZATION","PRODUCT"}`.

- Normalization utilities
  - Phonetic helpers: Double Metaphone + JW guard, optional G2P ARPABET; `sounds_alike(a,b)` treats sounds‑alike as equal in token comparisons.
  - `normalize_entity`, `pre_normalize_text_for_llm`, acronym collapse `collapse_acronym_like`, brand cleaning for ORG, product bucketing, and brand‑aware `bucket_entity` (used for display/debug and list building).

- Core LLM interactions
  - `LLM_EER_INSTRUCTIONS` – system prompt specifying extraction, canonicalization, and EER schema including detailed S/I/D lists.
  - `analyze_eer_via_llm(ref_text, hyp_text)` – sends both transcripts; expects JSON with `ref_entities`, `hyp_entities`, and `all_eer`/`concerned_eer` containing S/I/D/C/N/EER and details.
  - `_map_llm_eer_to_internal(e)` – converts model JSON to internal structure with `eer_percentage` normalized to percent.
  - `calculate_global_eer(all_ref_entities, all_hyp_entities)` – sends compact sequences `TYPE::canonical` to the LLM to compute global S/I/D over concatenated entities.

- Data I/O and filtering
  - `load_transcript_utterances`, `load_bot_utterances` – load/filter by speaker (`assistant` vs `agent`).
  - `filter_concerned_entities` – drops ACK tokens and medical conditions from the concerned subset.

- Orchestration
  - `process_single_call(call_dir)` – builds minimally pre‑normalized GT (Agent) and Hyp (Assistant) texts, calls `analyze_eer_via_llm`, prints and saves:
    - Per‑call comparison summary: `output/entity_comparison_gpt_lib.json`.
    - Persisted entity lists: `output/entities_lib.json`.
  - Token equality treats sounds‑alike as equal (cost 0) to avoid inflating EER in any auxiliary alignment.
  - `print_detailed_eer_results` – renders S/I/D details with canonical and bucket forms for quick inspection.
  - `print_summary_table(all_results)` – prints a compact table across calls.
  - `main()` – processes all calls, writes global summaries:
    - JSON: `eer_gpt_lib.json` (aggregate metrics, global EER, and all individual results)
    - CSV: `eer_summary_gpt_lib.csv`.
    - Console is mirrored to `eer_gpt/output_lib.txt`.


### eer_gpt/eer_gpt_llm.py (LLM as NER + LLM EER; OpenAI)

This variant mirrors `eer_gpt_lib.py` with the same normalization prompts and EER computation via the LLM. It exposes the same public functions and output shapes. Key outputs: per‑call `output/entity_comparison_gpt_llm.json` and `output/entities_llm.json`; global JSON `global_eer_gpt_llm.json` and CSV `global_eer_summary_gpt_llm.csv`. Console is mirrored to `eer_gpt/output_llm.txt`. Token equality uses the same phonetic pipeline (Double Metaphone → JW guard; fallback G2P) to treat sounds‑alike as equal.


### eer_gemini/eer_gemini_lib.py (LLM as NER + LLM EER; Gemini)

- Configuration
  - Requires `GOOGLE_API_KEY` in the environment; `genai.configure(api_key=...)` is used. Concerned labels are the same as in GPT variant.

- Normalization, prompts, and alignment logic
  - Mirrors the GPT `*_lib.py` variant: strong canonicalization (transliteration to Latin, numeric normalization, diacritics, lowercase, PERSON title stripping, acronym collapse), brand‑aware bucketing, compact global alignment via `TYPE::canonical` sequences computed by the LLM, and the same phonetic equality helpers for token comparisons.

- Orchestration and outputs
  - Same function set as the GPT `*_lib.py` variant. Key outputs: per‑call `output/entity_comparison_gemini_lib.json` and `output/entities_gemini_lib.json`; global JSON `global_eer_gemini_lib.json` and CSV `eer_summary_gemini_lib.csv`. Console is mirrored to `eer_gemini/output_lib.txt`.


### eer_gemini/eer_gemini_llm.py (LLM as NER + LLM EER; Gemini)

Functionally equivalent to `eer_gemini_lib.py` but structured like the GPT `*_llm.py` variant. It shares the same normalization rules, LLM EER response schema, orchestration (`process_single_call`, `main`), and uses the same phonetic equality helpers. Key outputs: per‑call `output/entity_comparison_gemini_llm.json`, `output/entities_gemini_llm.json`, and detailed `output/eer_detailed_results_enhanced_gemini_llm.json`; global JSON `global_eer_gemini_llm.json` and CSV `eer_summary_gemini_llm.csv`. Console is mirrored to `eer_gemini/output_llm.txt`.


## Per‑call and global artifacts (all methods)

- Per‑call in `calls/<call_id>/output/`:
  - Entities and EER per the method (see sections above).
  - LLM methods also persist raw entities for both transcripts (e.g., `entities_llm.json`, `entities_gemini_llm.json`). Some variants also emit a detailed per‑call results JSON (Gemini llm).

- Global (method directory root):
  - JSON: aggregated metrics and global EER across all calls.
  - CSV: compact metrics table.
  - `.txt` terminal captures: saved console output including per‑call S/I/D and the computed global EER.


## Environment and dependencies

- SpaCy model: install `en_core_web_sm` via: `python -m spacy download en_core_web_sm`.
- Stanza Hindi: models are downloaded on first run (`stanza.Pipeline(lang='hi', processors='tokenize,ner')`).
- OpenAI: set `OPENAI_API_KEY`.
- Gemini: set `GOOGLE_API_KEY`.
- See `requirements.txt` for Python package dependencies.


## Results table scaffold

Populate this table after running each method; fill using the global aggregates printed/written by each script (including S/I/D and N_ref from the console `.txt` captures and JSON summaries).

| Script | Method | Global N_ref | Global S | Global I | Global D | Global EER (%) | Concerned N_ref | Concerned S | Concerned I | Concerned D | Concerned EER (%) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `eer_spacy/eer_spacy.py` | Library NER only (SpaCy+Stanza, set‑based EER) |  |  |  |  | 29.65 |  |  |  |  | 94.24 |  |
| `eer_gpt/eer_gpt_lib.py` | LLM as NER (GPT), LLM computes EER | 310 | 10 | 5 | 5 | 6.45 | 200 | 15 | 5 | 5 | 12.50 |  |
| `eer_gpt/eer_gpt_llm.py` | LLM as NER (GPT), LLM computes EER | 292 | 12 | 4 | 4 | 5.48 | 57 | 5 | 3 | 2 | 17.54 |  |
| `eer_gemini/eer_gemini_lib.py` | LLM as NER (Gemini), LLM computes EER | 258 | 49 | 49 | 49 | 57.36 | 91 | 10 | 4 | 1 | 16.48 |  |
| `eer_gemini/eer_gemini_llm.py` | LLM as NER (Gemini), LLM computes EER | 408 | 40 | 10 | 10 | 12.25 | 248 | 29 | 13 | 10 | 21.77 |  |


## Notes on concerned labels and filtering

- Concerned labels default to PERSON, ORG/ORGANIZATION, PRODUCT.
- LLM variants filter out acknowledgements and medical condition phrases from the concerned subset.
- ORG bucketing strips legal suffixes (Inc, Ltd, Pvt, LLC, etc.) and leading articles; PRODUCT bucketing replaces `&` with `and` and collapses spacing.


## How to run (quick start)

- Library method:
  - `python eer_spacy/eer_spacy.py eer_spacy/calls` (adds `entity_summary_spacy.csv` in `eer_spacy/calls` and per‑call outputs under each call’s `output/`)

- GPT method(s):
  - `export OPENAI_API_KEY=...`
  - `python eer_gpt/eer_gpt_lib.py eer_gpt/calls` – writes per‑call `entity_comparison_gpt_lib.json`, `entities_lib.json`, and global `eer_gpt_lib.json` + `eer_summary_gpt_lib.csv`.
  - `python eer_gpt/eer_gpt_llm.py eer_gpt/calls` – writes per‑call `entity_comparison_gpt_llm.json`, `entities_llm.json`, and global `global_eer_gpt_llm.json` + `global_eer_summary_gpt_llm.csv`.

- Gemini method(s):
  - `export GOOGLE_API_KEY=...`
  - `python eer_gemini/eer_gemini_lib.py eer_gemini/calls` – writes per‑call `entity_comparison_gemini_lib.json`, `entities_gemini_lib.json`, and global `global_eer_gemini_lib.json` + `eer_summary_gemini_lib.csv`.
  - `python eer_gemini/eer_gemini_llm.py eer_gemini/calls` – writes per‑call `entity_comparison_gemini_llm.json`, `entities_gemini_llm.json` (and detailed `eer_detailed_results_enhanced_gemini_llm.json`), and global `global_eer_gemini_llm.json` + `eer_summary_gemini_llm.csv`.


## Implementation references (by function)

- SpaCy/Stanza method:
  - `extract_entities`, `calculate_alignment_based_eer`, `find_entity_errors_in_utterances`, `process_call`, `main`.
- GPT methods:
  - `normalize_entity`, `pre_normalize_text_for_llm`, `analyze_eer_via_llm`, `_map_llm_eer_to_internal`, `calculate_global_eer`, `extract_entities_gpt`, `filter_concerned_entities`, `print_detailed_eer_results`, `save_per_call_comparison_summary`, `process_single_call`, `print_summary_table`, `main`.
- Gemini methods:
  - Same as GPT, with provider configuration via `genai.configure`.

