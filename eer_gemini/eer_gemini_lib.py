#!/usr/bin/env python3
"""
Enhanced Entity Error Rate (EER) Calculator using Gemini LLM as NER
Calculates EER similar to WER with Substitutions, Insertions, and Deletions
Processes all calls in a directory and calculates comprehensive EER metrics
Install: pip install google-generativeai python-dotenv jiwer
Set environment variable: GOOGLE_API_KEY=your_api_key
"""
import json
import argparse
import os
import sys
import time
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import google.generativeai as genai
from difflib import SequenceMatcher

from dotenv import load_dotenv
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

# Labels to consider for concerned entity counts
CONCERNED_LABELS = {"PERSON", "PRODUCT", "ORGANIZATION"}

# NER prompt template
NER_PROMPT = """You are an expert NER (named entity annotator). Please read the input text and extract ALL named entities. 

For each entity found, respond ONLY with a JSON array where each item has:
- "index": the word position in the text (starting from 0)
- "value": the exact entity text as it appears
- "type": the entity type (PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PRODUCT, EVENT, LANGUAGE, NORP, FAC, GPE, LOC, ORG)

Be comprehensive and extract ALL entities including:
- Person names (PERSON)
- Company/Organization names (ORGANIZATION)
- Locations, cities, countries (LOCATION/GPE)
- Dates and times (DATE/TIME)
- Money amounts (MONEY)
- Products or services (PRODUCT)
- Events (EVENT)
- Nationalities, religions, political groups (NORP)
- Facilities (FAC)
- Any other proper nouns

Input text: "{text}"

Respond with ONLY the JSON array, no other text:"""

def extract_entities_gemini(text: str, max_retries: int = 3) -> List[Dict]:
    """Extract entities using Gemini LLM"""
    if not text.strip():
        return []
        
    prompt = NER_PROMPT.format(text=text)
    time.sleep(2)  # Rate limiting
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            time.sleep(1)  # Rate limiting
            
            # Parse the response
            response_text = response.text.strip()
            
            # Clean up response if needed
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            entities_raw = json.loads(response_text)
            
            # Convert to our format
            entities = []            
            for ent in entities_raw:
                # Find the actual position in the original text
                entity_text = ent['value']
                start_pos = text.find(entity_text)
                
                if start_pos != -1:
                    entities.append({
                        "text": entity_text,
                        "type": ent['type'],
                        "start": start_pos,
                        "end": start_pos + len(entity_text),
                        "word_index": ent.get('index', -1)
                    })
            
            return entities
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(3)  # Wait before retry
            else:
                print(f"Failed to extract entities after {max_retries} attempts")
                return []

def load_bot_utterances(filepath: Path, speaker_label: str) -> List[str]:
    """Load bot utterances from JSON transcript"""
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist")
        return []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    utterances = []
    for entry in data:
        if entry.get('speaker', '').strip().lower() == speaker_label.strip().lower():
            content = entry.get('content', '').strip()
            if content:
                utterances.append(content)
    
    return utterances

def normalize_entity(text: str) -> str:
    """Normalize entity text for comparison"""
    return text.lower().strip()

def create_entity_sequences(entities: List[Dict]) -> Tuple[List[str], Dict[str, Dict]]:
    """Create sequences of normalized entities for alignment"""
    sequence = []
    entity_details = {}
    
    for i, ent in enumerate(entities):
        normalized = normalize_entity(ent['text'])
        key = f"{ent['type']}:{normalized}"
        sequence.append(key)
        entity_details[key] = {
            'text': ent['text'],
            'type': ent['type'],
            'normalized': normalized,
            'index': i
        }
    
    return sequence, entity_details

def calculate_eer_detailed(ref_entities: List[Dict], hyp_entities: List[Dict]) -> Dict:
    """Calculate detailed EER similar to WER with S, I, D counts"""
    
    # Create entity sequences for alignment
    ref_seq, ref_details = create_entity_sequences(ref_entities)
    hyp_seq, hyp_details = create_entity_sequences(hyp_entities)
    
    # Use SequenceMatcher to find alignment
    matcher = SequenceMatcher(None, ref_seq, hyp_seq)
    opcodes = matcher.get_opcodes()
    
    # Initialize counters
    substitutions = []
    insertions = []
    deletions = []
    correct = []
    
    # Process alignment operations
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Correct entities
            for i in range(i1, i2):
                correct.append({
                    'ref_entity': ref_details[ref_seq[i]],
                    'hyp_entity': hyp_details[ref_seq[i]]
                })
        elif tag == 'replace':
            # Substitutions
            ref_items = ref_seq[i1:i2]
            hyp_items = hyp_seq[j1:j2]
            for ref_item, hyp_item in zip(ref_items, hyp_items):
                substitutions.append({
                    'ref_entity': ref_details[ref_item],
                    'hyp_entity': hyp_details[hyp_item]
                })
            # Handle unequal lengths in substitution
            if len(ref_items) > len(hyp_items):
                for ref_item in ref_items[len(hyp_items):]:
                    deletions.append({'ref_entity': ref_details[ref_item]})
            elif len(hyp_items) > len(ref_items):
                for hyp_item in hyp_items[len(ref_items):]:
                    insertions.append({'hyp_entity': hyp_details[hyp_item]})
        elif tag == 'delete':
            # Deletions
            for i in range(i1, i2):
                deletions.append({'ref_entity': ref_details[ref_seq[i]]})
        elif tag == 'insert':
            # Insertions
            for j in range(j1, j2):
                insertions.append({'hyp_entity': hyp_details[hyp_seq[j]]})
    
    # Calculate metrics
    S = len(substitutions)  # Substitutions
    I = len(insertions)     # Insertions
    D = len(deletions)      # Deletions
    C = len(correct)        # Correct
    N = len(ref_entities)   # Total reference entities
    
    # EER calculation similar to WER
    eer = ((S + I + D) / N * 100) if N > 0 else 0
    
    return {
        'substitutions': S,
        'insertions': I,
        'deletions': D,
        'correct': C,
        'total_ref': N,
        'total_hyp': len(hyp_entities),
        'eer_percentage': round(eer, 2),
        'detailed_substitutions': substitutions,
        'detailed_insertions': insertions,
        'detailed_deletions': deletions,
        'detailed_correct': correct
    }

def filter_concerned_entities(entities: List[Dict]) -> List[Dict]:
    """Filter entities to only concerned types"""
    return [e for e in entities if e['type'] in CONCERNED_LABELS]

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
    """Print detailed EER results in a human-readable format"""
    
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
    
    # Detailed breakdown for all entities
    if all_eer['detailed_substitutions']:
        print(f"\n📝 SUBSTITUTIONS ({len(all_eer['detailed_substitutions'])}):")
        for i, sub in enumerate(all_eer['detailed_substitutions'][:10]):  # Limit to 10
            ref_ent = sub['ref_entity']
            hyp_ent = sub['hyp_entity']
            print(f"  {i+1}. '{ref_ent['text']}' ({ref_ent['type']}) → '{hyp_ent['text']}' ({hyp_ent['type']})")
        if len(all_eer['detailed_substitutions']) > 10:
            print(f"  ... and {len(all_eer['detailed_substitutions']) - 10} more")
    
    if all_eer['detailed_deletions']:
        print(f"\n❌ DELETIONS ({len(all_eer['detailed_deletions'])}):")
        for i, deletion in enumerate(all_eer['detailed_deletions'][:10]):
            ref_ent = deletion['ref_entity']
            print(f"  {i+1}. '{ref_ent['text']}' ({ref_ent['type']}) - MISSING in hypothesis")
        if len(all_eer['detailed_deletions']) > 10:
            print(f"  ... and {len(all_eer['detailed_deletions']) - 10} more")
    
    if all_eer['detailed_insertions']:
        print(f"\n➕ INSERTIONS ({len(all_eer['detailed_insertions'])}):")
        for i, insertion in enumerate(all_eer['detailed_insertions'][:10]):
            hyp_ent = insertion['hyp_entity']
            print(f"  {i+1}. '{hyp_ent['text']}' ({hyp_ent['type']}) - EXTRA in hypothesis")
        if len(all_eer['detailed_insertions']) > 10:
            print(f"  ... and {len(all_eer['detailed_insertions']) - 10} more")
    
    # Concerned Entities Results
    print(f"\n🎯 CONCERNED ENTITIES ANALYSIS (PERSON, ORG, PRODUCT):")
    print(f"{'─'*50}")
    print(f"Reference concerned entities: {concerned_eer['total_ref']}")
    print(f"Hypothesis concerned entities: {concerned_eer['total_hyp']}")
    print(f"Correct (C): {concerned_eer['correct']}")
    print(f"Substitutions (S): {concerned_eer['substitutions']}")
    print(f"Insertions (I): {concerned_eer['insertions']}")
    print(f"Deletions (D): {concerned_eer['deletions']}")
    print(f"Concerned EER = (S+I+D)/N = ({concerned_eer['substitutions']}+{concerned_eer['insertions']}+{concerned_eer['deletions']})/{concerned_eer['total_ref']} = {concerned_eer['eer_percentage']:.2f}%")
    
    # Detailed breakdown for concerned entities
    if concerned_eer['detailed_substitutions']:
        print(f"\n📝 CONCERNED SUBSTITUTIONS ({len(concerned_eer['detailed_substitutions'])}):")
        for i, sub in enumerate(concerned_eer['detailed_substitutions']):
            ref_ent = sub['ref_entity']
            hyp_ent = sub['hyp_entity']
            print(f"  {i+1}. '{ref_ent['text']}' ({ref_ent['type']}) → '{hyp_ent['text']}' ({hyp_ent['type']})")
    
    if concerned_eer['detailed_deletions']:
        print(f"\n❌ CONCERNED DELETIONS ({len(concerned_eer['detailed_deletions'])}):")
        for i, deletion in enumerate(concerned_eer['detailed_deletions']):
            ref_ent = deletion['ref_entity']
            print(f"  {i+1}. '{ref_ent['text']}' ({ref_ent['type']}) - MISSING in hypothesis")
    
    if concerned_eer['detailed_insertions']:
        print(f"\n➕ CONCERNED INSERTIONS ({len(concerned_eer['detailed_insertions'])}):")
        for i, insertion in enumerate(concerned_eer['detailed_insertions']):
            hyp_ent = insertion['hyp_entity']
            print(f"  {i+1}. '{hyp_ent['text']}' ({hyp_ent['type']}) - EXTRA in hypothesis")
    
    print(f"\n{'='*80}")

def process_single_call(call_dir: Path) -> Dict:
    """Process a single call directory"""
    start_time = time.time()
    print(f"\n[{call_dir.name}] Processing started at {time.strftime('%X')}")
    
    # Create output directory
    output_dir = call_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Load transcripts
    ref_path = call_dir / "ref_transcript.json"
    gt_path = call_dir / "gt_transcript.json"
    
    ref_speaker = "assistant"
    gt_speaker = "Agent"
    
    ref_utts = load_bot_utterances(ref_path, ref_speaker)
    gt_utts = load_bot_utterances(gt_path, gt_speaker)
    
    print(f"[{call_dir.name}] Loaded {len(ref_utts)} ref and {len(gt_utts)} gt utterances")
    
    if not ref_utts or not gt_utts:
        print(f"[{call_dir.name}] Warning: Empty utterances found")
        return create_empty_result(call_dir.name)
    
    # Concatenate utterances
    ref_text = " ".join(ref_utts)
    gt_text = " ".join(gt_utts)
    
    # Extract entities
    print(f"[{call_dir.name}] Extracting entities from reference text...")
    ref_entities = extract_entities_gemini(ref_text)
    
    print(f"[{call_dir.name}] Extracting entities from hypothesis text...")
    gt_entities = extract_entities_gemini(gt_text)
    
    print(f"[{call_dir.name}] Found {len(ref_entities)} ref and {len(gt_entities)} gt entities")
    
    # Calculate EER for all entities
    all_eer = calculate_eer_detailed(ref_entities, gt_entities)
    
    # Calculate EER for concerned entities only
    ref_concerned = filter_concerned_entities(ref_entities)
    gt_concerned = filter_concerned_entities(gt_entities)
    concerned_eer = calculate_eer_detailed(ref_concerned, gt_concerned)
    
    print(f"[{call_dir.name}] Found {len(ref_concerned)} concerned ref and {len(gt_concerned)} concerned gt entities")
    
    # Entity type distributions
    ref_types = defaultdict(int)
    gt_types = defaultdict(int)
    
    for e in ref_entities:
        ref_types[e['type']] += 1
    for e in gt_entities:
        gt_types[e['type']] += 1
    
    # Get mispronunciation data
    mispronunciation_count, pronunciation_ids = get_mispronunciation_data(call_dir)
    
    # Print detailed results
    print_detailed_eer_results(call_dir.name, all_eer, concerned_eer)
    
    # Create comprehensive results
    results = {
        "call_id": call_dir.name,
        "method": "Gemini LLM as NER",
        "model": "gemini-1.5-flash",
        "processing_time": round(time.time() - start_time, 2),
        "all_entities_eer": all_eer,
        "concerned_entities_eer": concerned_eer,
        "entity_type_distribution": {
            "reference": dict(ref_types),
            "hypothesis": dict(gt_types)
        },
        "mispronunciation": {
            "count": mispronunciation_count,
            "pronunciation_ids": pronunciation_ids
        },
        "entity_types_found": {
            "reference": list(set(e['type'] for e in ref_entities)),
            "hypothesis": list(set(e['type'] for e in gt_entities))
        },
        "raw_entities": {
            "reference": ref_entities,
            "hypothesis": gt_entities,
            "concerned_reference": ref_concerned,
            "concerned_hypothesis": gt_concerned
        }
    }
    
    # Save detailed results for this call
    output_file = output_dir / "eer_detailed_results_lib.json"
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
    print(f"SUMMARY TABLE - EER RESULTS FOR ALL CALLS")
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
        description="Enhanced EER Calculator with S/I/D analysis - processes all calls"
    )
    parser.add_argument("calls_dir", help="Path to calls directory containing subdirectories")
    parser.add_argument("--output", default="eer_detailed_comprehensive_results_lib.json", 
                       help="Output JSON file for detailed results")
    parser.add_argument("--csv-output", default="eer_detailed_metrics_summary_lib.csv",
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
    global_ref_utts = []
    global_gt_utts = []
    
    # Process each call directory
    call_dirs = sorted([d for d in calls_dir.glob("*") if d.is_dir() 
                       and (d / "ref_transcript.json").exists() 
                       and (d / "gt_transcript.json").exists()])
    
    print(f"Found {len(call_dirs)} valid call directories to process")
    
    for call_dir in call_dirs:
        try:
            # Load utterances for global aggregation
            ref_path = call_dir / "ref_transcript.json"
            gt_path = call_dir / "gt_transcript.json"
            ref_speaker = "assistant"
            gt_speaker = "Agent"
            ref_utts = load_bot_utterances(ref_path, ref_speaker)
            gt_utts = load_bot_utterances(gt_path, gt_speaker)
            # Aggregate utterances for Global EER
            global_ref_utts.extend(ref_utts)
            global_gt_utts.extend(gt_utts)

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
    
    # Save comprehensive results
    comprehensive_results = {
        "processing_summary": {
            "total_calls": len(call_dirs),
            "successful_calls": len(successful_results),
            "total_processing_time": round(total_time, 2),
            "average_time_per_call": round(total_time / len(call_dirs), 2) if call_dirs else 0
        },
        "aggregate_metrics": {
            "average_all_entities_eer": round(avg_all_eer, 2),
            "average_concerned_entities_eer": round(avg_con_eer, 2),
            "total_ref_entities": sum(r["all_entities_eer"]["total_ref"] for r in successful_results),
            "total_hyp_entities": sum(r["all_entities_eer"]["total_hyp"] for r in successful_results),
            "total_concerned_ref": sum(r["concerned_entities_eer"]["total_ref"] for r in successful_results),
            "total_concerned_hyp": sum(r["concerned_entities_eer"]["total_hyp"] for r in successful_results)
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

    # Calculate and print Global EER across all calls
    print("\n=== GLOBAL EER ACROSS ALL CALLS ===")
    global_ref_text = " ".join(global_ref_utts)
    global_gt_text = " ".join(global_gt_utts)
    # Extract entities and compute EER
    global_ref_entities = extract_entities_gemini(global_ref_text)
    global_gt_entities = extract_entities_gemini(global_gt_text)
    global_all_eer = calculate_eer_detailed(global_ref_entities, global_gt_entities)
    print(f"Global EER = {global_all_eer['eer_percentage']:.2f}%  (S={global_all_eer['substitutions']}, I={global_all_eer['insertions']}, D={global_all_eer['deletions']}, N={global_all_eer['total_ref']})")

    # Compute and print Global Concerned Entities EER
    global_ref_concerned = filter_concerned_entities(global_ref_entities)
    global_gt_concerned = filter_concerned_entities(global_gt_entities)
    global_concerned_eer = calculate_eer_detailed(global_ref_concerned, global_gt_concerned)
    if global_concerned_eer["total_ref"] > 0:
        print(f"Global Concerned EER = {global_concerned_eer['eer_percentage']:.2f}%  (S={global_concerned_eer['substitutions']}, I={global_concerned_eer['insertions']}, D={global_concerned_eer['deletions']}, N={global_concerned_eer['total_ref']})")
    else:
        print("⚠️ Skipping Global Concerned EER calculation: No concerned reference entities.")

    # Print final summary
    print(f"\n✅ Processing Complete!")
    print(f"📊 Processed {len(call_dirs)} calls in {total_time:.2f}s")
    print(f"📈 Average EER: {comprehensive_results['aggregate_metrics']['average_all_entities_eer']:.2f}%")
    print(f"📁 Detailed results: {args.output}")
    print(f"📋 CSV summary: {args.csv_output}")

if __name__ == "__main__":
    main()