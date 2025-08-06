#!/usr/bin/env python3
"""
Combined Entity Error Rate (EER) Analysis Script
Processes all calls and calculates EER using both SpaCy/Stanza NER and S+I+D/N formula
"""
import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from dotenv import load_dotenv
import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import spacy
import stanza

# Configuration
RELEVANT_LABELS = {"PERSON", "ORG", "PRODUCT"}

# Load environment
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Initialize NLP models
print("Loading NLP models...")
try:
    nlp_en = spacy.load("en_core_web_sm")
    print("✓ SpaCy English model loaded")
except OSError:
    print("⚠️  SpaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp_en = None

try:
    stanza_hi = stanza.Pipeline(lang='hi', processors='tokenize,ner', use_gpu=False)
    print("✓ Stanza Hindi model loaded")
except Exception as e:
    print(f"⚠️  Stanza Hindi model failed to load: {e}")
    stanza_hi = None

# Global statistics
stats = []
entity_type_counter = Counter()


def extract_entities(text: str) -> List[Dict]:
    """Extract entities from both English (spaCy) and Hindi (Stanza)."""
    entities = []

    if not text.strip():
        return entities

    # English NER
    if nlp_en:
        try:
            doc = nlp_en(text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "script": "latin"
                })
        except Exception as e:
            print(f"⚠️  SpaCy processing error: {e}")
    
    # Hindi NER
    if stanza_hi:
        try:
            stanza_doc = stanza_hi(text)
            for sent in stanza_doc.sentences:
                for ent in sent.ents:
                    entities.append({
                        "text": ent.text,
                        "type": ent.type,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "script": "devanagari"
                    })
        except Exception as e:
            print(f"⚠️  Stanza processing error: {e}")
    
    return entities


def load_bot_utterances(filepath: Path, speaker_label: str) -> List[str]:
    """Load bot utterances from JSON transcript"""
    if not filepath.exists():
        print(f"⚠️  File not found: {filepath}")
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {filepath}: {e}")
        return []
    
    utterances = []
    for entry in data:
        if entry.get('speaker') == speaker_label:
            content = entry.get('content', '').strip()
            if content:
                utterances.append(content)
    
    return utterances


def normalize_entity(text: str) -> str:
    """Normalize entity text for comparison"""
    return text.lower().strip()


def calculate_alignment_based_eer(ref_entities: List[Dict], gt_entities: List[Dict]) -> Dict:
    """
    Calculate EER using proper alignment-based method
    EER = (Missed + Extra) / Total_Reference_Entities * 100
    """
    # Convert entities to normalized sets for relevant types only
    ref_set = set(normalize_entity(ent['text']) for ent in ref_entities if ent['type'] in RELEVANT_LABELS)
    gt_set = set(normalize_entity(ent['text']) for ent in gt_entities if ent['type'] in RELEVANT_LABELS)
    
    # Calculate components
    correct_entities = len(ref_set & gt_set)  # Entities present in both
    missed_entities = len(ref_set - gt_set)   # Entities in ref but not in gt (deletions)
    extra_entities = len(gt_set - ref_set)    # Entities in gt but not in ref (insertions)
    
    total_ref_entities = len(ref_set)
    total_errors = missed_entities + extra_entities
    
    # EER calculation: (Missed + Extra) / Total_Reference * 100, capped at 100%
    raw_eer = (total_errors / total_ref_entities * 100) if total_ref_entities > 0 else 0
    eer_percentage = min(raw_eer, 100.0)
    # Accuracy calculation for validation
    accuracy = (correct_entities / total_ref_entities * 100) if total_ref_entities > 0 else 0
    return {
        "correct_entities": correct_entities,
        "missed_entities": missed_entities,
        "extra_entities": extra_entities,
        "total_errors": total_errors,
        "total_ref_entities": total_ref_entities,
        "total_gt_entities": len(gt_set),
        "eer_percentage": round(eer_percentage, 2),
        "accuracy_percentage": round(accuracy, 2)
    }




def find_entity_errors_in_utterances(ref_utts: List[str], gt_utts: List[str]) -> List[Dict]:
    """Find utterance pairs with entity errors"""
    error_pairs = []
    
    min_length = min(len(ref_utts), len(gt_utts))
    
    for i in range(min_length):
        ref_utt = ref_utts[i]
        gt_utt = gt_utts[i]
        
        ref_ents = extract_entities(ref_utt)
        gt_ents = extract_entities(gt_utt)
        
        # Filter for relevant entity types
        ref_ents_filtered = [e for e in ref_ents if e['type'] in RELEVANT_LABELS]
        gt_ents_filtered = [e for e in gt_ents if e['type'] in RELEVANT_LABELS]
        
        if ref_ents_filtered:
            # Check if there are entity differences
            ref_texts = set(normalize_entity(e['text']) for e in ref_ents_filtered)
            gt_texts = set(normalize_entity(e['text']) for e in gt_ents_filtered)
            
            if ref_texts != gt_texts:
                error_pairs.append({
                    "index": i,
                    "reference": ref_utt,
                    "ground_truth": gt_utt,
                    "ref_entities": [{"text": e['text'], "type": e['type']} for e in ref_ents_filtered],
                    "gt_entities": [{"text": e['text'], "type": e['type']} for e in gt_ents_filtered]
                })
    
    return error_pairs[:10]  # Return first 10 error pairs




def process_call(call_dir: Path) -> Dict:
    """Process a single call directory"""
    call_start = time.time()
    start_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🚀 Processing {call_dir.name} - start at {start_ts}")

    # Create output directory
    out_dir = call_dir / "output"
    out_dir.mkdir(exist_ok=True)

    # Define file paths
    ref_path = call_dir / "ref_transcript.json"
    gt_path = call_dir / "gt_transcript.json"

    # Load utterances
    ref_utts = load_bot_utterances(ref_path, "assistant")
    gt_utts = load_bot_utterances(gt_path, "Agent")

    if not ref_utts or not gt_utts:
        print(f"⚠️  Warning: Missing or empty transcripts for {call_dir.name}")
        return {}

    print(f"  📝 Loaded {len(ref_utts)} reference utterances")
    print(f"  📝 Loaded {len(gt_utts)} ground truth utterances")

    # Concatenate all utterances (handles alignment issues)
    ref_text = " ".join(ref_utts)
    gt_text = " ".join(gt_utts)

    # Extract entities
    print("  🔍 Extracting entities...")
    ref_entities = extract_entities(ref_text)
    gt_entities = extract_entities(gt_text)

    # Count entity types
    ref_types = Counter(e['type'] for e in ref_entities)
    gt_types = Counter(e['type'] for e in gt_entities)

    # Update global counter
    entity_type_counter.update(ref_types)
    entity_type_counter.update(gt_types)

    print(f"  📊 Found {len(ref_entities)} entities in reference")
    print(f"  📊 Found {len(gt_entities)} entities in ground truth")

    # Calculate alignment-based EER
    alignment_eer = calculate_alignment_based_eer(ref_entities, gt_entities)

    # Debug output for troubleshooting
    print(f"  🔍 Debug - Ref entities (relevant): {alignment_eer['total_ref_entities']}")
    print(f"  🔍 Debug - GT entities (relevant): {alignment_eer['total_gt_entities']}")
    print(f"  🔍 Debug - Correct: {alignment_eer['correct_entities']}")
    print(f"  🔍 Debug - Missed: {alignment_eer['missed_entities']}")
    print(f"  🔍 Debug - Extra: {alignment_eer['extra_entities']}")
    print(f"📊 Global EER: {alignment_eer['eer_percentage']:.2f}%")
    print(f"📊 Accuracy: {alignment_eer['accuracy_percentage']:.2f}%")
    print()

    # Find error pairs
    error_pairs = find_entity_errors_in_utterances(ref_utts, gt_utts)

    # Compute concerned entities (relevant types only)
    concerned_ref = [e['text'] for e in ref_entities if e['type'] in RELEVANT_LABELS]
    concerned_gt = [e['text'] for e in gt_entities if e['type'] in RELEVANT_LABELS]

    # Create results
    results = {
        "call_id": call_dir.name,
        "method": "Combined SpaCy/Stanza NER",
        "processing_timestamp": start_ts,
        "metrics": {
            "global_EER": alignment_eer
        },
        "entity_summary": {
            "reference_entities": len(ref_entities),
            "gt_entities": len(gt_entities),
            "unique_ref_entities": len(set(e['text'] for e in ref_entities)),
            "unique_gt_entities": len(set(e['text'] for e in gt_entities)),
            "concerned_ref_entities": len(list(set(concerned_ref))),
            "concerned_gt_entities": len(list(set(concerned_gt)))
        },
        "entity_type_distribution": {
            "reference": dict(ref_types),
            "ground_truth": dict(gt_types)
        },
        "sample_error_pairs": error_pairs,
        "entity_types_found": list(set(e['type'] for e in ref_entities + gt_entities))
    }

    # Save detailed results
    results_path = out_dir / "combined_eer_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Add to stats for CSV summary
    stats.append({
        "call_id": call_dir.name,
        "ref_total_entities": len(ref_entities),
        "ref_unique_entities": len(set(e['text'] for e in ref_entities)),
        "gt_total_entities": len(gt_entities),
        "gt_unique_entities": len(set(e['text'] for e in gt_entities)),
        "concerned_ref_entities": len(list(set(concerned_ref))),
        "concerned_gt_entities": len(list(set(concerned_gt))),
        "global_eer_percentage": alignment_eer["eer_percentage"],
        "global_correct": alignment_eer["correct_entities"],
        "global_missed": alignment_eer["missed_entities"],
        "global_extra": alignment_eer["extra_entities"],
        "accuracy_percentage": alignment_eer["accuracy_percentage"],
        "processing_duration": 0  # Will be updated below
    })

    call_end = time.time()
    duration = call_end - call_start
    stats[-1]["processing_duration"] = round(duration, 2)

    end_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"✅ [{call_dir.name}] ended at {end_ts}")
    print(f"🕒 [{call_dir.name}] duration: {duration:.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Combined EER Analysis for all calls")
    parser.add_argument("calls_root", help="Path to calls root directory")
    parser.add_argument("--output-csv", default="entity_analysis_summary.csv", 
                       help="Output CSV filename")
    
    args = parser.parse_args()
    
    calls_root = Path(args.calls_root)
    if not calls_root.exists():
        print(f"❌ Calls root directory not found: {calls_root}")
        sys.exit(1)
    
    script_start = time.time()
    print(f"📁 Processing calls from: {calls_root}")
    
    # Process all call directories
    processed_calls = 0
    for entry in calls_root.iterdir():
        if entry.is_dir():
            try:
                process_call(entry)
                processed_calls += 1
            except Exception as e:
                print(f"❌ Error processing {entry.name}: {e}")
    
    if processed_calls == 0:
        print("❌ No calls were processed successfully")
        sys.exit(1)
    
    # Write CSV summary
    csv_path = calls_root / args.output_csv
    fieldnames = [
        "call_id",
        "ref_total_entities",
        "ref_unique_entities", 
        "gt_total_entities",
        "gt_unique_entities",
        "concerned_ref_entities",
        "concerned_gt_entities",
        "global_eer_percentage",
        "global_correct",
        "global_missed",
        "global_extra",
        "accuracy_percentage",
        "processing_duration"
    ]
    
    with open(csv_path, "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    
    print(f"📝 Entity analysis summary written to {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    print(f"Total calls processed: {processed_calls}")
    
    # Compute global EER and accuracy across all calls
    if stats:
        total_ref = sum(s["ref_total_entities"] for s in stats)
        total_missed = sum(s["global_missed"] for s in stats)
        total_extra = sum(s["global_extra"] for s in stats)
        total_errors = total_missed + total_extra
        global_eer_all = min(total_errors / total_ref * 100, 100.0) if total_ref > 0 else 0.0
        total_correct = sum(s["global_correct"] for s in stats)
        global_accuracy = (total_correct / total_ref * 100) if total_ref > 0 else 0.0
        print(f"🧮 Global EER Across All Calls: {global_eer_all:.2f}%")
        print(f"🧮 Global Accuracy Across All Calls: {global_accuracy:.2f}%")
        # Global Concerned EER
        total_concerned_ref = sum(s["concerned_ref_entities"] for s in stats)
        total_concerned_gt = sum(s["concerned_gt_entities"] for s in stats)
        total_concerned_correct = total_correct  # assume same correct set is used
        total_concerned_errors = total_errors  # assuming only relevant entities are in stats

        global_concerned_eer = min(total_concerned_errors / total_concerned_ref * 100, 100.0) if total_concerned_ref > 0 else 0.0
        print(f"🧮 Global Concerned EER Across All Calls: {global_concerned_eer:.2f}%")
    
    # Print entity type distribution
    print("\n📊 Entity types encountered across all calls:")
    for ent_type, count in entity_type_counter.most_common():
        print(f"  - {ent_type}: {count}")
    
    script_end = time.time()
    total_duration = script_end - script_start
    print(f"\n⏳ Total script runtime: {total_duration:.2f}s 🎉")


if __name__ == "__main__":
    main()