#!/usr/bin/env python3
"""
Calculate Entity Error Rate (EER) using Stanford CoreNLP via Stanza
Install: pip install stanza
First run: stanza.download('en')
"""
import json
import argparse
from pathlib import Path
import stanza
from typing import List, Dict, Tuple
from collections import defaultdict

# Initialize Stanza pipeline
nlp = stanza.Pipeline('en', processors='tokenize,ner', use_gpu=False)

def extract_entities_stanford(text: str) -> List[Dict]:
    """Extract entities using Stanford NER via Stanza"""
    doc = nlp(text)
    entities = []
    
    for sentence in doc.sentences:
        for ent in sentence.ents:
            entities.append({
                "text": ent.text,
                "type": ent.type,
                "start": ent.start_char,
                "end": ent.end_char
            })
    
    return entities

def load_bot_utterances(filepath: Path, speaker_label: str) -> List[str]:
    """Load bot utterances from JSON transcript, handling both list and dict formats."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Determine where the entries list is
    if isinstance(data, dict) and "utterances" in data and isinstance(data["utterances"], list):
        entries = data["utterances"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unexpected transcript format in {filepath}")
    utterances = []
    for entry in entries:
        if entry.get('speaker') == speaker_label:
            content = entry.get('content', '').strip()
            if content:
                utterances.append(content)
    return utterances

def normalize_entity(text: str) -> str:
    """Normalize entity text for comparison"""
    return text.lower().strip()

def calculate_eer(ref_entities: List[Dict], gt_entities: List[Dict]) -> Tuple[int, int, List[Dict]]:
    """Calculate Entity Error Rate"""
    # Group entities by type
    ref_by_type = defaultdict(list)
    gt_by_type = defaultdict(list)
    
    for ent in ref_entities:
        ref_by_type[ent['type']].append({
            'normalized': normalize_entity(ent['text']),
            'original': ent['text']
        })
    
    for ent in gt_entities:
        gt_by_type[ent['type']].append(normalize_entity(ent['text']))
    
    missed_entities = []
    total_entities = 0
    
    # Check each reference entity
    for ent_type, ref_items in ref_by_type.items():
        gt_texts = gt_by_type.get(ent_type, [])
        
        for ref_item in ref_items:
            total_entities += 1
            if ref_item['normalized'] not in gt_texts:
                missed_entities.append({
                    "text": ref_item['original'],
                    "type": ent_type,
                    "status": "missed"
                })
    
    return len(missed_entities), total_entities, missed_entities

def find_entity_errors_in_utterances(ref_utts: List[str], gt_utts: List[str]) -> List[Dict]:
    """Find utterance pairs with entity errors"""
    error_pairs = []
    
    # Compare utterances where possible
    for i in range(min(len(ref_utts), len(gt_utts), 10)):  # Limit to first 10
        ref_utt = ref_utts[i]
        gt_utt = gt_utts[i]
        
        ref_ents = extract_entities_stanford(ref_utt)
        gt_ents = extract_entities_stanford(gt_utt)
        
        if ref_ents:
            ref_normalized = set(normalize_entity(e['text']) for e in ref_ents)
            gt_normalized = set(normalize_entity(e['text']) for e in gt_ents)
            
            if not ref_normalized.issubset(gt_normalized):
                error_pairs.append({
                    "index": i,
                    "reference": ref_utt,
                    "gtothesis": gt_utt,
                    "ref_entities": [{"text": e['text'], "type": e['type']} for e in ref_ents],
                    "gt_entities": [{"text": e['text'], "type": e['type']} for e in gt_ents],
                    "missed": list(ref_normalized - gt_normalized)
                })
    
    return error_pairs

def main():
    parser = argparse.ArgumentParser(description="Calculate EER using Stanford CoreNLP (Stanza)")
    parser.add_argument("ref_transcript", help="Path to reference transcript JSON")
    parser.add_argument("gt_transcript", help="Path to gtothesis transcript JSON")
    parser.add_argument("--output", default="eer_stanford_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    ref_speaker = "assistant"
    gt_speaker = "Agent"

    # Load utterances
    print("Loading transcripts...")
    ref_utts = load_bot_utterances(Path(args.ref_transcript), ref_speaker)
    gt_utts = load_bot_utterances(Path(args.gt_transcript), gt_speaker)
    # Collect utterances in a dict for clarity
    utterances_dict = {
        ref_speaker: ref_utts,
        gt_speaker:  gt_utts
    }
    
    print(f"Loaded {len(ref_utts)} reference utterances")
    print(f"Loaded {len(gt_utts)} gtothesis utterances")
    
    # Concatenate all utterances
    ref_text = " ".join(ref_utts)
    gt_text = " ".join(gt_utts)
    
    # Extract entities
    print("\nExtracting entities using Stanford CoreNLP (this may take a moment)...")
    ref_entities = extract_entities_stanford(ref_text)
    gt_entities = extract_entities_stanford(gt_text)
    
    print(f"Found {len(ref_entities)} entities in reference")
    print(f"Found {len(gt_entities)} entities in gtothesis")
    
    # Calculate EER
    missed, total, missed_list = calculate_eer(ref_entities, gt_entities)
    eer_percentage = (missed / total * 100) if total > 0 else 0
    
    # Find error pairs
    print("Finding error pairs...")
    error_pairs = find_entity_errors_in_utterances(ref_utts, gt_utts)
    
    # Entity type distribution
    ref_types = defaultdict(int)
    gt_types = defaultdict(int)
    
    for e in ref_entities:
        ref_types[e['type']] += 1
    for e in gt_entities:
        gt_types[e['type']] += 1
    
    # Create results
    results = {
        "method": "Stanford CoreNLP (Stanza)",
        "model": "en",
        "metrics": {
            "EER": {
                "missed_entities": missed,
                "total_entities": total,
                "percentage": round(eer_percentage, 2)
            }
        },
        "entity_summary": {
            "reference_entities": len(ref_entities),
            "gtothesis_entities": len(gt_entities),
            "unique_ref_entities": len(set(e['text'] for e in ref_entities)),
            "unique_gt_entities": len(set(e['text'] for e in gt_entities))
        },
        "entity_type_distribution": {
            "reference": dict(ref_types),
            "gtothesis": dict(gt_types)
        },
        "missed_entities": missed_list[:20],
        "sample_error_pairs": error_pairs,
        "entity_types_found": list(set(e['type'] for e in ref_entities))
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n=== EER RESULTS (Stanford CoreNLP) ===")
    print(f"EER: {missed}/{total} = {eer_percentage:.2f}%")
    print(f"Entity types found: {', '.join(results['entity_types_found'])}")
    print(f"\nEntity type distribution:")
    for ent_type, count in ref_types.items():
        print(f"  {ent_type}: {count}")
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()


#     python3 eer_stanford.py ref_transcript.json gt_transcript.json \
#   --output stanford_results.json