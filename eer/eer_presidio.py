#!/usr/bin/env python3
"""
Calculate Entity Error Rate (EER) using Microsoft Presidio
Install: pip install presidio-analyzer presidio-anonymizer
Also need: python -m spacy download en_core_web_lg
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from presidio_analyzer import AnalyzerEngine

# Initialize Presidio analyzer
analyzer = AnalyzerEngine()

# Entity type mapping (Presidio uses different names)
PRESIDIO_TO_STANDARD = {
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
    "DATE_TIME": "DATE",
    "NRP": "NORP",  # Nationalities, religious, political groups
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "CREDIT_CARD": "CREDIT_CARD",
    "CRYPTO": "CRYPTO",
    "IBAN_CODE": "IBAN",
    "IP_ADDRESS": "IP",
    "MEDICAL_LICENSE": "MED_LICENSE",
    "US_SSN": "SSN",
    "US_DRIVER_LICENSE": "DL",
    "US_PASSPORT": "PASSPORT",
    "UK_NHS": "NHS",
    "US_BANK_NUMBER": "BANK_ACCT"
}

def extract_entities_presidio(text: str, score_threshold: float = 0.7) -> List[Dict]:
    """Extract entities using Microsoft Presidio"""
    # Analyze text
    results = analyzer.analyze(text=text, language='en')
    
    entities = []
    for result in results:
        if result.score >= score_threshold:
            entities.append({
                "text": text[result.start:result.end],
                "type": PRESIDIO_TO_STANDARD.get(result.entity_type, result.entity_type),
                "start": result.start,
                "end": result.end,
                "score": result.score
            })
    
    return entities

def load_bot_utterances(filepath: Path, speaker_label: str) -> List[str]:
    """Load bot utterances from JSON transcript"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    utterances = []
    if isinstance(data, dict) and "utterances" in data:
        utterance_list = data["utterances"]
    else:
        utterance_list = data
    for entry in utterance_list:
        if entry.get('speaker') == speaker_label:
            content = entry.get('content', '').strip()
            if content:
                utterances.append(content)
    return utterances

def normalize_entity(text: str) -> str:
    """Normalize entity text for comparison"""
    return text.lower().strip()

def calculate_eer(ref_entities: List[Dict], hyp_entities: List[Dict]) -> Tuple[int, int, List[Dict]]:
    """Calculate Entity Error Rate"""
    # Group entities by type
    ref_by_type = defaultdict(list)
    hyp_by_type = defaultdict(list)
    
    for ent in ref_entities:
        ref_by_type[ent['type']].append({
            'normalized': normalize_entity(ent['text']),
            'original': ent['text'],
            'score': ent.get('score', 1.0)
        })
    
    for ent in hyp_entities:
        hyp_by_type[ent['type']].append(normalize_entity(ent['text']))
    
    missed_entities = []
    total_entities = 0
    
    # Check each reference entity
    for ent_type, ref_items in ref_by_type.items():
        hyp_texts = hyp_by_type.get(ent_type, [])
        
        for ref_item in ref_items:
            total_entities += 1
            if ref_item['normalized'] not in hyp_texts:
                missed_entities.append({
                    "text": ref_item['original'],
                    "type": ent_type,
                    "confidence": ref_item['score'],
                    "status": "missed"
                })
    
    return len(missed_entities), total_entities, missed_entities

def find_entity_errors_in_utterances(ref_utts: List[str], hyp_utts: List[str]) -> List[Dict]:
    """Find utterance pairs with entity errors"""
    error_pairs = []
    
    # Compare utterances
    for i in range(min(len(ref_utts), len(hyp_utts), 10)):
        ref_utt = ref_utts[i]
        hyp_utt = hyp_utts[i]
        
        ref_ents = extract_entities_presidio(ref_utt)
        hyp_ents = extract_entities_presidio(hyp_utt)
        
        if ref_ents:
            ref_normalized = set(normalize_entity(e['text']) for e in ref_ents)
            hyp_normalized = set(normalize_entity(e['text']) for e in hyp_ents)
            
            if not ref_normalized.issubset(hyp_normalized):
                error_pairs.append({
                    "index": i,
                    "reference": ref_utt,
                    "hypothesis": hyp_utt,
                    "ref_entities": [{"text": e['text'], "type": e['type'], "score": e['score']} 
                                   for e in ref_ents],
                    "hyp_entities": [{"text": e['text'], "type": e['type'], "score": e['score']} 
                                   for e in hyp_ents],
                    "missed": list(ref_normalized - hyp_normalized)
                })
    
    return error_pairs

def main():
    parser = argparse.ArgumentParser(description="Calculate EER using Microsoft Presidio")
    parser.add_argument("ref_transcript", help="Path to reference transcript JSON")
    parser.add_argument("hyp_transcript", help="Path to hypothesis transcript JSON")
    parser.add_argument("--score-threshold", type=float, default=0.7, 
                       help="Minimum confidence score for entities (0-1)")
    parser.add_argument("--output", default="eer_presidio_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    ref_speaker = "assistant"
    hyp_speaker = "Agent"
    
    # Load utterances
    print("Loading transcripts...")
    ref_utts = load_bot_utterances(Path(args.ref_transcript), ref_speaker)
    hyp_utts = load_bot_utterances(Path(args.hyp_transcript), hyp_speaker)
    
    print(f"Loaded {len(ref_utts)} reference utterances")
    print(f"Loaded {len(hyp_utts)} hypothesis utterances")
    
    # Concatenate all utterances
    ref_text = " ".join(ref_utts)
    hyp_text = " ".join(hyp_utts)
    
    # Extract entities
    print(f"\nExtracting entities using Presidio (threshold: {args.score_threshold})...")
    ref_entities = extract_entities_presidio(ref_text, args.score_threshold)
    hyp_entities = extract_entities_presidio(hyp_text, args.score_threshold)
    
    print(f"Found {len(ref_entities)} entities in reference")
    print(f"Found {len(hyp_entities)} entities in hypothesis")
    
    # Calculate EER
    missed, total, missed_list = calculate_eer(ref_entities, hyp_entities)
    eer_percentage = (missed / total * 100) if total > 0 else 0
    
    # Find error pairs
    print("Finding error pairs...")
    error_pairs = find_entity_errors_in_utterances(ref_utts, hyp_utts)
    
    # Entity type distribution
    ref_types = defaultdict(int)
    hyp_types = defaultdict(int)
    
    for e in ref_entities:
        ref_types[e['type']] += 1
    for e in hyp_entities:
        hyp_types[e['type']] += 1
    
    # Calculate average confidence scores
    ref_avg_score = sum(e['score'] for e in ref_entities) / len(ref_entities) if ref_entities else 0
    hyp_avg_score = sum(e['score'] for e in hyp_entities) / len(hyp_entities) if hyp_entities else 0
    
    # Create results
    results = {
        "method": "Microsoft Presidio",
        "score_threshold": args.score_threshold,
        "metrics": {
            "EER": {
                "missed_entities": missed,
                "total_entities": total,
                "percentage": round(eer_percentage, 2)
            }
        },
        "entity_summary": {
            "reference_entities": len(ref_entities),
            "hypothesis_entities": len(hyp_entities),
            "unique_ref_entities": len(set(e['text'] for e in ref_entities)),
            "unique_hyp_entities": len(set(e['text'] for e in hyp_entities)),
            "avg_confidence_ref": round(ref_avg_score, 3),
            "avg_confidence_hyp": round(hyp_avg_score, 3)
        },
        "entity_type_distribution": {
            "reference": dict(ref_types),
            "hypothesis": dict(hyp_types)
        },
        "missed_entities": missed_list[:20],
        "sample_error_pairs": error_pairs,
        "entity_types_found": list(set(e['type'] for e in ref_entities)),
        "presidio_specific": {
            "pii_focused_types": ["PERSON", "EMAIL", "PHONE", "CREDIT_CARD", "SSN"],
            "note": "Presidio is optimized for PII detection"
        }
    }
    
    # Rename keys in results to match required output
    results["entity_summary"]["gtothesis_entities"] = results["entity_summary"].pop("hypothesis_entities", 0)
    results["entity_summary"]["unique_gt_entities"] = results["entity_summary"].pop("unique_hyp_entities", 0)
    results["entity_type_distribution"]["gtothesis"] = results["entity_type_distribution"].pop("hypothesis", {})
    
    # Rename keys in sample_error_pairs
    for pair in results["sample_error_pairs"]:
        pair["gtothesis"] = pair.pop("hypothesis", "")
        pair["gt_entities"] = pair.pop("hyp_entities", [])
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n=== EER RESULTS (Microsoft Presidio) ===")
    print(f"EER: {missed}/{total} = {eer_percentage:.2f}%")
    print(f"Average confidence: Ref={ref_avg_score:.3f}, Hyp={hyp_avg_score:.3f}")
    print(f"Entity types found: {', '.join(results['entity_types_found'])}")
    print(f"\nEntity type distribution:")
    for ent_type, count in ref_types.items():
        print(f"  {ent_type}: {count}")
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()


#     python3 eer_presidio.py ref_transcript.json gt_transcript.json \
#   --score-threshold 0.5 \
#   --output presidio_results.json