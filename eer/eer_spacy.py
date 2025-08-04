#!/usr/bin/env python3
"""
Calculate Entity Error Rate (EER) using SpaCy NER
"""
import json
import argparse
from pathlib import Path
import spacy
import stanza
from typing import List, Dict, Tuple
from collections import defaultdict

# Load SpaCy model (install with: python -m spacy download en_core_web_sm)
# nlp = spacy.load("en_core_web_sm")
# Load spaCy for English
nlp_en = spacy.load("en_core_web_sm")   # English

# Load Stanza for Hindi
stanza_hi = stanza.Pipeline(lang='hi', processors='tokenize,ner', use_gpu=False)

def extract_entities(text: str) -> list:
    """Extract entities from both English (spaCy) and Hindi (Stanza)."""
    entities = []

    # English NER
    doc = nlp_en(text)
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "type": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "script": "latin"
        })
    
    # Hindi NER
    # NOTE: This is most useful for Devanagari-script text
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
    return entities


def load_bot_utterances(filepath: Path, speaker_label: str) -> List[str]:
    """Load bot utterances from JSON transcript"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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

def calculate_eer(ref_entities: List[Dict], gt_entities: List[Dict], threshold: float = 0.9) -> Tuple[int, int, List[Dict]]:
    """Calculate Entity Error Rate"""
    # Group entities by type for better comparison
    ref_by_type = defaultdict(list)
    gt_by_type = defaultdict(list)
    
    for ent in ref_entities:
        ref_by_type[ent['type']].append(normalize_entity(ent['text']))
    
    for ent in gt_entities:
        gt_by_type[ent['type']].append(normalize_entity(ent['text']))
    
    missed_entities = []
    total_entities = 0
    
    # Check each reference entity
    for ent_type, ref_texts in ref_by_type.items():
        gt_texts = gt_by_type.get(ent_type, [])
        
        for ref_text in ref_texts:
            total_entities += 1
            if ref_text not in gt_texts:
                # Entity missed
                missed_entities.append({
                    "text": ref_text,
                    "type": ent_type,
                    "status": "missed"
                })
    
    return len(missed_entities), total_entities, missed_entities

def find_entity_errors_in_utterances(ref_utts: List[str], gt_utts: List[str]) -> List[Dict]:
    """Find utterance pairs with entity errors"""
    error_pairs = []
    
    # For alignment issues, we'll compare concatenated text
    # But still try to find which utterances have the most errors
    
    for i, ref_utt in enumerate(ref_utts[:min(len(ref_utts), len(gt_utts))]):
        if i < len(gt_utts):
            gt_utt = gt_utts[i]
            
            ref_ents = extract_entities(ref_utt)
            gt_ents = extract_entities(gt_utt)
            
            if ref_ents and not all(any(normalize_entity(r['text']) == normalize_entity(h['text']) 
                                      for h in gt_ents) for r in ref_ents):
                error_pairs.append({
                    "index": i,
                    "reference": ref_utt,
                    "gtothesis": gt_utt,
                    "ref_entities": [{"text": e['text'], "type": e['type']} for e in ref_ents],
                    "gt_entities": [{"text": e['text'], "type": e['type']} for e in gt_ents]
                })
    
    return error_pairs[:10]  # Return first 10 error pairs

def main():
    parser = argparse.ArgumentParser(description="Calculate EER using SpaCy NER")
    parser.add_argument("ref_transcript", help="Path to reference transcript JSON")
    parser.add_argument("gt_transcript", help="Path to gtothesis transcript JSON")
    parser.add_argument("--output", default="eer_spacy_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    ref_speaker = "assistant"
    gt_speaker = "Agent"
    
    # Load utterances
    print("Loading transcripts...")
    ref_utts = load_bot_utterances(Path(args.ref_transcript), ref_speaker)
    gt_utts = load_bot_utterances(Path(args.gt_transcript), gt_speaker)
    
    print(f"Loaded {len(ref_utts)} reference utterances")
    print(f"Loaded {len(gt_utts)} gtothesis utterances")
    
    # Concatenate all utterances (handles alignment issues)
    ref_text = " ".join(ref_utts)
    gt_text = " ".join(gt_utts)
    
    # Extract entities
    print("\nExtracting entities using SpaCy...")
    ref_entities = extract_entities(ref_text)
    gt_entities = extract_entities(gt_text)

    from collections import Counter
    ref_types = Counter(e['type'] for e in ref_entities)
    gt_types  = Counter(e['type'] for e in gt_entities)
    
    print(f"Found {len(ref_entities)} entities in reference")
    print(f"Found {len(gt_entities)} entities in gtothesis")
    
    # Calculate EER
    missed, total, missed_list = calculate_eer(ref_entities, gt_entities)
    eer_percentage = (missed / total * 100) if total > 0 else 0
    
    # Find error pairs
    error_pairs = find_entity_errors_in_utterances(ref_utts, gt_utts)
    
    # Create results
    results = {
        "method": "SpaCy NER",
        "model": "en_core_web_sm",
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
        "missed_entities": missed_list[:20],  # First 20 missed entities
        "sample_error_pairs": error_pairs,
        "entity_types_found": list(set(e['type'] for e in ref_entities))
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n=== EER RESULTS (SpaCy) ===")
    print(f"EER: {missed}/{total} = {eer_percentage:.2f}%")
    print(f"Entity types found: {', '.join(results['entity_types_found'])}")
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()