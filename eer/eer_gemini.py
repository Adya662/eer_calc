#!/usr/bin/env python3
"""
Calculate Entity Error Rate (EER) using Gemini LLM as NER
Install: pip install google-generativeai
Set environment variable: GOOGLE_API_KEY=your_api_key
"""
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import google.generativeai as genai
import time

from dotenv import load_dotenv
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

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
    prompt = NER_PROMPT.format(text=text)
    time.sleep(5)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            time.sleep(4)
            
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
            words = text.split()
            
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
                time.sleep(2)  # Wait before retry
            else:
                print(f"Failed to extract entities after {max_retries} attempts")
                return []

def load_bot_utterances(filepath: Path, speaker_label: str) -> List[str]:
    """Load bot utterances from JSON transcript"""
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

def match_entities_to_utterances(utterances: List[str], entities: List[Dict]) -> List[List[Dict]]:
    """Tag each utterance with the subset of entities it contains"""
    utt_entities = [[] for _ in utterances]
    for ent in entities:
        for i, utt in enumerate(utterances):
            if ent['text'] in utt:
                utt_entities[i].append(ent)
                break
    return utt_entities

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

def batch_process_text(text: str, chunk_size: int = 1000) -> List[Dict]:
    """Process full text without chunking"""
    print("Processing full text without chunking...")
    return extract_entities_gemini(text)

def find_entity_errors_in_utterances(
    ref_utts: List[str],
    gt_utts: List[str],
    ref_entities_per_utt: List[List[Dict]],
    gt_entities_per_utt: List[List[Dict]]
) -> List[Dict]:
    """Find utterance pairs with entity errors"""
    error_pairs = []
    for i in range(min(len(ref_utts), len(gt_utts), 5)):
        ref_utt = ref_utts[i]
        gt_utt = gt_utts[i]
        ref_ents = ref_entities_per_utt[i]
        gt_ents = gt_entities_per_utt[i]

        ref_norm = set(normalize_entity(e['text']) for e in ref_ents)
        gt_norm = set(normalize_entity(e['text']) for e in gt_ents)

        if not ref_norm.issubset(gt_norm):
            error_pairs.append({
                "index": i,
                "reference": ref_utt,
                "gtothesis": gt_utt,
                "ref_entities": [{"text": e['text'], "type": e['type']} for e in ref_ents],
                "gt_entities": [{"text": e['text'], "type": e['type']} for e in gt_ents],
                "missed": list(ref_norm - gt_norm)
            })
    return error_pairs

def main():
    parser = argparse.ArgumentParser(
        description="Calculate EER using Gemini LLM as NER"
    )
    parser.add_argument("ref_transcript", help="Path to reference transcript JSON")
    parser.add_argument("gt_transcript", help="Path to gtothesis transcript JSON")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--output", default="eer_gemini_results.json", help="Output JSON file")

    args = parser.parse_args()
    # hard-code speaker labels
    ref_speaker = "assistant"
    gt_speaker = "Agent"

    # Load utterances
    print("Loading transcripts...")
    ref_utts = load_bot_utterances(Path(args.ref_transcript), ref_speaker)
    gt_utts = load_bot_utterances(Path(args.gt_transcript), gt_speaker)
    
    print(f"Loaded {len(ref_utts)} reference utterances")
    print(f"Loaded {len(gt_utts)} gtothesis utterances")
    
    # Concatenate all utterances
    ref_text = " ".join(ref_utts)
    gt_text = " ".join(gt_utts)
    
    # Extract entities
    print(f"\nExtracting entities using Gemini LLM (chunk size: {args.chunk_size})...")
    print("Processing reference text...")
    ref_entities = batch_process_text(ref_text, args.chunk_size)
    
    print("Processing gtothesis text...")
    gt_entities = batch_process_text(gt_text, args.chunk_size)
    
    print(f"\nFound {len(ref_entities)} entities in reference")
    print(f"Found {len(gt_entities)} entities in gtothesis")
    
    # Calculate EER
    missed, total, missed_list = calculate_eer(ref_entities, gt_entities)
    eer_percentage = (missed / total * 100) if total > 0 else 0

    # Tag entities per utterance
    ref_entities_per_utt = match_entities_to_utterances(ref_utts, ref_entities)
    gt_entities_per_utt = match_entities_to_utterances(gt_utts, gt_entities)

    # Find error pairs (limited due to API costs)
    print("\nFinding error pairs (limited to 5)...")
    error_pairs = find_entity_errors_in_utterances(
        ref_utts[:5],
        gt_utts[:5],
        ref_entities_per_utt[:5],
        gt_entities_per_utt[:5]
    )
    
    # Entity type distribution
    ref_types = defaultdict(int)
    gt_types = defaultdict(int)
    
    for e in ref_entities:
        ref_types[e['type']] += 1
    for e in gt_entities:
        gt_types[e['type']] += 1
    
    # Create results
    results = {
        "method": "Gemini LLM as NER",
        "model": "gemini-1.5-pro",
        "chunk_size": args.chunk_size,
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
            "unique_ref_entities": len(set(normalize_entity(e['text']) for e in ref_entities)),
            "unique_gt_entities": len(set(normalize_entity(e['text']) for e in gt_entities))
        },
        "entity_type_distribution": {
            "reference": dict(ref_types),
            "gtothesis": dict(gt_types)
        },
        "missed_entities": missed_list[:20],
        "sample_error_pairs": error_pairs,
        "entity_types_found": list(set(e['type'] for e in ref_entities)),
        "llm_specific": {
            "advantages": [
                "Context-aware entity recognition",
                "Can identify complex and domain-specific entities",
                "Flexible entity types"
            ],
            "limitations": [
                "API costs",
                "Rate limits",
                "Processing time",
                "Token limits for long texts"
            ]
        }
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n=== EER RESULTS (Gemini LLM) ===")
    print(f"EER: {missed}/{total} = {eer_percentage:.2f}%")
    print(f"Unique entities in reference: {results['entity_summary']['unique_ref_entities']}")
    print(f"Unique entities in gtothesis: {results['entity_summary']['unique_gt_entities']}")
    print(f"Entity types found: {', '.join(results['entity_types_found'])}")
    print(f"\nEntity type distribution:")
    for ent_type, count in ref_types.items():
        print(f"  {ent_type}: {count}")
    print(f"\nResults saved to: {args.output}")
    print("\nNote: Limited error pairs due to API costs")

if __name__ == "__main__":
    main()