#!/usr/bin/env python3
"""
Script to run EER calculation for calls one by one, asking for user input after each call
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import from eer_gpt_lib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eer_gpt_lib import process_single_call

def main():
    calls_dir = Path("calls")
    if not calls_dir.exists() or not calls_dir.is_dir():
        print(f"Error: {calls_dir} is not a valid directory")
        sys.exit(1)
    
    # Get all valid call directories
    call_dirs = sorted([d for d in calls_dir.glob("*") if d.is_dir() 
                       and (d / "ref_transcript.json").exists() 
                       and (d / "gt_transcript.json").exists()])
    
    print(f"Found {len(call_dirs)} valid call directories to process")
    
    if not call_dirs:
        print("No valid call directories found!")
        return
    
    # Process calls one by one
    for i, call_dir in enumerate(call_dirs):
        print(f"\n{'='*80}")
        print(f"Processing call {i+1}/{len(call_dirs)}: {call_dir.name}")
        print(f"{'='*80}")
        
        try:
            result = process_single_call(call_dir)
            
            # Print results
            all_eer = result.get("all_entities_eer", {})
            con_eer = result.get("concerned_entities_eer", {})
            
            print(f"\n✅ Completed processing call: {call_dir.name}")
            print(f"📊 All Entities EER: {all_eer.get('eer_percentage', 0):.2f}% | Concerned Entities EER: {con_eer.get('eer_percentage', 0):.2f}%")
            
            # Ask user if they want to continue to the next call
            if i < len(call_dirs) - 1:
                next_call = call_dirs[i + 1].name
                user_input = input(f"\n🤔 Continue to next call ({next_call})? (y/n/q to quit): ")
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print(f"\n⏹️  Stopping after processing {i+1} calls. Total calls available: {len(call_dirs)}")
                    break
                elif user_input.lower() not in ['y', 'yes']:
                    print(f"\n⏹️  Stopping after processing {i+1} calls. Total calls available: {len(call_dirs)}")
                    break
            else:
                print(f"\n🎉 All {len(call_dirs)} calls have been processed!")
                
        except Exception as e:
            print(f"❌ Error processing {call_dir.name}: {e}")
            
            # Ask user if they want to continue despite the error
            if i < len(call_dirs) - 1:
                next_call = call_dirs[i + 1].name
                user_input = input(f"\n🤔 Continue to next call ({next_call}) despite error? (y/n/q to quit): ")
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print(f"\n⏹️  Stopping after processing {i+1} calls. Total calls available: {len(call_dirs)}")
                    break
                elif user_input.lower() not in ['y', 'yes']:
                    print(f"\n⏹️  Stopping after processing {i+1} calls. Total calls available: {len(call_dirs)}")
                    break
            else:
                print(f"\n🎉 All {len(call_dirs)} calls have been processed!")

if __name__ == "__main__":
    main() 