
import json
import os

# Target order from biomedclip/zero_shot_classification.py
TARGET_ORDER = [
    'Mountain', 
    'Fire Earth', 
    'Earth', 
    'Wind', 
    'Ground', 
    'Dream', 
    'Fire', 
    'Fire Light', 
    'Water', 
    'Electricity', 
    'Drill', 
    'Light', 
    'Electricity with Wind', 
    'Rock', 
    'Stone', 
    'Fire Wood', 
    'Wood', 
    'Gold'
]

# Source order inferred from Gemini_prompts.json / chat_gpt_prompts.json keys (0-18)
# Based on my analysis of the file content
SOURCE_ORDER_MAP = {
    'Stone': "0",
    'Gold': "1",
    'Dream': "2",
    'Electricity': "3",
    'Wind': "4",
    'Electricity with Wind': "5",
    'Drill': "6",
    'Light': "7",
    'Water': "8",
    'Fire': "9",
    'Wood': "10",
    'Earth': "11",
    'Ground': "12",
    'Mountain': "13",
    'Rock': "14",
    'Fire Light': "15",
    'Fire Wood': "16",
    'Fire Earth': "17",
    'Fire Drill': "18" 
}

FILES_TO_FIX = [
    'prompts/chat_gpt_prompts.json',
    'prompts/Gemini_prompts.json',
    'tda/prompts/prompts_simple.json',
    'tda/prompts/prompts_cupl.json'
]

def reorder_file(filepath):
    print(f"Processing {filepath}...")
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

    if 'finger' not in data:
        print(f"  No 'finger' key in {filepath}")
        return

    original_finger = data['finger']
    new_finger = {}

    for i, target_class in enumerate(TARGET_ORDER):
        # find the source key for this class name
        if target_class in SOURCE_ORDER_MAP:
            src_key = SOURCE_ORDER_MAP[target_class]
            if src_key in original_finger:
                new_finger[str(i)] = original_finger[src_key]
            else:
                print(f"  Warning: Source key {src_key} ({target_class}) not found in {filepath} finger dict")
        else:
            print(f"  Error: {target_class} not found in known source map")

    # Check if we should keep Fire Drill (index 18)
    # The dataset only has 18 classes (0-17). So we probably drop it.
    # But just in case, if the file HAD 18, we warn.
    if "18" in original_finger:
        print("  Dropping 'Fire Drill' (key 18) as it is not in the target 18 classes.")

    data['finger'] = new_finger

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Updated {filepath}")

def main():
    root_dir = '/Users/liruirui/Documents/code/study/FPaCo'
    for relative_path in FILES_TO_FIX:
        full_path = os.path.join(root_dir, relative_path)
        reorder_file(full_path)

if __name__ == '__main__':
    main()
