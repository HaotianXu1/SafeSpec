import json
import os
import glob
from pathlib import Path

def main():
    # Define input and output paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.join(base_dir, "data_gen", "raw_output")
    output_root = os.path.join(base_dir, "output_split")
    
    print(f"Searching for JSONL files in: {input_root}")
    
    # Recursively find all .jsonl files in subdirectories
    jsonl_files = []
    for root, dirs, files in os.walk(input_root):
        # Filter out Qwen3-32B directory
        if "Qwen3-32B" in root or "Qwen3-8B" in root or "Qwen3-1.7B" in root or "DeepSeek-R1-Distill-Llama-70B" in root or "DeepSeek-R1-Distill-Qwen-1.5B" in root or "DeepSeek-R1-Distill-Llama-8B" in root:
            continue
            
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))
    
    print(f"Found {len(jsonl_files)} JSONL files (excluding Qwen3-32B).")
    
    for file_path in jsonl_files:
        try:
            # Determine relative path to maintain structure
            rel_path = os.path.relpath(file_path, input_root)
            # Create output file path
            # Example: Qwen3-4B/generated_CodeChameleon.jsonl -> Qwen3-4B
            sub_dir = os.path.dirname(rel_path)
            
            # Filename handling: generated_CodeChameleon.jsonl -> CodeChameleon_split.jsonl
            filename = os.path.basename(file_path)
            name_part = os.path.splitext(filename)[0]
            if name_part.startswith("generated_"):
                name_part = name_part.replace("generated_", "", 1)
            
            new_filename = f"{name_part}_split.jsonl"
            
            output_dir = os.path.join(output_root, sub_dir)
            output_file = os.path.join(output_dir, new_filename)
            
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Processing: {file_path} -> {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as out_f:
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    for line_idx, line in enumerate(in_f):
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON at line {line_idx+1} in {file_path}")
                            continue
                            
                        # Extract fields
                        prompt = data.get('original_prompt', '') or data.get('prompt', '')
                        full_prompt = data.get('full_prompt', '')
                        generated_text = data.get('generated_text', '')
                        
                        if not generated_text:
                            continue
                            
                        # Split by \n\n
                        steps = [s for s in generated_text.split('\n\n') if s.strip()]
                        
                        current_steps = []
                        for step in steps:
                            current_steps.append(step)
                            
                            # Construct the new data entry
                            combined_text = "\n\n".join(current_steps)
                            
                            new_entry = {
                                "prompt": prompt,
                                "full_prompt": full_prompt,
                                "response": combined_text,
                                "current_step": step,
                                "step_index": len(current_steps),
                                "source_file": os.path.basename(file_path)
                            }
                            
                            out_f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
                
    print(f"Done. Processed files are saved in {output_root}")

if __name__ == "__main__":
    main()
