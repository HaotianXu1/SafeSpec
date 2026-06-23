import os
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
import glob
DEFAULT_SYSTEM_PROMPT = ("You are a Qwen3 reasoning model. \n"
                         "Always reason step by step inside <think>...</think>. \n"
                         "In your <think> process, reflect whether the user’s request or your reasoning is harmful, illegal, or unsafe. \n"
                         "If harmful, stop pursuing it and output a polite refusal instead of harmful content.")
def parse_args():
    parser = argparse.ArgumentParser(description="Generate answers using vLLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local model.")
    parser.add_argument("--input_paths", type=str, nargs='+', default=["/data/xuhaotian/Safety_head/prompt/jailbreak_prompt_train"], help="List of input file or directory paths.")
    parser.add_argument("--output_dir", type=str, default="/data/xuhaotian/Safety_head/data_gen/raw_output", help="Directory to save results.")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to use.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for generation.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per prompt.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization.")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--gpus", type=str, default='6,7', help="IDs of GPUs to use, e.g. '0' or '0,1'.")

    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # Extract model name and update output directory
    model_name = os.path.basename(os.path.normpath(args.model_path))
    args.output_dir = os.path.join(args.output_dir, model_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize vLLM
    print(f"Loading model from {args.model_path}...")
    llm = LLM(model=args.model_path, 
              tensor_parallel_size=args.tensor_parallel_size, 
              gpu_memory_utilization=args.gpu_memory_utilization,
              trust_remote_code=True)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        n=args.num_samples,
        temperature=args.temperature if args.num_samples > 1 else 0.0 # Use greedy if n=1 unless specified otherwise? 
        # Actually user might want diversity even with n=1 if they run it multiple times, but n>1 implies sampling.
        # Let's trust the args.temperature for n>1, and maybe default to 0 for n=1 if user didn't specify?
        # To be safe, let's just use the provided temperature.
    )
    # Re-setting sampling params to use user provided args directly is cleaner
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        n=args.num_samples,
        temperature=args.temperature
    )

    # Process files
    all_files = []
    for path in args.input_paths:
        if os.path.isfile(path):
            all_files.append(path)
        elif os.path.isdir(path):
            all_files.extend(glob.glob(os.path.join(path, "*.csv")))
            all_files.extend(glob.glob(os.path.join(path, "*.parquet")))
        else:
            print(f"Path {path} does not exist.")

    if not all_files:
        print(f"No input files found in {args.input_paths}")
        return

    for file_path in all_files:
        print(f"Processing {file_path}...")
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                print(f"Unsupported file format: {file_path}")
                continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        # Determine prompt column
        prompt_col = None
        for col in ['jailbreak_prompt', 'prompt']:
            if col in df.columns:
                prompt_col = col
                break
        
        if not prompt_col:
            print(f"Column 'jailbreak_prompt' or 'prompt' not found in {file_path}, skipping. Available columns: {df.columns.tolist()}")
            continue

        prompts = df[prompt_col].tolist()
        
        # Prepare inputs with system prompt
        final_prompts = []
        for p in prompts:
            # Construct prompt with system prompt if available
            current_prompt = ""
            if args.system_prompt:
                current_prompt += f"{args.system_prompt}\n\n"
            current_prompt += f"{p}\n\n<think>"
            final_prompts.append(current_prompt)
            
        outputs = llm.generate(final_prompts, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            original_prompt_text = prompts[i]
            full_prompt_text = final_prompts[i]
            # Handle multiple samples
            for sample_idx, sample in enumerate(output.outputs):
                results.append({
                    "original_prompt": original_prompt_text,
                    "full_prompt": full_prompt_text,
                    "generated_text": sample.text,
                    "sample_index": sample_idx,
                    "finish_reason": sample.finish_reason
                })
        
        output_df = pd.DataFrame(results)
        base_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(args.output_dir, f"generated_{file_name_without_ext}.jsonl")
        output_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()

