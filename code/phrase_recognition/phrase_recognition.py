"""
Phrase Recognition Analysis using Patchscopes
Checks if model realizes multi-word phrases before reaching the final token.
"""
import torch
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from phrase_patchscopes import check_phrase_recognition, PhraseRecognitionConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Phrase Recognition Analysis")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-2-1124-7B",
                       help="HuggingFace model identifier")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace API token (if needed)")
    
    # Input/Output arguments
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input CSV file with phrases")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output CSV file for results")
    parser.add_argument("--name_column", type=str, default="entity",
                       help="Column name containing phrases")
    
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--max_phrases", type=int, default=None,
                       help="Maximum number of phrases to process (for testing)")
    parser.add_argument("--save_every", type=int, default=100,
                       help="Save checkpoint every N phrases")
    
    # Patchscopes arguments
    parser.add_argument("--patchscope_prompt", type=str, 
                       default="Repeat this: X",
                       help="Patchscopes prompt template (X will be replaced)")
    parser.add_argument("--replace_token", type=str, default=" X",
                       help="Token to replace with hidden states")
    parser.add_argument("--num_patchscope_tokens", type=int, default=10,
                       help="Number of tokens to generate in patchscope")
    parser.add_argument("--enable_rescaling", action="store_true",
                       help="Rescale hidden states to embedding norm")
    parser.add_argument("--amplifying_factor", type=float, default=1.0,
                       help="Amplification factor for hidden states")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="Model dtype")
    
    return parser.parse_args()


def load_phrases(input_file: str, name_column: str, max_phrases: int = None) -> list:
    """Load phrases from CSV file."""
    df = pd.read_csv(input_file)
    
    if name_column not in df.columns:
        raise ValueError(f"Column '{name_column}' not found in {input_file}")
    
    phrases = df[name_column].dropna().tolist()
    
    if max_phrases:
        phrases = phrases[:max_phrases]
    
    print(f"Loaded {len(phrases)} phrases from {input_file}")
    return phrases


def save_results(results: list, output_file: str):
    """Save results to CSV file."""
    df = pd.DataFrame(results, columns=["name", "success"])
    df.to_csv(output_file, index=False)
    print(f"Saved {len(results)} results to {output_file}")


def process_batch(model, tokenizer, phrases: list, config: PhraseRecognitionConfig) -> list:
    """Process a batch of phrases."""
    results = []
    
    for phrase in phrases:
        try:
            success = check_phrase_recognition(
                model=model,
                tokenizer=tokenizer,
                phrase=phrase,
                config=config
            )
            results.append((phrase, 1 if success else 0))
        except Exception as e:
            print(f"Error processing '{phrase}': {e}")
            results.append((phrase, 0))  # Mark as failure on error
    
    return results


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup device and dtype
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    print(f"Using device: {device}")
    print(f"Using dtype: {args.dtype}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=args.hf_token,
        torch_dtype=dtype,
        device_map="auto",  # Automatically handle multi-GPU
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully")
    
    # Create patchscopes configuration
    config = PhraseRecognitionConfig(
        patchscope_prompt=args.patchscope_prompt,
        replace_token=args.replace_token,
        num_patchscope_tokens=args.num_patchscope_tokens,
        enable_rescaling=args.enable_rescaling,
        amplifying_factor=args.amplifying_factor,
        device=device
    )
    
    # Load phrases
    phrases = load_phrases(args.input_file, args.name_column, args.max_phrases)
    
    # Check if output file exists (resume from checkpoint)
    output_path = Path(args.output_file)
    processed_phrases = set()
    all_results = []
    
    if output_path.exists():
        print(f"Found existing output file: {args.output_file}")
        existing_df = pd.read_csv(args.output_file)
        processed_phrases = set(existing_df["name"].tolist())
        all_results = existing_df.values.tolist()
        print(f"Resuming from checkpoint: {len(processed_phrases)} already processed")
    
    # Filter out already processed phrases
    remaining_phrases = [p for p in phrases if p not in processed_phrases]
    print(f"Processing {len(remaining_phrases)} remaining phrases")
    
    # Process in batches with progress bar
    total_batches = (len(remaining_phrases) + args.batch_size - 1) // args.batch_size
    
    with tqdm(total=len(remaining_phrases), desc="Processing phrases") as pbar:
        for batch_idx in range(0, len(remaining_phrases), args.batch_size):
            batch_phrases = remaining_phrases[batch_idx:batch_idx + args.batch_size]
            
            # Process batch
            batch_results = process_batch(model, tokenizer, batch_phrases, config)
            all_results.extend(batch_results)
            
            # Update progress
            pbar.update(len(batch_phrases))
            
            # Save checkpoint periodically
            if (batch_idx // args.batch_size + 1) % (args.save_every // args.batch_size) == 0:
                save_results(all_results, args.output_file)
                print(f"Checkpoint saved at {len(all_results)} phrases")
    
    # Final save
    save_results(all_results, args.output_file)
    
    # Print summary statistics
    successes = sum(1 for _, success in all_results if success == 1)
    total = len(all_results)
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"Total phrases: {total}")
    print(f"Successful: {successes} ({100*successes/total:.1f}%)")
    print(f"Failed: {total - successes} ({100*(total-successes)/total:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
