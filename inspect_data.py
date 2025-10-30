"""
Inspection utilities for demonstration data - IMPORTABLE VERSION
Can be imported by other scripts or run standalone
"""

import numpy as np


def inspect_demonstrations_detailed(demos, tokenizer, num_samples=10):
    """
    Detailed inspection with pretty formatting.
    
    Args:
        demos: List of demonstration dicts with 'text' field
        tokenizer: Tokenizer to use for token counting
        num_samples: Number of detailed samples to show
    """
    
    print("\n" + "🔥"*40)
    print(" "*20 + "DEMONSTRATION INSPECTOR")
    print("🔥"*40)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("📊 DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"   Total demonstrations: {len(demos):,}")
    
    # Text lengths
    text_lengths = [len(d['text']) for d in demos]
    print(f"\n   Character Counts:")
    print(f"      Minimum:   {min(text_lengths):,}")
    print(f"      Maximum:   {max(text_lengths):,}")
    print(f"      Average:   {np.mean(text_lengths):,.0f}")
    print(f"      Median:    {int(np.median(text_lengths)):,}")
    print(f"      Std Dev:   {np.std(text_lengths):,.0f}")
    
    # Token lengths (sample for speed)
    print(f"\n   Tokenizing {min(100, len(demos))} samples...")
    token_lengths = []
    for demo in demos[:100]:
        tokens = tokenizer(demo['text'], return_tensors="pt")
        token_lengths.append(len(tokens['input_ids'][0]))
    
    print(f"   Token Counts (sampled):")
    print(f"      Minimum:   {min(token_lengths)}")
    print(f"      Maximum:   {max(token_lengths)}")
    print(f"      Average:   {np.mean(token_lengths):.0f}")
    print(f"      Median:    {int(np.median(token_lengths))}")
    
    # Check for truncation
    too_long = sum(1 for l in token_lengths if l > 2048)
    if too_long > 0:
        print(f"\n   ⚠️  WARNING: {too_long} samples exceed 2048 tokens (will be truncated)")
    
    # Action distribution
    print(f"\n{'='*80}")
    print("🎯 ACTION DISTRIBUTION")
    print(f"{'='*80}")
    
    action_counts = {"water": 0, "break": 0, "wait": 0, "unknown": 0}
    
    for demo in demos:
        text = demo['text'].lower()
        if "water " in text and any(c.isdigit() for c in text):
            action_counts["water"] += 1
        elif "break " in text and any(c.isdigit() for c in text):
            action_counts["break"] += 1
        elif "wait" in text:
            action_counts["wait"] += 1
        else:
            action_counts["unknown"] += 1
    
    total = sum(action_counts.values())
    print(f"\n   Total actions: {total:,}\n")
    
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total) * 100 if total > 0 else 0
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"   {action:8s}: {count:5,} ({percentage:5.1f}%) │{bar}│")
    
    # Detailed sample inspection
    print(f"\n{'='*80}")
    print(f"📝 DETAILED SAMPLE INSPECTION (showing {min(num_samples, len(demos))} samples)")
    print(f"{'='*80}")
    
    for i in range(min(num_samples, len(demos))):
        print(f"\n{'┏' + '━'*78 + '┓'}")
        print(f"┃ SAMPLE #{i+1:<70} ┃")
        print(f"{'┗' + '━'*78 + '┛'}")
        
        text = demos[i]['text']
        
        # Parse sections
        sections = {}
        if "<|im_start|>system" in text:
            start = text.find("<|im_start|>system") + len("<|im_start|>system\n")
            end = text.find("<|im_end|>", start)
            sections['system'] = text[start:end].strip()
        
        if "<|im_start|>user" in text:
            start = text.find("<|im_start|>user") + len("<|im_start|>user\n")
            end = text.find("<|im_end|>", start)
            sections['user'] = text[start:end].strip()
        
        if "<|im_start|>assistant" in text:
            start = text.find("<|im_start|>assistant") + len("<|im_start|>assistant\n")
            end = text.find("<|im_end|>", start)
            sections['assistant'] = text[start:end].strip()
        
        # Display sections (truncate long lines)
        print("\n┌─ 🤖 System Prompt ─────────────────────────────────────────────┐")
        if 'system' in sections:
            for line in sections['system'].split('\n'):
                print(f"│ {line[:62]:<62} │")
        print("└────────────────────────────────────────────────────────────────┘")
        
        print("\n┌─ 👤 User Input (Observation) ──────────────────────────────────┐")
        if 'user' in sections:
            for line in sections['user'].split('\n'):
                print(f"│ {line[:62]:<62} │")
        print("└────────────────────────────────────────────────────────────────┘")
        
        print("\n┌─ 🎯 Assistant Response (Expert Action) ────────────────────────┐")
        if 'assistant' in sections:
            action_line = sections['assistant']
            print(f"│ {action_line:<62} │")
            print(f"│ {'':62} │")
            print(f"│ This is what the model will learn to predict! {'':15} │")
        print("└────────────────────────────────────────────────────────────────┘")
        
        # Tokenization info
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens['input_ids'][0]
        
        print(f"\n📊 Statistics:")
        print(f"   Characters: {len(text):,}")
        print(f"   Tokens:     {len(input_ids):,}")
        print(f"   Fits in context: {'✅ Yes' if len(input_ids) <= 2048 else '❌ No (will truncate)'}")
        
        # Show token IDs
        print(f"\n🔢 Token IDs:")
        print(f"   First 15: {input_ids[:15].tolist()}")
        print(f"   Last 10:  {input_ids[-10:].tolist()}")
        
        # Highlight what gets trained
        print(f"\n💡 Training Focus:")
        if 'assistant' in sections:
            response_tokens = tokenizer(sections['assistant'], return_tensors="pt")
            num_response_tokens = len(response_tokens['input_ids'][0])
            num_prompt_tokens = len(input_ids) - num_response_tokens
            
            print(f"   Prompt tokens (masked):    {num_prompt_tokens} ❄️  (not trained)")
            print(f"   Response tokens (trained): {num_response_tokens} 🔥 (model learns this)")
            print(f"   Training efficiency:       {(num_response_tokens/len(input_ids)*100):.1f}%")
        
        print("\n" + "─"*80)
    
    print("\n" + "="*80)
    return True


# ============================================================================
# STANDALONE SCRIPT MODE
# ============================================================================

def save_samples_to_file(demos, output_file="demo_samples.txt", num_samples=20):
    """Save demonstrations to a text file for inspection."""
    print(f"\n💾 Saving {num_samples} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DEMONSTRATION SAMPLES FOR WILDFIRE SFT TRAINING\n")
        f.write("="*80 + "\n\n")
        
        for i in range(min(num_samples, len(demos))):
            f.write(f"\n{'='*80}\n")
            f.write(f"SAMPLE #{i+1}\n")
            f.write(f"{'='*80}\n\n")
            f.write(demos[i]['text'])
            f.write("\n\n")
    
    print(f"✅ Saved to {output_file}")


def main():
    """Main function for standalone execution."""
    import sys
    import argparse
    from unsloth import FastLanguageModel
    
    # Import environment dependencies
    sys.path.append("/workspace/OpenEnv/src")
    from envs.wildfire_env import WildfireEnv, WildfireAction
    
    # Import from Supervised.py
    sys.path.append("/workspace/trainingfiles")
    from Supervised import (
        WildfireExpert,
        collect_expert_demonstrations,
    )
    
    parser = argparse.ArgumentParser(description="Inspect demonstration data")
    parser.add_argument("--url", default="http://localhost:8010")
    parser.add_argument("--demos", type=int, default=50,
                       help="Number of episodes to collect")
    parser.add_argument("--samples", type=int, default=10,
                       help="Number of samples to display in detail")
    parser.add_argument("--save", type=str, default=None,
                       help="Save samples to file (e.g., --save demos.txt)")
    parser.add_argument("--save-count", type=int, default=20,
                       help="Number of samples to save to file")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔍 DEMONSTRATION DATA INSPECTOR")
    print("="*80)
    
    # Connect to environment
    print(f"\n📡 Connecting to wildfire environment...")
    env = WildfireEnv(args.url)
    obs = env.reset()
    width, height = obs.observation.width, obs.observation.height
    print(f"✅ Environment ready: {width}x{height} grid")
    
    # Collect demonstrations
    expert = WildfireExpert(width, height)
    demos, rewards = collect_expert_demonstrations(
        env, expert, num_episodes=args.demos, max_steps=128
    )
    
    print(f"\n✅ Collected {len(demos)} demonstrations")
    print(f"   Expert mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    # Load tokenizer
    print(f"\n🤖 Loading tokenizer...")
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Inspect
    inspect_demonstrations_detailed(demos, tokenizer, num_samples=args.samples)
    
    # Save to file if requested
    if args.save:
        save_samples_to_file(demos, args.save, num_samples=args.save_count)
    
    print("\n" + "="*80)
    print("✅ INSPECTION COMPLETE!")
    print("="*80)
    print("\n💡 Tip: Review the samples above to ensure:")
    print("   1. Observations contain relevant fire information")
    print("   2. Actions are appropriate for the situation")
    print("   3. Format is consistent across samples")
    print("   4. Token counts fit within 2048 limit")
    print("\n")


if __name__ == "__main__":
    main()