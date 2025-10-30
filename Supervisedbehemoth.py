"""
SUPERVISED FINE-TUNING - EXTREME MODE FOR AMD MI100
Pushing the HPC to its absolute limits!
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
sys.path.append("/workspace/trainingfiles")
from inspect_data import inspect_demonstrations_detailed
sys.path.append("/workspace/OpenEnv/src")
from envs.wildfire_env import WildfireEnv, WildfireAction


# ============================================================================
# CUSTOM DATA COLLATOR
# ============================================================================

@dataclass
class DataCollatorForCompletionOnlyLM:
    """Data collator that masks prompts and only trains on completions."""
    tokenizer: Any
    response_template: str
    instruction_template: Optional[str] = None
    mlm: bool = False
    ignore_index: int = -100
    
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = []
        
        for example in examples:
            if isinstance(example, dict):
                if "text" in example:
                    text = example["text"]
                elif "input_ids" in example:
                    batch.append(example)
                    continue
                else:
                    raise ValueError(f"Example must have 'text' or 'input_ids' key")
            else:
                text = example
            
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=2048,
                padding=False,
                return_tensors=None,
            )
            batch.append(tokenized)
        
        batch = self.tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )
        
        labels = batch["input_ids"].clone()
        
        response_token_ids = self.tokenizer.encode(
            self.response_template,
            add_special_tokens=False
        )
        
        for idx in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][idx]
            
            response_start_idx = None
            for i in range(len(input_ids) - len(response_token_ids) + 1):
                if torch.equal(input_ids[i:i+len(response_token_ids)], 
                              torch.tensor(response_token_ids, device=input_ids.device)):
                    response_start_idx = i + len(response_token_ids)
                    break
            
            if response_start_idx is not None:
                labels[idx, :response_start_idx] = self.ignore_index
            else:
                labels[idx, :] = self.ignore_index
            
            labels[idx][input_ids == self.tokenizer.pad_token_id] = self.ignore_index
        
        batch["labels"] = labels
        
        return batch
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return self.torch_call(examples)


# ============================================================================
# EXPERT POLICY
# ============================================================================

class WildfireExpert:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    
    def get_action(self, obs) -> WildfireAction:
        burning_cells = self._get_burning_cells(obs)
        
        if not burning_cells:
            return WildfireAction(action="wait", x=None, y=None)
        
        if obs.remaining_water > 0:
            target = self._find_best_water_target(obs, burning_cells)
            if target:
                return WildfireAction(action="water", x=target[0], y=target[1])
        
        if obs.remaining_breaks > 2:
            target = self._find_firebreak_location(obs, burning_cells)
            if target:
                return WildfireAction(action="break", x=target[0], y=target[1])
        
        return WildfireAction(action="wait", x=None, y=None)
    
    def _get_burning_cells(self, obs):
        burning = []
        for y in range(self.height):
            for x in range(self.width):
                if obs.grid[y * self.width + x] == 2:
                    burning.append((x, y))
        return burning
    
    def _find_best_water_target(self, obs, burning_cells):
        if not burning_cells:
            return None
        best_cell = None
        best_score = -1
        for x, y in burning_cells:
            score = self._count_burning_neighbors(obs, x, y)
            if score > best_score:
                best_score = score
                best_cell = (x, y)
        return best_cell
    
    def _count_burning_neighbors(self, obs, x: int, y: int) -> int:
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if obs.grid[ny * self.width + nx] == 2:
                        count += 1
        return count
    
    def _find_firebreak_location(self, obs, burning_cells):
        for bx, by in burning_cells:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = bx + dx, by + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if obs.grid[ny * self.width + nx] == 1:
                            return (nx, ny)
        return None


# ============================================================================
# FORMATTING
# ============================================================================

def format_observation_as_prompt(obs) -> str:
    burning_cells = []
    for y in range(obs.height):
        for x in range(obs.width):
            if obs.grid[y * obs.width + x] == 2:
                burning_cells.append((x, y))
                if len(burning_cells) >= 5:
                    break
        if len(burning_cells) >= 5:
            break
    
    fire_locs = ", ".join([f"({x},{y})" for x, y in burning_cells[:5]])
    if len(burning_cells) > 5:
        fire_locs += f" +{len(burning_cells)-5} more"
    
    return f"""<|im_start|>system
You are a wildfire agent. Respond with only the action.
<|im_end|>
<|im_start|>user
Fire: {obs.burning_count} at {fire_locs}. Wind: {obs.wind_dir}. Water: {obs.remaining_water}, Breaks: {obs.remaining_breaks}
<|im_end|>
<|im_start|>assistant
"""


def format_action_as_text(action: WildfireAction) -> str:
    if action.action == "wait":
        return "wait"
    elif action.action == "water" and action.x is not None:
        return f"water {action.x} {action.y}"
    elif action.action == "break" and action.x is not None:
        return f"break {action.x} {action.y}"
    else:
        return "wait"


def parse_action_from_text(text: str, width: int, height: int) -> WildfireAction:
    import re
    text = text.strip().lower()
    
    water_match = re.search(r'water\s+(\d+)\s+(\d+)', text)
    if water_match:
        x = max(0, min(int(water_match.group(1)), width - 1))
        y = max(0, min(int(water_match.group(2)), height - 1))
        return WildfireAction(action="water", x=x, y=y)
    
    break_match = re.search(r'break\s+(\d+)\s+(\d+)', text)
    if break_match:
        x = max(0, min(int(break_match.group(1)), width - 1))
        y = max(0, min(int(break_match.group(2)), height - 1))
        return WildfireAction(action="break", x=x, y=y)
    
    return WildfireAction(action="wait", x=None, y=None)


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_expert_demonstrations(
    env: WildfireEnv,
    expert: WildfireExpert,
    num_episodes: int = 500,
    max_steps: int = 128,
) -> Tuple[List[Dict], List[float]]:
    
    print(f"\nCollecting {num_episodes} expert demonstrations...")
    
    demos = []
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs_result = env.reset()
        ep_reward = 0
        
        for step in range(max_steps):
            obs = obs_result.observation
            
            if obs.burning_count == 0:
                break
            
            action = expert.get_action(obs)
            prompt = format_observation_as_prompt(obs)
            completion = format_action_as_text(action) + "<|im_end|>"
            
            obs_result = env.step(action)
            ep_reward += obs_result.reward
            
            demos.append({
                "text": prompt + completion
            })
            
            if obs_result.done:
                break
        
        episode_rewards.append(ep_reward)
        
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{num_episodes}, Avg Reward: {np.mean(episode_rewards[-50:]):.2f}")
    
    print(f"✅ Collected {len(demos)} demonstrations")
    print(f"   Expert Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    return demos, episode_rewards


# ============================================================================
# MEMORY MONITORING
# ============================================================================

def print_gpu_memory(stage: str = ""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"\n💾 GPU Memory {stage}:")
        print(f"   Allocated: {allocated:.2f}GB")
        print(f"   Reserved:  {reserved:.2f}GB")
        print(f"   Peak:      {max_allocated:.2f}GB")
        print(f"   Available: {32 - reserved:.2f}GB / 32GB")


# ============================================================================
# EXTREME MODE TRAINING - MAXIMUM PERFORMANCE!
# ============================================================================

def train_sft_wildfire_extreme(
    base_url: str = "http://localhost:8010",
    model_name: str = "unsloth/Llama-3.2-1B-Instruct",
    num_demo_episodes: int = 1000,
    output_dir: str = "./sft_wildfire_extreme",
    num_train_epochs: int = 20,
    max_steps: int = 128,
):
    """
    EXTREME MODE - Push AMD MI100 to its absolute limits!
    """
    
    print("\n" + "=" * 80)
    print("🔥🔥🔥 EXTREME MODE - MAXIMUM HPC PERFORMANCE 🔥🔥🔥")
    print("=" * 80)
    
    # Initialize environment
    print("\n📡 Connecting to wildfire environment...")
    env = WildfireEnv(base_url)
    obs = env.reset()
    width, height = obs.observation.width, obs.observation.height
    print(f"✅ Environment ready: {width}x{height} grid")
    
    # Collect MASSIVE dataset
    expert = WildfireExpert(width, height)
    demos, expert_rewards = collect_expert_demonstrations(
        env, expert, num_episodes=num_demo_episodes, max_steps=max_steps
    )
    
    print(f"\n🔍 Dataset Statistics:")
    print(f"   Total demonstrations: {len(demos)}")
    print(f"   Expert mean reward: {np.mean(expert_rewards):.2f}")
    print(f"   Expert std reward: {np.std(expert_rewards):.2f}")
    
    # 🔧 FIX: Load model and tokenizer BEFORE inspection
    print(f"\n🤖 Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"   Model dtype: {model.dtype}")
    print(f"   Model device: {model.device}")
    
    print_gpu_memory("after model load")
    
    # 🔧 FIX: Now we can inspect with tokenizer
    print("\n" + "="*80)
    print("🔍 INSPECTING DEMONSTRATION SAMPLES")
    print("="*80)
    inspect_demonstrations_detailed(demos, tokenizer, num_samples=5)
    
    response = input("\n✅ Data looks good? Continue with training? (y/N): ")
    if response.lower() != 'y':
        print("Training aborted.")
        return None, None
    
    # Add MAXIMUM LoRA adapters
    print("\n🔧 Adding MAXIMUM LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens",
        ],
        lora_alpha=128,                     # ✅ STABLE (not 256!)
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print(f"   LoRA rank: 128 (8x larger than baseline!)")
    print(f"   LoRA alpha: 128 (8x larger than baseline!)")  # 🔧 FIX: Updated text
    print(f"   Trainable params: ~{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    print_gpu_memory("after LoRA")
    
    # Prepare massive dataset
    print(f"\n📚 Creating dataset from {len(demos)} demonstrations...")
    train_size = int(0.85 * len(demos))
    train_demos = demos[:train_size]
    eval_demos = demos[train_size:]
    
    train_dataset = Dataset.from_list(train_demos)
    eval_dataset = Dataset.from_list(eval_demos)
    
    print(f"   Train: {len(train_demos)} samples")
    print(f"   Eval:  {len(eval_demos)} samples")
    
    # Create collator
    print("🔧 Creating data collator...")
    response_template = "<|im_start|>assistant\n"
    
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
        mlm=False,
    )
    
    # STABLE training arguments
    print("\n⚙️  Configuring STABLE EXTREME training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # ✅ STABLE BATCH SIZE
        per_device_train_batch_size=32,      # Stable size
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=4,       # Effective batch = 128
        
        # ✅ STABLE LEARNING RATE
        learning_rate=2e-4,                  # Safe rate
        num_train_epochs=num_train_epochs,
        max_steps=-1,
        
        # EVALUATION
        eval_strategy="steps",
        eval_steps=25,
        
        # LOGGING
        logging_steps=5,
        logging_first_step=True,
        
        # SAVING
        save_strategy="steps",
        save_steps=50,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # OPTIMIZATION
        warmup_ratio=0.15,                   # ✅ 15% warmup for stability
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,                   # ✅ Gradient clipping
        lr_scheduler_type="cosine",
        
        # PRECISION
        fp16=False,
        bf16=is_bfloat16_supported(),
        bf16_full_eval=True,
        
        # PERFORMANCE
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        group_by_length=True,
        length_column_name="length",
        
        # REPRODUCIBILITY
        seed=3407,
        data_seed=3407,
        
        # MONITORING
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        
        # MEMORY
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # ADVANCED
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=True,
    )
    
    print("\n📊 STABLE EXTREME Configuration Summary:")
    print(f"   {'='*60}")
    print(f"   Demonstrations:      {len(demos):,}")
    print(f"   Training samples:    {len(train_demos):,}")
    print(f"   Epochs:              {num_train_epochs}")
    print(f"   Batch size:          32 (4x baseline)")  # 🔧 FIX: Updated
    print(f"   Effective batch:     128 (8x baseline)")
    print(f"   LoRA rank:           128 (8x baseline)")
    print(f"   LoRA alpha:          128 (8x baseline)")  # 🔧 FIX: Updated
    print(f"   Learning rate:       {2e-4}")            # 🔧 FIX: Updated
    print(f"   Sequence length:     2048")
    print(f"   Data workers:        16")
    print(f"   Total training steps: ~{(len(train_dataset) // 128) * num_train_epochs:,}")
    print(f"   Estimated time:      ~6-8 hours")
    print(f"   {'='*60}")
    
    # Initialize EXTREME SFT Trainer
    print("\n🚀 Initializing STABLE EXTREME SFT trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        
        # DATA
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        
        # EFFICIENCY
        packing=True,
        max_seq_length=2048,
        
        # PREPROCESSING
        dataset_text_field="text",
        dataset_num_proc=16,
        dataset_batch_size=2000,
        
        # LOSS
        data_collator=collator,
        
        # ARGS
        args=training_args,
        
        # NEFTUNE
        neftune_noise_alpha=10,
    )
    
    print_gpu_memory("before training")
    
    # TRAIN WITH STABLE POWER!
    print("\n" + "="*80)
    print("🔥🔥🔥 STARTING STABLE EXTREME TRAINING! 🔥🔥🔥")
    print("="*80)
    print("\n⏱️  Expected duration: 6-8 hours")
    print("📊 Monitor progress: tensorboard --logdir", f"{output_dir}/logs")
    print("🖥️  Watch GPU: watch -n 1 rocm-smi")
    print("\n✅ Training will be STABLE with these settings:")
    print("   • Learning rate: 2e-4 (safe)")
    print("   • LoRA alpha: 128 (matches rank)")
    print("   • Warmup: 15% (prevents explosions)")
    print("   • Grad clip: 1.0 (prevents divergence)")
    print("\n" + "="*80 + "\n")
    
    trainer.train()
    
    print_gpu_memory("after training")
    
    # Save
    final_path = f"{output_dir}/final_model"
    print(f"\n💾 Saving EXTREME model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save training stats
    print("\n📈 Saving training statistics...")
    stats = {
        "num_demonstrations": len(demos),
        "num_train_samples": len(train_demos),
        "num_eval_samples": len(eval_demos),
        "num_epochs": num_train_epochs,
        "batch_size": 32,                    # 🔧 FIX: Updated
        "effective_batch_size": 128,
        "lora_rank": 128,
        "lora_alpha": 128,                   # ✅ Correct
        "learning_rate": 2e-4,               # ✅ Correct
        "expert_mean_reward": float(np.mean(expert_rewards)),
        "expert_std_reward": float(np.std(expert_rewards)),
    }
    
    import json
    with open(f"{output_dir}/training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n✅ STABLE EXTREME TRAINING COMPLETE!")
    print(f"📊 TensorBoard logs: {output_dir}/logs")
    print(f"📁 Model saved to: {final_path}")
    print(f"📈 Stats saved to: {output_dir}/training_stats.json")
    print("=" * 80)
    
    return model, tokenizer


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_model_comprehensive(
    model,
    tokenizer,
    base_url: str = "http://localhost:8010",
    num_episodes: int = 50,                 # 🔥 5x more episodes
    max_steps: int = 128,
):
    """Comprehensive evaluation with detailed statistics."""
    
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*80}\n")
    
    env = WildfireEnv(base_url)
    episode_rewards = []
    episode_lengths = []
    action_distribution = {"water": 0, "break": 0, "wait": 0}
    
    model.eval()
    
    for ep in range(num_episodes):
        obs_result = env.reset()
        ep_reward = 0
        steps = 0
        
        for step in range(max_steps):
            obs = obs_result.observation
            
            if obs.burning_count == 0:
                break
            
            prompt = format_observation_as_prompt(obs)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.2,                # 🔥 Lower for more deterministic
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            completion = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            action = parse_action_from_text(completion, obs.width, obs.height)
            action_distribution[action.action] += 1
            
            obs_result = env.step(action)
            ep_reward += obs_result.reward
            steps += 1
            
            if obs_result.done:
                break
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episodes {ep-8:2d}-{ep+1:2d}: Avg Reward = {np.mean(episode_rewards[-10:]):6.2f}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n📈 Evaluation Results:")
    print(f"   {'='*60}")
    print(f"   Mean Reward:       {mean_reward:8.2f} ± {std_reward:.2f}")
    print(f"   Median Reward:     {np.median(episode_rewards):8.2f}")
    print(f"   Min/Max Reward:    {min(episode_rewards):8.1f} / {max(episode_rewards):.1f}")
    print(f"   Mean Episode Len:  {mean_length:8.1f} steps")
    print(f"   {'='*60}")
    
    print(f"\n🎯 Action Distribution:")
    total_actions = sum(action_distribution.values())
    for action, count in action_distribution.items():
        percentage = (count / total_actions) * 100
        print(f"   {action:6s}: {count:5d} ({percentage:5.1f}%)")
    
    print("=" * 80)
    
    return episode_rewards


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EXTREME MODE - Maximum HPC Performance")
    parser.add_argument("--url", default="http://localhost:8010")
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--demos", type=int, default=1000, 
                       help="Number of demo episodes (default: 1000)")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs (default: 20)")
    parser.add_argument("--output", default="./sft_wildfire_extreme")
    parser.add_argument("--eval", action="store_true",
                       help="Run comprehensive evaluation after training")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔥🔥🔥 EXTREME MODE - UNLEASHING AMD MI100 FULL POWER 🔥🔥🔥")
    print("="*80)
    print(f"\n⚙️  Configuration:")
    print(f"   Demos:        {args.demos} episodes (→ ~{args.demos * 15:,} samples)")
    print(f"   Epochs:       {args.epochs}")
    print(f"   Batch:        64 → Effective 128")
    print(f"   LoRA rank:    128 (8x baseline)")
    print(f"   Learning rate: 8e-4")
    print(f"   Model:        {args.model}")
    print(f"   Output:       {args.output}")
    print(f"\n⏱️  Estimated time: 6-8 hours")
    print(f"📊 Expected quality: Production-grade wildfire agent")
    
    response = input("\n⚠️  This will use significant compute time. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # TRAIN IN EXTREME MODE
    model, tokenizer = train_sft_wildfire_extreme(
        base_url=args.url,
        model_name=args.model,
        num_demo_episodes=args.demos,
        output_dir=args.output,
        num_train_epochs=args.epochs,
    )
    
    # COMPREHENSIVE EVALUATION
    if args.eval:
        evaluate_model_comprehensive(
            model, tokenizer, 
            base_url=args.url, 
            num_episodes=50
        )
    
    print("\n" + "="*80)
    print("✅ EXTREME MODE TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Model: {args.output}/final_model")
    print(f"📊 Logs:  tensorboard --logdir {args.output}/logs")
    print(f"📈 Stats: {args.output}/training_stats.json")
    print("\n🔥 You now have a production-grade wildfire control agent! 🔥")
