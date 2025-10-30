"""
Enhanced Inference Script with AI vs Baseline Comparison
Creates side-by-side GIF visualizations comparing trained AI with baseline agent
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib import animation
import re

sys.path.append("/workspace/OpenEnv/src")
from envs.wildfire_env import WildfireEnv, WildfireAction

# Import Unsloth for loading models
from unsloth import FastLanguageModel

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(model_path: str):
    """Load a trained model from disk."""
    print(f"📦 Loading model from: {model_path}")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("✅ Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


# ============================================================================
# OBSERVATION FORMATTING
# ============================================================================

def format_observation_for_inference(obs) -> str:
    """Format observation exactly like during training."""
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
    
    prompt = f"""<|im_start|>system
Wildfire agent. Choose action: "water X Y", "break X Y", or "wait"
<|im_end|>
<|im_start|>user
Fire: {obs.burning_count} at {fire_locs}
Wind: {obs.wind_dir}, Water: {obs.remaining_water}, Breaks: {obs.remaining_breaks}
<|im_end|>
<|im_start|>assistant
"""
    return prompt


# ============================================================================
# ACTION PARSING
# ============================================================================

def parse_action_from_model_output(text: str, width: int, height: int) -> WildfireAction:
    """Parse model output into WildfireAction."""
    text = text.strip().lower()
    
    # Try to extract "water X Y"
    water_match = re.search(r'water\s+(\d+)\s+(\d+)', text)
    if water_match:
        x = int(water_match.group(1))
        y = int(water_match.group(2))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return WildfireAction(action="water", x=x, y=y)
    
    # Try to extract "break X Y"
    break_match = re.search(r'break\s+(\d+)\s+(\d+)', text)
    if break_match:
        x = int(break_match.group(1))
        y = int(break_match.group(2))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return WildfireAction(action="break", x=x, y=y)
    
    # Check for "wait"
    if "wait" in text:
        return WildfireAction(action="wait", x=None, y=None)
    
    return WildfireAction(action="wait", x=None, y=None)


# ============================================================================
# BASELINE AGENTS
# ============================================================================

def baseline_random_action(obs) -> WildfireAction:
    """Baseline: Random action."""
    action_type = np.random.choice(['water', 'break', 'wait'], p=[0.4, 0.3, 0.3])
    
    if action_type == 'wait':
        return WildfireAction(action="wait", x=None, y=None)
    
    x = np.random.randint(0, obs.width)
    y = np.random.randint(0, obs.height)
    return WildfireAction(action=action_type, x=x, y=y)


def baseline_wait_only(obs) -> WildfireAction:
    """Baseline: Always wait (do nothing)."""
    return WildfireAction(action="wait", x=None, y=None)


def baseline_nearest_fire(obs) -> WildfireAction:
    """Baseline: Water nearest burning cell."""
    # Find burning cells
    burning_cells = []
    for y in range(obs.height):
        for x in range(obs.width):
            if obs.grid[y * obs.width + x] == 2:
                burning_cells.append((x, y))
    
    if not burning_cells or obs.remaining_water <= 0:
        return WildfireAction(action="wait", x=None, y=None)
    
    # Pick first burning cell
    x, y = burning_cells[0]
    return WildfireAction(action="water", x=x, y=y)


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def predict_action(
    model,
    tokenizer,
    obs,
    temperature: float = 0.3,
    max_new_tokens: int = 20,
) -> Tuple[WildfireAction, str]:
    """Use model to predict action for given observation."""
    prompt = format_observation_for_inference(obs)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    action = parse_action_from_model_output(generated_text, obs.width, obs.height)
    return action, generated_text


# ============================================================================
# EPISODE RUNNER
# ============================================================================

def run_episode_with_agent(
    env: WildfireEnv,
    agent_fn,
    agent_name: str,
    max_steps: int = 128,
    verbose: bool = False,
) -> dict:
    """Run episode with any agent function."""
    obs_result = env.reset()
    obs = obs_result.observation
    
    total_reward = 0.0
    actions_taken = []
    observations = [obs]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{agent_name} - Starting Episode")
        print(f"{'='*80}")
    
    for step in range(max_steps):
        if obs.burning_count == 0:
            if verbose:
                print(f"✅ Fire contained at step {step}!")
            break
        
        action = agent_fn(obs)
        
        obs_result = env.step(action)
        obs = obs_result.observation
        reward = obs_result.reward
        total_reward += reward
        
        actions_taken.append({
            'step': step,
            'action': action.action,
            'x': action.x,
            'y': action.y,
            'reward': reward,
            'burning_count': obs.burning_count,
        })
        observations.append(obs)
        
        if obs_result.done:
            break
    
    return {
        'agent_name': agent_name,
        'total_reward': total_reward,
        'steps_taken': len(actions_taken),
        'final_burning': obs.burning_count,
        'fire_contained': obs.burning_count == 0,
        'actions': actions_taken,
        'observations': observations,
    }


# ============================================================================
# COMPARISON VISUALIZATION
# ============================================================================

def create_comparison_gif(
    ai_stats: dict,
    baseline_stats: dict,
    save_path: str = "comparison.gif",
):
    """Create side-by-side GIF comparing AI vs Baseline."""
    print(f"\n🎬 Creating comparison animation...")
    
    ai_obs = ai_stats['observations']
    baseline_obs = baseline_stats['observations']
    
    # Use minimum length to avoid index errors
    max_frames = max(len(ai_obs), len(baseline_obs))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('AI Agent vs Baseline Comparison', fontsize=16, fontweight='bold')
    
    def update_frame(frame_idx):
        """Update animation frame."""
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
        
        # AI Agent (left column)
        if frame_idx < len(ai_obs):
            ai_obs_current = ai_obs[frame_idx]
            grid_ai = np.array(ai_obs_current.grid).reshape(ai_obs_current.height, ai_obs_current.width)
            
            im1 = axes[0, 0].imshow(grid_ai, cmap='hot', vmin=0, vmax=4, interpolation='nearest')
            axes[0, 0].set_title(f"AI Agent - Step {frame_idx}\nBurning: {ai_obs_current.burning_count}", 
                                fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel("X")
            axes[0, 0].set_ylabel("Y")
            
            # Show AI action
            if frame_idx > 0 and frame_idx - 1 < len(ai_stats['actions']):
                action = ai_stats['actions'][frame_idx - 1]
                if action['x'] is not None:
                    color = 'cyan' if action['action'] == 'water' else 'lime'
                    marker = 'o' if action['action'] == 'water' else 's'
                    axes[0, 0].scatter(action['x'], action['y'], c=color, s=300, marker=marker,
                                      edgecolors='white', linewidths=3, zorder=10)
        
        # Baseline Agent (right column)
        if frame_idx < len(baseline_obs):
            baseline_obs_current = baseline_obs[frame_idx]
            grid_baseline = np.array(baseline_obs_current.grid).reshape(
                baseline_obs_current.height, baseline_obs_current.width)
            
            im2 = axes[0, 1].imshow(grid_baseline, cmap='hot', vmin=0, vmax=4, interpolation='nearest')
            axes[0, 1].set_title(f"Baseline - Step {frame_idx}\nBurning: {baseline_obs_current.burning_count}",
                                fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel("X")
            axes[0, 1].set_ylabel("Y")
            
            # Show baseline action
            if frame_idx > 0 and frame_idx - 1 < len(baseline_stats['actions']):
                action = baseline_stats['actions'][frame_idx - 1]
                if action['x'] is not None:
                    color = 'cyan' if action['action'] == 'water' else 'lime'
                    marker = 'o' if action['action'] == 'water' else 's'
                    axes[0, 1].scatter(action['x'], action['y'], c=color, s=300, marker=marker,
                                      edgecolors='white', linewidths=3, zorder=10)
        
        # Metrics comparison (bottom row)
        steps_ai = list(range(min(frame_idx + 1, len(ai_obs))))
        steps_baseline = list(range(min(frame_idx + 1, len(baseline_obs))))
        
        burning_ai = [ai_obs[i].burning_count for i in range(min(frame_idx + 1, len(ai_obs)))]
        burning_baseline = [baseline_obs[i].burning_count for i in range(min(frame_idx + 1, len(baseline_obs)))]
        
        # Burning cells comparison
        axes[1, 0].plot(steps_ai, burning_ai, 'b-', linewidth=3, label='AI Agent', marker='o')
        axes[1, 0].plot(steps_baseline, burning_baseline, 'r-', linewidth=3, label='Baseline', marker='s')
        axes[1, 0].set_xlabel('Step', fontsize=11)
        axes[1, 0].set_ylabel('Burning Cells', fontsize=11)
        axes[1, 0].set_title('Fire Progression', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(-0.5, max_frames)
        
        # Cumulative rewards
        if frame_idx > 0:
            rewards_ai = [ai_stats['actions'][i]['reward'] for i in range(min(frame_idx, len(ai_stats['actions'])))]
            rewards_baseline = [baseline_stats['actions'][i]['reward'] for i in range(min(frame_idx, len(baseline_stats['actions'])))]
            
            cumsum_ai = np.cumsum([0] + rewards_ai)
            cumsum_baseline = np.cumsum([0] + rewards_baseline)
            
            axes[1, 1].plot(range(len(cumsum_ai)), cumsum_ai, 'b-', linewidth=3, label='AI Agent', marker='o')
            axes[1, 1].plot(range(len(cumsum_baseline)), cumsum_baseline, 'r-', linewidth=3, label='Baseline', marker='s')
            axes[1, 1].set_xlabel('Step', fontsize=11)
            axes[1, 1].set_ylabel('Cumulative Reward', fontsize=11)
            axes[1, 1].set_title('Reward Accumulation', fontsize=12, fontweight='bold')
            axes[1, 1].legend(fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim(-0.5, max_frames)
        
        plt.tight_layout()
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update_frame,
        frames=max_frames,
        interval=500,
        blit=False,
        repeat=True
    )
    
    # Save
    print(f"💾 Saving comparison to {save_path}...")
    try:
        anim.save(save_path, writer='pillow', fps=2)
        print(f"✅ Comparison saved! ({max_frames} frames)")
    except Exception as e:
        print(f"❌ Error saving: {e}")
    finally:
        plt.close()
    
    return anim


# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def compare_agents(
    model,
    tokenizer,
    env: WildfireEnv,
    baseline_type: str = "random",
    temperature: float = 0.3,
    save_path: str = "ai_vs_baseline.gif",
):
    """Run comparison between AI and baseline agent."""
    
    print(f"\n{'='*80}")
    print(f"COMPARING AI AGENT VS {baseline_type.upper()} BASELINE")
    print(f"{'='*80}\n")
    
    # Select baseline
    baseline_agents = {
        'random': baseline_random_action,
        'wait': baseline_wait_only,
        'nearest': baseline_nearest_fire,
    }
    
    if baseline_type not in baseline_agents:
        print(f"⚠️  Unknown baseline '{baseline_type}', using 'random'")
        baseline_type = 'random'
    
    baseline_fn = baseline_agents[baseline_type]
    
    # Define AI agent wrapper
    def ai_agent(obs):
        action, _ = predict_action(model, tokenizer, obs, temperature=temperature)
        return action
    
    # Run AI agent
    print("🤖 Running AI Agent...")
    ai_stats = run_episode_with_agent(env, ai_agent, "AI Agent", verbose=True)
    
    # Reset environment with same seed for fair comparison
    print(f"\n🎲 Running {baseline_type.title()} Baseline...")
    baseline_stats = run_episode_with_agent(env, baseline_fn, f"{baseline_type.title()} Baseline", verbose=True)
    
    # Print comparison
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'AI Agent':<20} {baseline_type.title() + ' Baseline':<20}")
    print("-" * 80)
    print(f"{'Total Reward':<25} {ai_stats['total_reward']:<20.2f} {baseline_stats['total_reward']:<20.2f}")
    print(f"{'Steps Taken':<25} {ai_stats['steps_taken']:<20} {baseline_stats['steps_taken']:<20}")
    print(f"{'Fire Contained':<25} {'✅ Yes' if ai_stats['fire_contained'] else '❌ No':<20} {'✅ Yes' if baseline_stats['fire_contained'] else '❌ No':<20}")
    print(f"{'Final Burning Cells':<25} {ai_stats['final_burning']:<20} {baseline_stats['final_burning']:<20}")
    
    # Create comparison GIF
    create_comparison_gif(ai_stats, baseline_stats, save_path)
    
    return ai_stats, baseline_stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare AI agent with baseline")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--url", default="http://localhost:8010", help="Server URL")
    parser.add_argument("--baseline", choices=["random", "wait", "nearest"], default="random",
                       help="Baseline agent type")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--output", default="ai_vs_baseline.gif", help="Output GIF path")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_trained_model(args.model)
    
    # Initialize environment
    env = WildfireEnv(args.url)
    
    # Run comparison
    compare_agents(
        model, tokenizer, env,
        baseline_type=args.baseline,
        temperature=args.temperature,
        save_path=args.output
    )
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()