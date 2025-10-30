"""
Fixed Inference Script with proper animation
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple
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
    
    # Get burning cell locations
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
    
    print(f"⚠️  Could not parse: '{text}', defaulting to wait")
    return WildfireAction(action="wait", x=None, y=None)


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
# SINGLE EPISODE INFERENCE
# ============================================================================

def run_single_episode(
    model,
    tokenizer,
    env: WildfireEnv,
    max_steps: int = 128,
    temperature: float = 0.3,
    verbose: bool = True,
) -> dict:
    """Run a single episode using the trained model."""
    
    obs_result = env.reset()
    obs = obs_result.observation
    
    total_reward = 0.0
    actions_taken = []
    observations = [obs]  # Store initial observation
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Starting Episode")
        print(f"{'='*80}")
        print(f"Initial state: {obs.burning_count} cells burning")
    
    for step in range(max_steps):
        if obs.burning_count == 0:
            if verbose:
                print(f"\n✅ Fire contained at step {step}!")
            break
        
        action, raw_output = predict_action(
            model, tokenizer, obs, temperature=temperature
        )
        
        if verbose:
            print(f"\nStep {step + 1}:")
            print(f"  Burning: {obs.burning_count} cells")
            print(f"  Resources: Water={obs.remaining_water}, Breaks={obs.remaining_breaks}")
            print(f"  Model output: '{raw_output.strip()}'")
            print(f"  Action: {action.action}", end="")
            if action.x is not None:
                print(f" at ({action.x}, {action.y})")
            else:
                print()
        
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
        observations.append(obs)  # Append observation AFTER each step
        
        if verbose:
            print(f"  Reward: {reward:.3f}")
        
        if obs_result.done:
            if verbose:
                print(f"\n🏁 Episode ended at step {step + 1}")
            break
    
    final_stats = {
        'total_reward': total_reward,
        'steps_taken': len(actions_taken),
        'final_burning': obs.burning_count,
        'fire_contained': obs.burning_count == 0,
        'actions': actions_taken,
        'observations': observations,
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Episode Summary")
        print(f"{'='*80}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps: {len(actions_taken)}")
        print(f"Fire Contained: {'Yes ✅' if final_stats['fire_contained'] else 'No ❌'}")
        print(f"Final Burning Cells: {obs.burning_count}")
        print(f"Total Observations Collected: {len(observations)}")
        
        action_counts = {}
        for a in actions_taken:
            action_counts[a['action']] = action_counts.get(a['action'], 0) + 1
        print(f"\nAction Breakdown:")
        for action_type, count in action_counts.items():
            print(f"  {action_type}: {count}")
    
    return final_stats


# ============================================================================
# VISUALIZATION - FIXED VERSION
# ============================================================================

def visualize_episode(
    model,
    tokenizer,
    env: WildfireEnv,
    save_path: str = None,
):
    """
    Run episode and create visualization/animation.
    FIXED: Properly stores and uses all observations.
    """
    
    print("\n🎬 Running episode with visualization...")
    
    # Run episode and collect data
    stats = run_single_episode(
        model, tokenizer, env,
        verbose=True
    )
    
    observations = stats['observations']
    actions = stats['actions']
    
    print(f"\n📊 Creating animation with {len(observations)} frames...")
    
    # Verify we have observations
    if len(observations) < 2:
        print("❌ Error: Not enough observations collected!")
        return None
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def update_frame(frame_idx):
        """Update animation frame."""
        ax1.clear()
        ax2.clear()
        
        obs = observations[frame_idx]
        
        # Plot 1: Grid visualization
        grid_2d = np.array(obs.grid).reshape(obs.height, obs.width)
        im = ax1.imshow(grid_2d, cmap='hot', vmin=0, vmax=4, interpolation='nearest')
        ax1.set_title(f"Step {frame_idx} - Burning: {obs.burning_count} cells", fontsize=14, fontweight='bold')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Cell State')
        
        # Show action if available (frame_idx 0 is initial state, actions start at frame 1)
        if frame_idx > 0 and frame_idx - 1 < len(actions):
            action = actions[frame_idx - 1]  # actions[0] corresponds to frame 1
            if action['x'] is not None:
                color = 'cyan' if action['action'] == 'water' else 'lime'
                marker = 'o' if action['action'] == 'water' else 's'
                ax1.scatter(action['x'], action['y'], c=color, s=300, marker=marker, 
                           edgecolors='white', linewidths=3, zorder=10)
                ax1.text(action['x'], action['y'] - 0.7, action['action'].upper(), 
                        ha='center', va='bottom', color='white', fontweight='bold',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Plot 2: Metrics over time
        steps = list(range(len(observations[:frame_idx + 1])))
        burning_counts = [o.burning_count for o in observations[:frame_idx + 1]]
        
        ax2.plot(steps, burning_counts, 'r-', linewidth=3, label='Burning Cells', marker='o')
        ax2.fill_between(steps, burning_counts, alpha=0.3, color='red')
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Burning Cells', fontsize=12)
        ax2.set_title('Fire Progression Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(-0.5, len(observations))
        ax2.set_ylim(0, max(burning_counts) * 1.1 if burning_counts else 10)
        
        # Add current stats text
        stats_text = f"Step {frame_idx}\n"
        stats_text += f"Burning: {obs.burning_count}\n"
        stats_text += f"Water: {obs.remaining_water}\n"
        stats_text += f"Breaks: {obs.remaining_breaks}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, family='monospace')
        
        return [im]
    
    # Create animation
    print(f"🎨 Rendering animation...")
    anim = animation.FuncAnimation(
        fig, update_frame,
        frames=len(observations),
        interval=500,
        blit=False,
        repeat=True
    )
    
    # Handle saving/display
    if save_path:
        print(f"💾 Saving animation to {save_path}")
        try:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"✅ Animation saved successfully! ({len(observations)} frames)")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")
        finally:
            plt.close()
    else:
        # Default save location for headless environments
        default_path = 'wildfire_visualization.gif'
        print(f"💾 No save path provided. Saving to {default_path}")
        try:
            anim.save(default_path, writer='pillow', fps=2)
            print(f"✅ Animation saved successfully! ({len(observations)} frames)")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")
        finally:
            plt.close()
    
    return anim


# ============================================================================
# BATCH EVALUATION
# ============================================================================

def evaluate_model(
    model,
    tokenizer,
    base_url: str,
    num_episodes: int = 10,
    temperature: float = 0.3,
) -> dict:
    """Evaluate model over multiple episodes."""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL OVER {num_episodes} EPISODES")
    print(f"{'='*80}\n")
    
    env = WildfireEnv(base_url)
    
    episode_rewards = []
    containment_successes = 0
    
    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}...", end=" ")
        
        stats = run_single_episode(
            model, tokenizer, env,
            temperature=temperature,
            verbose=False
        )
        
        episode_rewards.append(stats['total_reward'])
        if stats['fire_contained']:
            containment_successes += 1
        
        print(f"Reward: {stats['total_reward']:.2f}, Contained: {'✅' if stats['fire_contained'] else '❌'}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'containment_rate': containment_successes / num_episodes,
        'episode_rewards': episode_rewards,
    }
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Min/Max: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"Containment Rate: {results['containment_rate']*100:.1f}% ({containment_successes}/{num_episodes})")
    
    return results


# ============================================================================
# MAIN USAGE EXAMPLES
# ============================================================================

def main():
    """Main function showing different usage patterns."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with trained wildfire model")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--url", default="http://localhost:8010", help="Wildfire server URL")
    parser.add_argument("--mode", choices=["single", "eval", "visualize"], default="single",
                       help="Inference mode")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for eval mode")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--save-gif", type=str, help="Path to save GIF (visualize mode)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_trained_model(args.model)
    
    # Initialize environment
    env = WildfireEnv(args.url)
    
    # Run selected mode
    if args.mode == "single":
        print("\n🎯 Running single episode...")
        stats = run_single_episode(
            model, tokenizer, env,
            temperature=args.temperature,
            verbose=True
        )
    
    elif args.mode == "eval":
        print(f"\n📊 Evaluating over {args.episodes} episodes...")
        results = evaluate_model(
            model, tokenizer, args.url,
            num_episodes=args.episodes,
            temperature=args.temperature
        )
    
    elif args.mode == "visualize":
        print("\n🎬 Creating visualization...")
        anim = visualize_episode(
            model, tokenizer, env,
            save_path=args.save_gif
        )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()