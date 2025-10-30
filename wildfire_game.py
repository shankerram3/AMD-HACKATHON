"""
🔥 WILDFIRE CONTROL GAME 🔥
Interactive game where you compete against your trained AI model!

Usage:
    python wildfire_game.py --model ./sft_extreme_stable/final_model

Then open: http://localhost:7860
"""

import sys
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import io
from typing import Optional, Tuple, List
from dataclasses import dataclass
import json

sys.path.append("/workspace/OpenEnv/src")
from envs.wildfire_env import WildfireEnv, WildfireAction, WildfireObservation

sys.path.append("/workspace/trainingfiles")
from Supervised import (
    format_observation_as_prompt,
    parse_action_from_text,
)

from unsloth import FastLanguageModel


# ============================================================================
# GAME STATE
# ============================================================================

@dataclass
class GameState:
    """Tracks the current game state."""
    human_env: WildfireEnv
    ai_env: WildfireEnv
    human_obs: Optional[WildfireObservation] = None
    ai_obs: Optional[WildfireObservation] = None
    human_score: int = 0
    ai_score: int = 0
    turn: int = 0
    game_over: bool = False
    human_history: List = None
    ai_history: List = None
    
    def __post_init__(self):
        if self.human_history is None:
            self.human_history = []
        if self.ai_history is None:
            self.ai_history = []


# ============================================================================
# HELPER FUNCTION TO EXTRACT OBSERVATION
# ============================================================================

def extract_observation(result):
    """Extract WildfireObservation from result (handles both StepResult and direct obs)."""
    # Check if it's a StepResult object by looking for the observation attribute
    if hasattr(result, 'observation') and result.observation is not None:
        return result.observation
    # Check if it's already a WildfireObservation by checking for width attribute
    elif hasattr(result, 'width'):
        return result
    # If it has an obs attribute (some implementations use this)
    elif hasattr(result, 'obs'):
        return result.obs
    # Last resort - might be wrapped differently
    else:
        print(f"Warning: Unknown result type: {type(result)}, attributes: {dir(result)}")
        return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_fire_grid_image(obs, title="Fire Grid", agent_action=None) -> Image.Image:
    """
    Create a beautiful visualization of the fire grid.
    
    Legend:
    🟩 Green  = Vegetation (healthy)
    🟨 Yellow = Burned (destroyed)
    🔥 Red    = Burning (active fire)
    🟦 Blue   = Water drop (if action shown)
    🟫 Brown  = Firebreak (if action shown)
    """
    cell_size = 40
    width = obs.width * cell_size
    height = obs.height * cell_size + 60  # Extra space for title
    
    # Create image
    img = Image.new('RGB', (width, height), color='#1a1a1a')
    draw = ImageDraw.Draw(img)
    
    # Colors
    colors = {
        0: '#8B4513',  # Empty/Burned - brown
        1: '#2ecc71',  # Vegetation - green
        2: '#e74c3c',  # Burning - red
        3: '#8B4513',  # Firebreak - brown
        4: '#3498db',  # Watered - blue
    }
    
    # Draw title
    title_text = f"{title} | Turn {obs.step} | Fires: {obs.burning_count}"
    draw.text((10, 10), title_text, fill='white')
    
    # Draw stats
    stats = f"💧 Water: {obs.remaining_water} | 🧱 Breaks: {obs.remaining_breaks}"
    draw.text((10, 30), stats, fill='white')
    
    # Draw grid
    for y in range(obs.height):
        for x in range(obs.width):
            cell_value = obs.grid[y * obs.width + x]
            
            # Calculate position
            px = x * cell_size
            py = y * cell_size + 60
            
            # Draw cell
            color = colors.get(cell_value, '#95a5a6')
            draw.rectangle(
                [px, py, px + cell_size - 2, py + cell_size - 2],
                fill=color,
                outline='#34495e',
                width=1
            )
            
            # Draw fire emoji for burning cells
            if cell_value == 2:
                emoji_size = cell_size // 2
                emoji_x = px + cell_size // 4
                emoji_y = py + cell_size // 4
                draw.ellipse(
                    [emoji_x, emoji_y, emoji_x + emoji_size, emoji_y + emoji_size],
                    fill='#ff6b6b',
                    outline='#ff0000',
                    width=2
                )
    
    # Draw agent action if provided
    if agent_action:
        if agent_action.action == "water" and agent_action.x is not None:
            # Draw water drop
            px = agent_action.x * cell_size + cell_size // 2
            py = agent_action.y * cell_size + 60 + cell_size // 2
            draw.ellipse(
                [px - 8, py - 8, px + 8, py + 8],
                fill='#3498db',
                outline='#2980b9',
                width=3
            )
            draw.text((px - 6, py - 8), "💧", fill='white')
        
        elif agent_action.action == "break" and agent_action.x is not None:
            # Draw firebreak
            px = agent_action.x * cell_size + cell_size // 2
            py = agent_action.y * cell_size + 60 + cell_size // 2
            draw.rectangle(
                [px - 10, py - 10, px + 10, py + 10],
                fill='#8B4513',
                outline='#654321',
                width=3
            )
    
    return img


def create_comparison_image(human_obs, ai_obs, human_score, ai_score, turn):
    """Create side-by-side comparison of human vs AI."""
    human_img = create_fire_grid_image(human_obs, f"YOU - Score: {human_score}")
    ai_img = create_fire_grid_image(ai_obs, f"AI - Score: {ai_score}")
    
    # Combine images side by side
    total_width = human_img.width + ai_img.width + 20
    total_height = max(human_img.height, ai_img.height)
    
    combined = Image.new('RGB', (total_width, total_height), color='#1a1a1a')
    combined.paste(human_img, (0, 0))
    combined.paste(ai_img, (human_img.width + 20, 0))
    
    return combined


# ============================================================================
# AI AGENT
# ============================================================================

class AIAgent:
    """Your trained AI wildfire agent."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def predict(self, obs, temperature=0.2) -> Tuple[WildfireAction, str, float]:
        """
        Predict action with reasoning and confidence.
        
        Returns:
            action: The predicted action
            reasoning: Text explanation of decision
            confidence: Model confidence (0-1)
        """
        # Format prompt
        prompt = format_observation_as_prompt(obs)
        
        # Generate action
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode action
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        action_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        action = parse_action_from_text(action_text, obs.width, obs.height)
        
        # Calculate confidence (avg probability of generated tokens)
        scores = outputs.scores
        if scores:
            probs = torch.stack([torch.softmax(s, dim=-1).max() for s in scores])
            confidence = probs.mean().item()
        else:
            confidence = 0.5
        
        # Generate reasoning
        reasoning = self._explain_action(obs, action, confidence)
        
        return action, reasoning, confidence
    
    def _explain_action(self, obs, action: WildfireAction, confidence: float) -> str:
        """Generate human-readable explanation of action."""
        reasons = []
        
        # Analyze situation
        if obs.burning_count == 0:
            reasons.append("✅ No active fires - situation under control")
        elif obs.burning_count > 10:
            reasons.append(f"🚨 HIGH ALERT: {obs.burning_count} active fires!")
        else:
            reasons.append(f"🔥 {obs.burning_count} active fire(s) detected")
        
        # Explain action choice
        if action.action == "water":
            reasons.append(f"💧 Using water at ({action.x}, {action.y})")
            reasons.append(f"   Water remaining: {obs.remaining_water - 1}")
            
            # Check if targeting burning cell
            target_idx = action.y * obs.width + action.x
            if obs.grid[target_idx] == 2:
                reasons.append("   ✓ Direct hit on burning cell!")
            else:
                reasons.append("   ⚠ Preventive action (no fire yet)")
        
        elif action.action == "break":
            reasons.append(f"🧱 Creating firebreak at ({action.x}, {action.y})")
            reasons.append(f"   Breaks remaining: {obs.remaining_breaks - 1}")
        
        elif action.action == "wait":
            reasons.append("⏸️ Waiting (conserving resources)")
            if obs.remaining_water == 0 and obs.remaining_breaks == 0:
                reasons.append("   ⚠ No resources left!")
        
        # Add confidence
        confidence_emoji = "🎯" if confidence > 0.8 else "🤔" if confidence > 0.5 else "😕"
        reasons.append(f"\n{confidence_emoji} Confidence: {confidence:.1%}")
        
        return "\n".join(reasons)


# ============================================================================
# GAME LOGIC
# ============================================================================

def initialize_game(env_url: str, model, tokenizer):
    """Initialize a new game with fresh environments."""
    human_env = WildfireEnv(env_url)
    ai_env = WildfireEnv(env_url)

    # Reset environments - returns StepResult with .observation attribute
    human_result = human_env.reset()
    ai_result = ai_env.reset()
    
    # StepResult has .observation, .reward, .done attributes
    human_obs = human_result.observation
    ai_obs = ai_result.observation

    game_state = GameState(
        human_env=human_env,
        ai_env=ai_env,
        human_obs=human_obs,
        ai_obs=ai_obs,
        human_score=0,
        ai_score=0,
        turn=0
    )

    ai_agent = AIAgent(model, tokenizer)
    return game_state, ai_agent


def play_turn(
    game_state: GameState,
    ai_agent: AIAgent,
    human_action_type: str,
    human_x: Optional[int],
    human_y: Optional[int],
) -> Tuple[GameState, Image.Image, str, str, int, int, str]:
    """
    Play one turn of the game.
    
    Returns:
        Updated game state and display elements
    """
    
    if game_state.game_over:
        return game_state, None, "Game Over!", "", game_state.human_score, game_state.ai_score, "Game Over"
    
    # === HUMAN TURN ===
    human_obs = game_state.human_obs
    
    # Parse human action
    if human_action_type == "💧 Water":
        if human_x is None or human_y is None:
            return game_state, None, "Please enter X and Y coordinates!", "", game_state.human_score, game_state.ai_score, "Invalid input"
        human_action = WildfireAction(action="water", x=int(human_x), y=int(human_y))
    elif human_action_type == "🧱 Firebreak":
        if human_x is None or human_y is None:
            return game_state, None, "Please enter X and Y coordinates!", "", game_state.human_score, game_state.ai_score, "Invalid input"
        human_action = WildfireAction(action="break", x=int(human_x), y=int(human_y))
    else:  # Wait
        human_action = WildfireAction(action="wait", x=None, y=None)
    
    # Execute human action
    human_result = game_state.human_env.step(human_action)
    game_state.human_score += human_result.reward
    game_state.human_history.append({
        'turn': game_state.turn,
        'action': human_action,
        'reward': human_result.reward,
        'fires': human_obs.burning_count
    })
    
    # === AI TURN ===
    ai_obs = game_state.ai_obs
    ai_action, ai_reasoning, ai_confidence = ai_agent.predict(ai_obs)
    
    # Execute AI action
    ai_result = game_state.ai_env.step(ai_action)
    game_state.ai_score += ai_result.reward
    game_state.ai_history.append({
        'turn': game_state.turn,
        'action': ai_action,
        'reward': ai_result.reward,
        'fires': ai_obs.burning_count,
        'reasoning': ai_reasoning,
        'confidence': ai_confidence
    })
    
    # Increment turn
    game_state.turn += 1
    
    # Get new observations - step() returns StepResult with .observation
    human_new_obs = human_result.observation
    ai_new_obs = ai_result.observation
    
    # Update game state with new observations
    game_state.human_obs = human_new_obs
    game_state.ai_obs = ai_new_obs
    
    # Check if game over
    human_done = human_result.done or human_new_obs.burning_count == 0
    ai_done = ai_result.done or ai_new_obs.burning_count == 0
    
    if human_done and ai_done:
        game_state.game_over = True
        winner = determine_winner(game_state)
    else:
        winner = ""
    
    # Create visualization
    comparison_img = create_comparison_image(
        human_new_obs, ai_new_obs,
        game_state.human_score, game_state.ai_score,
        game_state.turn
    )
    
    # Status message
    status = f"Turn {game_state.turn} | You: {game_state.human_score} | AI: {game_state.ai_score}"
    if game_state.game_over:
        status += f"\n\n🎮 {winner}"
    
    # Human feedback
    feedback = f"Your action: {human_action.action}"
    if human_action.x is not None:
        feedback += f" at ({human_action.x}, {human_action.y})"
    feedback += f"\nReward: {human_result.reward:+.1f}"
    feedback += f"\nFires remaining: {human_new_obs.burning_count}"
    
    return (
        game_state,
        comparison_img,
        status,
        ai_reasoning,
        game_state.human_score,
        game_state.ai_score,
        feedback
    )


def determine_winner(game_state: GameState) -> str:
    """Determine who won the game."""
    if game_state.human_score > game_state.ai_score:
        return "🎉 YOU WIN! You beat the AI! 🎉"
    elif game_state.ai_score > game_state.human_score:
        return "🤖 AI WINS! Better luck next time! 🤖"
    else:
        return "🤝 TIE! Equally matched! 🤝"


# ============================================================================
# GRADIO UI
# ============================================================================

def create_game_interface(model_path: str, env_url: str = "http://localhost:8010"):
    """Create the Gradio interface for the game."""
    
    # Load model
    print(f"🤖 Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model.eval()
    print("✅ Model loaded!")
    
    # Game state (will be initialized on new game)
    game_state_container = [None]
    ai_agent_container = [None]
    
    def new_game():
        """Start a new game."""
        game_state, ai_agent = initialize_game(env_url, model, tokenizer)
        game_state_container[0] = game_state
        ai_agent_container[0] = ai_agent
        
        # The observations are already in game_state.human_obs / ai_obs
        img = create_comparison_image(
            game_state.human_obs, game_state.ai_obs, 0, 0, 0
        )
        
        return (
            img,
            "🔥 New game started! Your turn!",
            "AI is ready...",
            0,
            0,
            "Choose your action below"
        )
    
    def take_turn(action_type, x, y):
        """Execute one turn of gameplay."""
        game_state = game_state_container[0]
        ai_agent = ai_agent_container[0]
        
        if game_state is None:
            return None, "Start a new game first!", "", 0, 0, ""
        
        # Play turn
        (
            updated_state,
            img,
            status,
            ai_reasoning,
            human_score,
            ai_score,
            feedback
        ) = play_turn(game_state, ai_agent, action_type, x, y)
        
        game_state_container[0] = updated_state
        
        return img, status, ai_reasoning, human_score, ai_score, feedback
    
    # Create UI
    with gr.Blocks(title="🔥 Wildfire Control Challenge", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔥 Wildfire Control Challenge 🔥
        ### Can you beat the AI at fighting wildfires?
        
        **How to Play:**
        1. Click "🎮 Start New Game"
        2. Choose your action (Water, Firebreak, or Wait)
        3. Enter coordinates (0-15) if using Water or Firebreak
        4. Click "▶️ Take Turn" to execute
        5. Compare your strategy with the AI!
        
        **Goal:** Control the wildfire more effectively than the AI to get the highest score!
        """)
        
        with gr.Row():
            new_game_btn = gr.Button("🎮 Start New Game", variant="primary", size="lg")
        
        with gr.Row():
            # Main game display
            with gr.Column(scale=2):
                game_display = gr.Image(label="Game State", height=600)
                status_text = gr.Textbox(
                    label="Game Status",
                    lines=3,
                    max_lines=5,
                    interactive=False
                )
        
            # Controls and info
            with gr.Column(scale=1):
                gr.Markdown("### 🎮 Your Controls")
                
                action_type = gr.Radio(
                    ["💧 Water", "🧱 Firebreak", "⏸️ Wait"],
                    label="Action Type",
                    value="💧 Water"
                )
                
                with gr.Row():
                    x_coord = gr.Number(
                        label="X Coordinate (0-15)",
                        minimum=0,
                        maximum=15,
                        value=0,
                        precision=0
                    )
                    y_coord = gr.Number(
                        label="Y Coordinate (0-15)",
                        minimum=0,
                        maximum=15,
                        value=0,
                        precision=0
                    )
                
                turn_btn = gr.Button("▶️ Take Turn", variant="primary", size="lg")
                
                feedback_text = gr.Textbox(
                    label="Your Last Action",
                    lines=4,
                    interactive=False
                )
                
                gr.Markdown("### 📊 Scores")
                with gr.Row():
                    human_score = gr.Number(label="Your Score", value=0, interactive=False)
                    ai_score = gr.Number(label="AI Score", value=0, interactive=False)
                
                gr.Markdown("### 🤖 AI Reasoning")
                ai_reasoning_box = gr.Textbox(
                    label="What the AI is thinking",
                    lines=8,
                    interactive=False
                )
        
        # Event handlers
        new_game_btn.click(
            fn=new_game,
            outputs=[
                game_display,
                status_text,
                ai_reasoning_box,
                human_score,
                ai_score,
                feedback_text
            ]
        )
        
        turn_btn.click(
            fn=take_turn,
            inputs=[action_type, x_coord, y_coord],
            outputs=[
                game_display,
                status_text,
                ai_reasoning_box,
                human_score,
                ai_score,
                feedback_text
            ]
        )
        
        gr.Markdown("""
        ---
        ### 🎯 Tips:
        - **Water** is most effective on burning cells (red)
        - **Firebreaks** prevent fire from spreading
        - **Wait** to conserve resources when the situation is under control
        - Watch the AI's reasoning to learn advanced strategies!
        """)
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="🔥 Wildfire Control Game")
    parser.add_argument(
        "--model",
        type=str,
        default="./sft_extreme_stable/final_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8010",
        help="Wildfire environment URL"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔥 WILDFIRE CONTROL GAME")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Environment: {args.url}")
    print(f"Port: {args.port}")
    print("="*80 + "\n")
    
    # Create and launch interface
    demo = create_game_interface(args.model, args.url)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )