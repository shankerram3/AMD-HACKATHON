
import os
import random, uuid
from typing import List
from dataclasses import replace
from core.env_server import Environment
from ..models import WildfireAction, WildfireObservation, WildfireState

# Helpers
DIRS_8 = {
    "N":  (0, -1), "NE": (1, -1), "E":  (1, 0), "SE": (1, 1),
    "S":  (0,  1), "SW": (-1, 1), "W":  (-1, 0), "NW": (-1, -1),
    "CALM": (0, 0),
}

def idx(x: int, y: int, w: int) -> int:
    return y * w + x

def in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


class WildfireEnvironment(Environment):
    """
    Weather-aware wildfire simulation.

    Grid encodings:
      0 = ash (burned out)
      1 = fuel / vegetation
      2 = burning
      3 = firebreak
      4 = watered / damp

    Each step:
      - agent acts (water/break/wait)
      - burning spreads to neighbors with wind + humidity effects
      - burning cells burn for multiple ticks, then become ash
    """

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        base_ignite_prob: float = 0.30,
        wind_bias: float = 0.20,      # kept for compatibility (not directly used in B model)
        diag_factor: float = 0.7,     # kept for compatibility (not directly used in B model)
        humidity: float = 0.25,
        init_sources: int = 2,
        seed: int = 3407,
        max_steps: int = 128,
        water_capacity: int = 8,      # ↓ encourage strategic water use
        break_capacity: int = 50,
    ):
        super().__init__()

        # --- Env-var overrides (optional) ---
        self.width     = int(os.environ.get("WILDFIRE_WIDTH", width))
        self.height    = int(os.environ.get("WILDFIRE_HEIGHT", height))
        self.humidity  = float(os.environ.get("WILDFIRE_HUMIDITY", humidity))
        self.w = width
        self.h = height
        self.base_ignite_prob = float(os.environ.get("WILDFIRE_BASE_IGNITE_PROB", base_ignite_prob))
        self.wind_bias        = float(os.environ.get("WILDFIRE_WIND_BIAS", wind_bias))
        self.diag_factor      = float(os.environ.get("WILDFIRE_DIAG_FACTOR", diag_factor))
        self.init_sources     = int(os.environ.get("WILDFIRE_INIT_SOURCES", init_sources))
        self.rng              = random.Random(int(os.environ.get("WILDFIRE_SEED", seed)))
        self.max_steps        = int(os.environ.get("WILDFIRE_MAX_STEPS", max_steps))
        self.init_water       = int(os.environ.get("WILDFIRE_WATER_CAPACITY", water_capacity))
        self.init_breaks      = int(os.environ.get("WILDFIRE_BREAK_CAPACITY", break_capacity))
        self.burn_lifetime    = int(os.environ.get("WILDFIRE_BURN_LIFETIME", 3))  # ✅ new
        self.forced_wind = os.getenv("WILDFIRE_WIND", "CALM")
        self._state = WildfireState()
    # --- Core API ---

    def reset(self) -> WildfireObservation:
        # Start with all fuel
        grid = [1] * (self.w * self.h)

        # Wind (forced if provided)
        if self.forced_wind and self.forced_wind in DIRS_8:
            wind_dir = self.forced_wind
        else:
            wind_dir = self.rng.choice(list(DIRS_8.keys()))

        # Humidity small variation around init
        humidity = min(1.0, max(0.0, self.humidity + self.rng.uniform(-0.05, 0.05)))

        # Place initial fires
        for _ in range(self.init_sources):
            x = self.rng.randrange(self.w)
            y = self.rng.randrange(self.h)
            grid[idx(x, y, self.w)] = 2

        self._state = WildfireState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_burned=0,
            total_extinguished=0,
            last_action="reset",
            width=self.w,
            height=self.h,
            wind_dir=wind_dir,
            humidity=humidity,
            remaining_water=self.init_water,
            remaining_breaks=self.init_breaks,
            grid=grid,
        )

        # per-cell burn timers (persist across steps)
        self._state.burn_timers = [0] * (self.w * self.h)

        obs = self._make_observation(reward_hint=0.0)
        return obs

    def step(self, action: WildfireAction) -> WildfireObservation:
        st = self._state
        reward = 0.0

        # --- Agent action effects ---
        if (
            action.action == "water"
            and st.remaining_water > 0
            and action.x is not None
            and action.y is not None
        ):
            reward += self._apply_water(action.x, action.y)
        elif (
            action.action == "break"
            and st.remaining_breaks > 0
            and action.x is not None
            and action.y is not None
        ):
            reward += self._apply_break(action.x, action.y)
        elif action.action == "wait":
            pass
        else:
            reward -= 0.05  # invalid or exhausted resources

        # --- Natural fire dynamics ---
        prev_burning = self._burning_count()
        prev_burned = sum(1 for v in st.grid if v == 0)

        newly_burned = self._spread_fire()
        new_burning = self._burning_count()
        now_burned = sum(1 for v in st.grid if v == 0)

        st.total_burned += newly_burned
        st.step_count += 1
        st.last_action = action.action

        # --- Spread vs containment shaping ---
        spread_delta = new_burning - prev_burning
        burned_delta = now_burned - prev_burned

        # Strong penalty for spread
        if spread_delta > 0:
            reward -= 0.15 * spread_delta  # 🔥 focus on containment
        elif spread_delta < 0:
            reward += 0.10 * abs(spread_delta)  # reward shrinkage

        # Mild penalty for newly burned cells (area loss)
        if burned_delta > 0:
            reward -= 0.05 * burned_delta

        # Small time penalty to prefer fast control
        reward -= 0.01

        done = self._is_done()

        # --- End of episode bonuses ---
        if done:
            saved_ratio = self._saved_cells() / (self.w * self.h)
            burned_ratio = now_burned / (self.w * self.h)
            burning_left = self._burning_count()

            # Big containment bonus
            if burning_left == 0:
                reward += 0.5 + 0.5 * saved_ratio

            # Fallback proportional reward
            reward += 0.2 * (1.0 - burned_ratio)

        obs = self._make_observation(reward_hint=reward)
        obs.done = done
        obs.reward = reward
        return obs


    # --- Internal mechanics ---

    def _apply_water(self, x: int, y: int) -> float:
        st = self._state
        if not in_bounds(x, y, self.w, self.h):
            return -0.05

        # Strong penalty if no water left
        if st.remaining_water <= 0:
            return -0.5

        i = idx(x, y, self.w)
        reward = 0.0

        if st.grid[i] == 2:
            st.grid[i] = 4  # extinguish & dampen
            st.burn_timers[i] = 0
            st.total_extinguished += 1
            reward += 0.25
        elif st.grid[i] == 1:
            st.grid[i] = 4  # dampen fuel (mild penalty to avoid spamming)
            st.burn_timers[i] = 0
            reward -= 0.10
        elif st.grid[i] == 4:
            # redundant watering
            reward -= 0.05
        else:
            # watering ash/break gives slight penalty
            reward -= 0.05

        st.remaining_water -= 1
        return reward

    def _apply_break(self, x: int, y: int) -> float:
        st = self._state
        if not in_bounds(x, y, self.w, self.h):
            return -0.05
        i = idx(x, y, self.w)
        reward = 0.0

        if st.grid[i] in (1, 4):
            st.grid[i] = 3
            st.burn_timers[i] = 0
            reward += 0.15  # slightly more than before to make firebreaks attractive
        elif st.grid[i] == 2:
            st.grid[i] = 3
            st.burn_timers[i] = 0
            reward -= 0.02
        elif st.grid[i] == 3:
            reward -= 0.01
        else:
            reward -= 0.02

        st.remaining_breaks -= 1
        return reward

    def _spread_fire(self) -> int:
        """
        Balanced wildfire spread model:
          - burning cells persist for multiple ticks before turning to ash
          - 8-direction spread (diagonals weaker)
          - wind accelerates in wind direction, weakens upwind
          - humidity suppresses ignition probability
          - water (4) is IMMUNE to ignition while damp and reverts to fuel after several ticks
        """
        st = self._state
        new_grid = st.grid[:]
        newly_burned = 0

        # 8-neighbor model
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (1, -1), (-1, 1), (1, 1)]
        wx, wy = DIRS_8.get(st.wind_dir, (0, 0))

        base = self.base_ignite_prob
        humidity_factor = (1.0 - st.humidity)

        ignite_flags = [False] * (self.w * self.h)

        # First pass: evaluate ignitions, increment burn timers
        for y in range(self.h):
            for x in range(self.w):
                i = idx(x, y, self.w)
                cell = st.grid[i]

                if cell == 2:  # burning
                    st.burn_timers[i] += 1

                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if not in_bounds(nx, ny, self.w, self.h):
                            continue
                        ni = idx(nx, ny, self.w)
                        target = st.grid[ni]

                        # Only fuel or damp can be candidates, but WATER IS IMMUNE during damp
                        if target == 4:
                            # Damp cells do not ignite at all while damp
                            continue
                        if target != 1:
                            continue

                        # Wind multiplier
                        if (dx, dy) == (wx, wy):
                            wind_mult = 2.0
                        elif (dx, dy) == (-wx, -wy):
                            wind_mult = 0.5
                        else:
                            wind_mult = 1.0

                        # Diagonals weaker
                        diag_mult = 0.6 if (dx != 0 and dy != 0) else 1.0

                        p = base * humidity_factor * wind_mult * diag_mult
                        p = max(0.0, min(1.0, p))
                        if self.rng.random() < p:
                            ignite_flags[ni] = True

        # Second pass: apply transitions
        for i, cell in enumerate(st.grid):
            if cell == 2:
                # burns for burn_lifetime ticks before turning to ash
                if st.burn_timers[i] >= self.burn_lifetime:
                    new_grid[i] = 0  # ash
                    newly_burned += 1
                else:
                    new_grid[i] = 2  # keep burning
            elif ignite_flags[i] and new_grid[i] == 1:
                new_grid[i] = 2
                st.burn_timers[i] = 0
            elif cell == 4:
                # Water stays damp for several ticks before reverting to fuel
                st.burn_timers[i] += 1
                if st.burn_timers[i] >= 6:   # was 3; extend to make water useful
                    new_grid[i] = 1

        st.grid = new_grid
        return newly_burned

    def _burning_count(self) -> int:
        return sum(1 for v in self._state.grid if v == 2)

    def _saved_cells(self) -> int:
        # cells not turned to ash (includes fuel, burning, break, water)
        return sum(1 for v in self._state.grid if v in (1, 2, 3, 4))

    def _is_done(self) -> bool:
        return self._burning_count() == 0 or self._state.step_count >= self.max_steps

    def _make_observation(self, reward_hint: float = 0.0) -> WildfireObservation:
        st = self._state
        burning = self._burning_count()
        burned = sum(1 for v in st.grid if v == 0)
        return WildfireObservation(
            grid=st.grid[:],
            width=self.w,
            height=self.h,
            step=st.step_count,
            wind_dir=st.wind_dir,
            humidity=st.humidity,
            burning_count=burning,
            remaining_water=st.remaining_water,     # ✅ new
            remaining_breaks=st.remaining_breaks,   # ✅ new
            burned_count=burned,
            reward_hint=reward_hint,
        )
           # --- Required abstract property implementation ---
    @property
    def state(self) -> WildfireState:
     """Return the current environment state."""
     return self._state

