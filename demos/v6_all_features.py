"""
v6_all_features.py  –  CPG Hexapod – All 5 Bio-Inspired Features
=================================================================
Combines every enhancement from variants 1–5 into one simulation.

Feature summary
───────────────
1. Hebbian / STDP Synaptic Plasticity
   Coupling weights are a 6×6 dynamic matrix. Pairs that maintain
   their desired phase relationship strengthen; chaotic pairs weaken.
   Weights reset to the genomic baseline each assessment.

2. Proprioception (Sensory Feedback Reflex)
   CPGNetwork.step() accepts a per-leg feedback vector. When a leg
   detects a collision load above threshold, its phase is immediately
   snapped to swing onset (stumble-and-recover reflex).

3. Morphological Computation (Body–Brain Co-evolution)
   Genome extended to 13 genes: leg_length and joint_stiffness.
   Longer legs scale drive and visual segments; stiff joints store
   and release elastic energy (spring-mass locomotion).

4. Foraging Behaviour (Energy-Based Fitness)
   FoodPellet objects grant calories. Agents die when energy runs out.
   Fitness is driven by calories gathered, not raw distance.

5. Neuromodulation (Fatigue)
   A _fatigue variable accumulates with movement. At high fatigue,
   effective CPG frequency and amplitude are both downregulated.
   Eating food provides partial fatigue recovery.

All five effects interact: fatigue modulates the Hebbian learning rate,
proprioception interacts with spring energy, and morphology affects
how quickly the agent tires. The GA must find (body, brain, lifetime
behaviour) triples that survive the foraging trial.
"""

import math
import json
import os

import numpy as np

from OpenGL.GL import (
    glBegin, glEnd, glVertex2d, glColor4fv, glLineWidth,
    GL_LINE_LOOP, GL_LINE_STRIP, GL_LINES, GL_QUADS,
    glEnable, glDisable, GL_LINE_SMOOTH, GL_BLEND,
    glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    glPointSize, GL_POINTS
)
from OpenGL.GLU import gluNewQuadric, gluDisk, gluDeleteQuadric

from core.agent.agent import Agent
from core.evolve.evolver import Evolver
from core.evolve.base import NormalMutator
from core.evolve.genetic_algorithm import GeneticAlgorithm
from core.evolve.population import Population
from core.simulation import Simulation
from core.utils import (
    Vec2, AgentSettings as AS,
    ColourPalette, ColourType as CT,
    WORLD_DISPLAY_PARAMETERS as WDP
)
from core.world.world_object import WorldObject

IS_DEMO    = True
DEMO_NAME  = "CPG Hexapod – All Features"
CLASS_NAME = "CPGHexapodSimulation"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Simulation ────────────────────────────────────────────────────────
    POPULATION_SIZE   = 20
    GENERATIONS       = 100
    ASSESSMENTS       = 3
    TIMESTEPS         = 600

    # ── GA ────────────────────────────────────────────────────────────────
    CROSSOVER_RATE    = 0.70
    MUTATION_RATE     = 0.05
    ELITISM           = 2
    MUTATION_SIGMA    = 0.10

    # ── Mode ─────────────────────────────────────────────────────────────
    FITNESS_MODE      = 'FORAGING'
    DEFAULT_GAIT      = 'TRIPOD'
    ENVIRONMENT       = 'FLAT'

    # ── CPG baseline ─────────────────────────────────────────────────────
    CPG_FREQ          = 1.2
    CPG_COUPLING      = 0.65
    CPG_DUTY          = 0.60
    CPG_AMPLITUDE     = 0.80

    # ── Visualisation ─────────────────────────────────────────────────────
    DRAW_CPG_NETWORK  = True
    DRAW_TRAILS       = True
    DRAW_FORCE_ARROWS = True
    SAVE_BEST_GENOME  = True
    GENOME_SAVE_PATH  = "best_hexapod_all_features.json"

    # ── Physics ───────────────────────────────────────────────────────────
    TIMESTEP_DT       = 0.05
    MAX_SPEED         = 120.0
    MIN_SPEED         = 0.0

    # ── Feature 4: Foraging ───────────────────────────────────────────────
    FOOD_COUNT        = 12
    FOOD_CALORIES     = 25.0
    AGENT_START_ENERGY = 40.0
    ENERGY_PER_STEP   = 0.04

    # ── Feature 1: Hebbian / STDP ─────────────────────────────────────────
    HEBBIAN_LR        = 0.002
    HEBBIAN_DECAY     = 0.0001
    HEBBIAN_W_MIN     = 0.05
    HEBBIAN_W_MAX     = 2.0
    HEBBIAN_SYNC_THR  = 0.25    # radians

    # ── Feature 2: Proprioception ─────────────────────────────────────────
    FEEDBACK_THRESHOLD  = 0.5
    FEEDBACK_DECAY      = 0.85
    FEEDBACK_MAGNITUDE  = 1.0

    # ── Feature 3: Morphology ─────────────────────────────────────────────
    LEG_LENGTH_MIN    = 0.5
    LEG_LENGTH_MAX    = 2.0
    STIFFNESS_MIN     = 0.0
    STIFFNESS_MAX     = 1.0

    # ── Feature 5: Neuromodulation ────────────────────────────────────────
    FATIGUE_GROWTH         = 0.0008
    FATIGUE_RECOVERY       = 0.0020
    FATIGUE_FOOD_RESET     = 0.40
    FATIGUE_MAX            = 1.0
    FATIGUE_FREQ_GAIN      = 0.60
    FATIGUE_AMPLITUDE_GAIN = 0.50
    # Fatigue also slows Hebbian learning when exhausted
    FATIGUE_HEBBIAN_SCALE  = 0.30   # LR multiplied by (1 - fatigue * scale)


GAIT_PHASES = {
    'TRIPOD': np.array([0, np.pi, 0, np.pi, 0, np.pi], dtype=np.float32),
    'WAVE':   np.array([0, np.pi/3, 2*np.pi/3,
                        np.pi, 4*np.pi/3, 5*np.pi/3], dtype=np.float32),
    'RIPPLE': np.array([0, 2*np.pi/3, 4*np.pi/3,
                        np.pi/3, np.pi, 5*np.pi/3], dtype=np.float32),
}

LEG_NAMES = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CPG NETWORK  (Hebbian + proprioception + modulated freq/amplitude)
# ══════════════════════════════════════════════════════════════════════════════

class CPGNetwork:
    """
    Full bio-inspired CPG:

    State variables
    ───────────────
    phi[6]              : current oscillator phases
    coupling_weights[6][6]: dynamic Hebbian weights (Feature 1)

    Per-step inputs
    ───────────────
    feedback[6]         : proprioceptive load signals → phase reset (Feature 2)
    eff_freq            : neuromodulated frequency (Feature 5)
    eff_amplitude       : neuromodulated amplitude (Feature 5)
    fatigue             : current fatigue level, scales Hebbian LR (Features 1+5)
    """

    def __init__(self, freq, coupling, duty, amplitude, phase_offsets):
        self.freq          = float(freq)
        self.base_coupling = float(coupling)
        self.duty          = float(duty)
        self.amplitude     = float(amplitude)
        self.phi           = np.array(phase_offsets, dtype=np.float64)
        self.desired       = np.array(phase_offsets, dtype=np.float64)

        # Hebbian weight matrix (Feature 1)
        self.coupling_weights = np.full((6, 6), coupling, dtype=np.float64)
        np.fill_diagonal(self.coupling_weights, 0.0)

        # Effective (modulated) parameters
        self.eff_freq      = self.freq
        self.eff_amplitude = self.amplitude

    def reset_weights(self):
        """Reset Hebbian weights to genomic baseline (between assessments)."""
        self.coupling_weights[:] = self.base_coupling
        np.fill_diagonal(self.coupling_weights, 0.0)

    def step(self, dt,
             feedback=None,
             eff_freq=None,
             eff_amplitude=None,
             fatigue=0.0):
        """
        Parameters
        ----------
        dt            : time delta
        feedback      : 6-vector of per-leg load signals (Feature 2)
        eff_freq      : neuromodulated frequency override (Feature 5)
        eff_amplitude : neuromodulated amplitude override (Feature 5)
        fatigue       : current fatigue level (0-1) for Hebbian scaling (F1+F5)
        """
        self.eff_freq      = eff_freq      if eff_freq      is not None else self.freq
        self.eff_amplitude = eff_amplitude if eff_amplitude is not None else self.amplitude

        # ── Feature 2: Proprioceptive phase reset ────────────────────────
        if feedback is not None:
            thr = Config.FEEDBACK_THRESHOLD
            for i in range(6):
                if feedback[i] > thr:
                    self.phi[i] = self.desired[i] + 0.05  # snap to swing onset

        # ── Kuramoto integration with Hebbian weights ─────────────────────
        omega = 2.0 * np.pi * self.eff_freq
        dphi  = np.zeros(6)
        for i in range(6):
            cs = 0.0
            for j in range(6):
                if i != j:
                    delta = self.desired[j] - self.desired[i]
                    cs   += self.coupling_weights[i, j] * math.sin(
                        self.phi[j] - self.phi[i] - delta)
            dphi[i] = omega + cs
        self.phi += dphi * dt

        # ── Feature 1: Hebbian weight update ─────────────────────────────
        self._update_weights(dt, fatigue=fatigue)

    def _update_weights(self, dt, fatigue=0.0):
        # Fatigue slows learning (exhaustion impairs plasticity)
        lr = Config.HEBBIAN_LR * (
            1.0 - Config.FATIGUE_HEBBIAN_SCALE * fatigue)
        decay = Config.HEBBIAN_DECAY
        thr   = Config.HEBBIAN_SYNC_THR

        for i in range(6):
            for j in range(6):
                if i == j: continue
                delta     = self.desired[j] - self.desired[i]
                phase_err = abs((self.phi[j] - self.phi[i] - delta)
                                % (2 * math.pi))
                if phase_err > math.pi:
                    phase_err = 2 * math.pi - phase_err
                sync = 1.0 if phase_err < thr else -0.5
                dW   = lr * (sync - decay * self.coupling_weights[i, j])
                self.coupling_weights[i, j] = max(Config.HEBBIAN_W_MIN,
                    min(Config.HEBBIAN_W_MAX,
                        self.coupling_weights[i, j] + dW * dt))

    def output(self, leg_idx):
        phi_norm = self.phi[leg_idx] % (2.0 * np.pi)
        x = math.sin(phi_norm) * self.eff_amplitude
        return x, math.sin(phi_norm) > 0.0

    @property
    def phase_fractions(self):
        return (self.phi % (2.0 * np.pi)) / (2.0 * np.pi)

    def coupling_strength(self, i, j):
        delta = self.desired[j] - self.desired[i]
        return math.sin(self.phi[j] - self.phi[i] - delta)

    @property
    def avg_weight(self):
        mask = ~np.eye(6, dtype=bool)
        return float(np.mean(self.coupling_weights[mask]))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  WORLD OBJECTS
# ══════════════════════════════════════════════════════════════════════════════

class Obstacle(WorldObject):
    def __init__(self, location=None, radius=12.0):
        super().__init__(location=location, radius=radius, solid=True)
        self.colour = ColourPalette[CT.DARK_GREY]

    def draw(self):
        glColor4fv([0.35, 0.33, 0.30, 0.9])
        q = gluNewQuadric()
        gluDisk(q, 0, self.radius, 16, 1)
        gluDeleteQuadric(q)

    def __del__(self): pass


class FoodPellet(WorldObject):
    """Feature 4 – food source."""

    def __init__(self, location=None, calories=None):
        super().__init__(location=location, radius=10.0, solid=False)
        self.calories = calories if calories is not None else Config.FOOD_CALORIES
        self.eaten    = False
        self._pulse   = 0.0

    def draw(self):
        if self.eaten: return
        self._pulse = (self._pulse + 0.05) % (2 * math.pi)
        glow = 0.7 + 0.3 * math.sin(self._pulse)
        glColor4fv([0.9*glow, 0.8*glow, 0.1, 0.35])
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for a in range(20):
            ang = a / 20.0 * 2 * math.pi
            glVertex2d(14 * math.cos(ang), 14 * math.sin(ang))
        glEnd()
        glColor4fv([1.0*glow, 0.85*glow, 0.1, 0.85])
        q = gluNewQuadric(); gluDisk(q, 0, self.radius, 20, 1); gluDeleteQuadric(q)
        glColor4fv([1.0, 1.0, 0.6, 0.9]); glPointSize(4.0)
        glBegin(GL_POINTS); glVertex2d(0, 0); glEnd()

    def __del__(self): pass


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HEXAPOD AGENT
# ══════════════════════════════════════════════════════════════════════════════

class CPGHexapod(Agent, Evolver):
    """
    Genome (13 genes):
      0   freq          [0.3, 3.0] Hz
      1   coupling      [0.0, 1.0]
      2   duty          [0.3, 0.85]
      3   amplitude     [0.1, 1.0]
      4-9 phase offsets [0, 2π] × 6
      10  leg_length    [0.5, 2.0]  (Feature 3)
      11  joint_stiffness [0.0, 1.0] (Feature 3)
    """

    GENOME_LENGTH = 13
    GENE_SCALE = [
        (0.3, 3.0),   # 0 freq
        (0.0, 1.0),   # 1 coupling
        (0.3, 0.85),  # 2 duty
        (0.1, 1.0),   # 3 amplitude
    ] + [(0.0, 2 * np.pi)] * 6 + [   # 4-9 phase offsets
        (Config.LEG_LENGTH_MIN, Config.LEG_LENGTH_MAX),    # 10
        (Config.STIFFNESS_MIN,  Config.STIFFNESS_MAX),     # 11
    ]

    BASE_COXA  = 10.0
    BASE_FEMUR = 14.0
    BASE_TIBIA = 10.0

    def __init__(self):
        Agent.__init__(self, min_speed=Config.MIN_SPEED, max_speed=Config.MAX_SPEED,
                       timestep=Config.TIMESTEP_DT, random_colour=False, solid=False)
        Evolver.__init__(self)
        self.radius = 18.0
        self.colour = ColourPalette[CT.GREEN]

        default_phases = GAIT_PHASES.get(Config.DEFAULT_GAIT,
                                         GAIT_PHASES['TRIPOD']).copy()
        self.cpg = CPGNetwork(Config.CPG_FREQ, Config.CPG_COUPLING,
                               Config.CPG_DUTY, Config.CPG_AMPLITUDE,
                               default_phases)

        # Feature 3: morphological parameters
        self.leg_length      = 1.0
        self.joint_stiffness = 0.3
        self._spring_energy  = 0.0

        # Feature 2: proprioceptive feedback
        self._feedback     = np.zeros(6, dtype=np.float64)
        self._reflex_count = 0

        # Feature 5: neuromodulation
        self._fatigue = 0.0

        # Metrics
        self._distance_travelled = 0.0
        self._energy_consumed    = 0.0
        self._calories_gathered  = 0.0
        self._energy_remaining   = Config.AGENT_START_ENERGY
        self._coordination_score = 0.0
        self._step_count         = 0
        self._collision_penalty  = 0.0
        self._foods_eaten        = 0
        self._trail              = []
        self._leg_states         = [False] * 6
        self._foot_positions     = [None] * 6
        self._slope_angle        = 0.0

    # ── Genome ────────────────────────────────────────────────────────────

    def _scale_gene(self, raw, idx):
        lo, hi = self.GENE_SCALE[idx]
        return lo + (hi - lo) * max(0.0, min(1.0, raw))

    def set_genotype(self, genome):
        assert len(genome) == self.GENOME_LENGTH, (
            f"Expected {self.GENOME_LENGTH} genes, got {len(genome)}")
        g = np.asarray(genome, dtype=np.float64)
        freq     = self._scale_gene(g[0], 0)
        coupling = self._scale_gene(g[1], 1)
        duty     = self._scale_gene(g[2], 2)
        amp      = self._scale_gene(g[3], 3)
        phases   = np.array([self._scale_gene(g[4+i], 4+i) for i in range(6)],
                            dtype=np.float64)
        self.cpg = CPGNetwork(freq, coupling, duty, amp, phases)

        # Feature 3: morphological genes
        self.leg_length      = self._scale_gene(g[10], 10)
        self.joint_stiffness = self._scale_gene(g[11], 11)

    def get_genotype(self):
        g = np.zeros(self.GENOME_LENGTH, dtype=np.float64)
        lo, hi = self.GENE_SCALE[0];  g[0]  = (self.cpg.freq          - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[1];  g[1]  = (self.cpg.base_coupling  - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[2];  g[2]  = (self.cpg.duty           - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[3];  g[3]  = (self.cpg.amplitude      - lo) / (hi - lo)
        for i in range(6):
            lo, hi = self.GENE_SCALE[4+i]
            g[4+i] = (self.cpg.desired[i] - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[10]; g[10] = (self.leg_length       - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[11]; g[11] = (self.joint_stiffness  - lo) / (hi - lo)
        return g

    # ── Fitness ───────────────────────────────────────────────────────────

    def get_fitness(self):
        """
        Composite foraging fitness incorporating all five features:
        - calories gathered   (Feature 4)
        - exploration bonus   (distance)
        - plasticity bonus    (avg Hebbian weight)
        - reflex bonus        (Feature 2 — successful recoveries)
        - fatigue penalty     (Feature 5)
        - morphological cost  (Feature 3)
        - collision penalty
        """
        if self._step_count == 0: return 0.0
        base          = self._calories_gathered
        explore       = self._distance_travelled * 0.01
        plasticity    = self.cpg.avg_weight * 0.3
        reflex_bonus  = self._reflex_count * 0.3
        penalty       = self._collision_penalty * 5.0
        fatigue_pen   = self._fatigue * 3.0
        morph_cost    = (self.leg_length - 1.0)**2 * self.joint_stiffness * 1.5
        return max(0.0,
            base + explore + plasticity + reflex_bonus
            - penalty - fatigue_pen - morph_cost)

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self):
        self._fatigue            = 0.0
        self._feedback[:]        = 0.0
        self._reflex_count       = 0
        self._spring_energy      = 0.0
        self._distance_travelled = 0.0
        self._energy_consumed    = 0.0
        self._calories_gathered  = 0.0
        self._energy_remaining   = Config.AGENT_START_ENERGY
        self._coordination_score = 0.0
        self._step_count         = 0
        self._collision_penalty  = 0.0
        self._foods_eaten        = 0
        self._trail.clear()
        self.cpg.reset_weights()   # Feature 1
        super().reset()

    # ── Collisions ────────────────────────────────────────────────────────

    def on_collision(self, other):
        if isinstance(other, Obstacle):
            self._collision_penalty += 1.0
            # Feature 2: inject proprioceptive signal
            self._feedback[:] = Config.FEEDBACK_MAGNITUDE
            self._reflex_count += 1
        elif isinstance(other, FoodPellet) and not other.eaten:
            # Feature 4: consume food
            other.eaten              = True
            other.dead               = True
            self._calories_gathered += other.calories
            self._energy_remaining  += other.calories
            self._foods_eaten       += 1
            # Feature 5: eating reduces fatigue
            self._fatigue = max(0.0,
                self._fatigue - self._fatigue * Config.FATIGUE_FOOD_RESET)

    # ── Control ───────────────────────────────────────────────────────────

    def control(self):
        dt = self._timestep

        if self._energy_remaining <= 0:
            self.controls['left'] = self.controls['right'] = 0.0
            self._fatigue = max(0.0,
                self._fatigue - Config.FATIGUE_RECOVERY * dt)
            return

        # Feature 5: compute neuromodulated CPG params
        eff_freq = self.cpg.freq * (
            1.0 - Config.FATIGUE_FREQ_GAIN * self._fatigue)
        eff_amp  = self.cpg.amplitude * (
            1.0 - Config.FATIGUE_AMPLITUDE_GAIN * self._fatigue)

        # Step CPG with all active features
        self.cpg.step(
            dt,
            feedback=self._feedback,         # Feature 2
            eff_freq=eff_freq,               # Feature 5
            eff_amplitude=eff_amp,           # Feature 5
            fatigue=self._fatigue,           # Feature 1+5 interaction
        )

        # Decay proprioceptive feedback
        self._feedback *= Config.FEEDBACK_DECAY

        left_drive = right_drive = energy_tick = 0.0

        for i, name in enumerate(LEG_NAMES):
            disp, swing = self.cpg.output(i)
            self._leg_states[i] = swing
            is_left = name.startswith('L')

            if not swing:
                # Feature 3: leg_length scales drive
                drive = self.cpg.duty * disp * self.leg_length
                if is_left: left_drive  += drive / 3.0
                else:       right_drive += drive / 3.0
                energy_tick += abs(drive)
                # Feature 3: accumulate spring energy
                self._spring_energy += (
                    0.5 * self.joint_stiffness * (disp ** 2) * dt)
            else:
                # Feature 3: release stored spring energy
                spring_boost = self._spring_energy * 0.3 * self.leg_length
                if is_left: left_drive  += spring_boost / 3.0
                else:       right_drive += spring_boost / 3.0
                self._spring_energy = max(
                    0.0, self._spring_energy - spring_boost * 2)
                energy_tick += 0.02

        self.controls['left']  = max(-1.0, min(1.0, left_drive))
        self.controls['right'] = max(-1.0, min(1.0, right_drive))

        # Energy drain
        movement_cost = energy_tick * Config.ENERGY_PER_STEP * dt
        self._energy_consumed  += movement_cost
        self._energy_remaining -= movement_cost
        self._step_count       += 1

        # Feature 5: update fatigue
        self._fatigue = min(Config.FATIGUE_MAX,
            self._fatigue
            + Config.FATIGUE_GROWTH * energy_tick
            - Config.FATIGUE_RECOVERY * dt)
        self._fatigue = max(0.0, self._fatigue)

        spd = math.hypot(*self.velocity) if hasattr(self, 'velocity') else 0.0
        self._distance_travelled += spd * dt

        pair_diff = (abs(math.sin(self.cpg.phi[0] - self.cpg.phi[2])) +
                     abs(math.sin(self.cpg.phi[2] - self.cpg.phi[4])) +
                     abs(math.sin(self.cpg.phi[1] - self.cpg.phi[3])) +
                     abs(math.sin(self.cpg.phi[3] - self.cpg.phi[5])))
        self._coordination_score += (1.0 - pair_diff / 4.0)

        if Config.DRAW_TRAILS and hasattr(self, 'location'):
            self._trail.append(tuple(self.location))
            if len(self._trail) > 200: self._trail.pop(0)

    # ══════════════════════════════════════════════════════════════════════
    # RENDERING
    # ══════════════════════════════════════════════════════════════════════

    def draw(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)

        if Config.DRAW_TRAILS and len(self._trail) > 1:
            glColor4fv([0.4, 0.8, 0.4, 0.25])
            glLineWidth(1.0)
            glBegin(GL_LINE_STRIP)
            for px, py in self._trail:
                glVertex2d(px - self.location[0], py - self.location[1])
            glEnd()

        # Feature 1: Hebbian coupling network (weight-encoded)
        if Config.DRAW_CPG_NETWORK:
            self._draw_cpg_network_hebbian()

        for i in range(6):
            self._draw_leg(i)
        self._draw_body()

        if Config.DRAW_FORCE_ARROWS:
            self._draw_force_arrows()

        # Feature 4: energy bar
        self._draw_energy_bar()
        # Feature 5: fatigue ring
        self._draw_fatigue_ring()
        # Feature 2: reflex halo
        self._draw_feedback_halo()

        glDisable(GL_LINE_SMOOTH)

    # ── Sub-renderers ─────────────────────────────────────────────────────

    def _draw_cpg_network_hebbian(self):
        radius = 28.0
        for i in range(6):
            ai = i / 6.0 * 2 * math.pi - math.pi / 2
            xi = radius * math.cos(ai); yi = radius * math.sin(ai)
            for j in range(i + 1, 6):
                aj = j / 6.0 * 2 * math.pi - math.pi / 2
                xj = radius * math.cos(aj); yj = radius * math.sin(aj)
                w = self.cpg.coupling_weights[i, j]
                norm_w = (w - Config.HEBBIAN_W_MIN) / (
                    Config.HEBBIAN_W_MAX - Config.HEBBIAN_W_MIN + 1e-9)
                alpha = 0.05 + norm_w * 0.6
                glColor4fv([norm_w * 0.9, norm_w * 0.7, 1.0 - norm_w, alpha])
                glLineWidth(max(0.5, norm_w * 3.0))
                glBegin(GL_LINES)
                glVertex2d(xi, yi); glVertex2d(xj, yj)
                glEnd()
        glLineWidth(1.0)

    def _draw_energy_bar(self):
        ratio = max(0.0, min(1.0,
            self._energy_remaining / (Config.AGENT_START_ENERGY
                                      + self._calories_gathered + 0.01)))
        bar_w = 30.0; bar_h = 4.0; x0 = -15.0; y0 = -28.0
        glColor4fv([0.3, 0.0, 0.0, 0.7])
        glBegin(GL_QUADS)
        glVertex2d(x0, y0); glVertex2d(x0+bar_w, y0)
        glVertex2d(x0+bar_w, y0+bar_h); glVertex2d(x0, y0+bar_h); glEnd()
        glColor4fv([1.0-ratio, ratio, 0.1, 0.9])
        glBegin(GL_QUADS)
        glVertex2d(x0, y0); glVertex2d(x0+bar_w*ratio, y0)
        glVertex2d(x0+bar_w*ratio, y0+bar_h); glVertex2d(x0, y0+bar_h); glEnd()

    def _draw_fatigue_ring(self):
        if self._fatigue < 0.02: return
        r_ring = 20.0 + self._fatigue * 8.0
        alpha  = self._fatigue * 0.7
        glColor4fv([1.0, 0.4 * (1 - self._fatigue), 0.0, alpha])
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for a in range(24):
            ang = a / 24.0 * 2 * math.pi
            glVertex2d(r_ring * math.cos(ang), r_ring * math.sin(ang))
        glEnd()
        glLineWidth(1.0)

    def _draw_feedback_halo(self):
        max_fb = float(np.max(self._feedback))
        if max_fb < 0.05: return
        alpha = min(0.6, max_fb * 0.6)
        glColor4fv([1.0, 0.1, 0.1, alpha])
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        for a in range(24):
            ang = a / 24.0 * 2 * math.pi
            glVertex2d(24 * math.cos(ang), 24 * math.sin(ang))
        glEnd()
        glLineWidth(1.0)

    def _leg_geometry(self, leg_idx):
        # Feature 3: leg_length scales attachment points
        bw = 14.0 * self.leg_length; bh = 20.0
        return [(-bw,-bh*.6,-1),(-bw*1.2,0.,-1),(-bw,+bh*.6,-1),
                (+bw,-bh*.6,+1),(+bw*1.2,0.,+1),(+bw,+bh*.6,+1)][leg_idx]

    def _draw_leg(self, i):
        disp, swing = self.cpg.output(i)
        ax, ay, side = self._leg_geometry(i)

        coxa  = self.BASE_COXA  * self.leg_length
        femur = self.BASE_FEMUR * self.leg_length

        sweep = disp * 8.0 * side * self.leg_length
        lift  = max(0.0, disp) * 8.0 * self.leg_length if swing else 0.0

        cx = ax + side * coxa; cy = ay
        kx = ax + side * (coxa + femur * 0.5); ky = ay + femur * 0.3 + lift * 0.4
        fx = ax + side * (coxa + femur * 0.6 + sweep); fy = ay + femur * 0.5 + lift

        # Colour: fatigue-tinted (red), stiffness-tinted (blue), swing (orange)
        fb_active = self._feedback[i] > Config.FEEDBACK_THRESHOLD
        if fb_active:
            glColor4fv([1.0, 0.1, 0.1, 0.9])
        elif swing:
            glColor4fv([0.9, 0.45, 0.1, 0.9])
        else:
            r  = 0.15 + 0.7 * (1 - self.joint_stiffness) + 0.15 * self._fatigue
            b  = 0.2  + 0.7 * self.joint_stiffness
            glColor4fv([r, 0.80 * (1 - 0.5 * self._fatigue), b, 0.9])

        glLineWidth(1.5 + self.joint_stiffness * 2.0)
        glBegin(GL_LINE_STRIP)
        glVertex2d(ax, ay); glVertex2d(cx, cy)
        glVertex2d(kx, ky); glVertex2d(fx, fy)
        glEnd()

        glColor4fv([0.0, 1.0, 0.5, 1.0] if not swing else [1.0, 0.6, 0.1, 0.8])
        glPointSize(5.0 + self.leg_length * 2.0)
        glBegin(GL_POINTS); glVertex2d(fx, fy); glEnd()

        if hasattr(self, 'location'):
            self._foot_positions[i] = (
                self.location[0] + fx, self.location[1] + fy, not swing)

    def _draw_body(self):
        s = 0.8 + 0.4 * (self.leg_length - Config.LEG_LENGTH_MIN) / (
            Config.LEG_LENGTH_MAX - Config.LEG_LENGTH_MIN)
        for colour, rx, ry, dy, n in [
            ([0.15, 0.55, 0.25, 0.95], int(10*s), int(7*s), -16, 20),
            ([0.12, 0.48, 0.22, 0.95], int(14*s), int(11*s),  0, 24),
            ([0.10, 0.40, 0.18, 0.90], int(11*s), int(9*s),  15, 20),
        ]:
            glColor4fv(colour); glLineWidth(1.5); glBegin(GL_LINE_LOOP)
            for a in range(n):
                ang = a / n * 2 * math.pi
                glVertex2d(rx * math.cos(ang), ry * math.sin(ang) + dy)
            glEnd()
        for side, idx in [(-6, 0), (6, 3)]:
            frac = self.cpg.phase_fractions[idx]
            glColor4fv([0.2 + 0.8 * frac, 0.8 - 0.6 * frac, 0.5, 1.0])
            glPointSize(6.0); glBegin(GL_POINTS); glVertex2d(side, -18); glEnd()
        glColor4fv([0.5, 0.3, 0.8, 0.6]); glLineWidth(1.0)
        for side in (-1, 1):
            glBegin(GL_LINE_STRIP)
            glVertex2d(side*6,-18); glVertex2d(side*12,-28); glVertex2d(side*9,-38)
            glEnd()

    def _draw_force_arrows(self):
        for fp in self._foot_positions:
            if fp is None: continue
            fx, fy, contact = fp
            if not contact: continue
            lx = fx - self.location[0]; ly = fy - self.location[1]
            glColor4fv([0.0, 0.9, 0.5, 0.6]); glLineWidth(1.5)
            glBegin(GL_LINES); glVertex2d(lx, ly); glVertex2d(lx, ly - 6); glEnd()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GENETIC ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

class CPGGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generation_log = []

    def generate(self):
        super().generate()
        gen_idx = self.generations
        avg  = self._average_fitness_record[-1] if self._average_fitness_record else 0
        best = self._best_fitness_record[-1]    if self._best_fitness_record    else 0
        self._generation_log.append({"generation": gen_idx, "avg": avg, "best": best})
        print(f"[GA] Gen {gen_idx:3d}  avg={avg:.4f}  best={best:.4f}")

    def save_best(self, path):
        if self._best_ever_genome is None: return
        data = {
            "best_fitness": self._best_ever_fitness,
            "genome": self._best_ever_genome.tolist(),
            "fitness_mode": Config.FITNESS_MODE,
            "genome_length": CPGHexapod.GENOME_LENGTH,
            "generations_run": self.generations,
            "history": self._generation_log,
        }
        with open(path, 'w') as f: json.dump(data, f, indent=2)
        print(f"[GA] Best genome saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class CPGHexapodSimulation(Simulation):
    def __init__(self):
        super().__init__("CPGHexapod_AllFeatures")
        self.generations = Config.GENERATIONS
        self.assessments = Config.ASSESSMENTS
        self.timesteps   = Config.TIMESTEPS

        mutator = NormalMutator(mu=0.0, sigma=Config.MUTATION_SIGMA)
        self._ga = CPGGeneticAlgorithm(
            crossover=Config.CROSSOVER_RATE, mutation=Config.MUTATION_RATE,
            elitism=Config.ELITISM, mutator=mutator)

        pop = Population(Config.POPULATION_SIZE, CPGHexapod, self._ga)
        self.add("hexapods", pop)
        self._build_food()

    def _build_food(self):
        rng = np.random.default_rng(7)
        w, h = WDP.width, WDP.height
        food_items = []
        for _ in range(Config.FOOD_COUNT):
            for _ in range(100):
                x = rng.uniform(0.08 * w, 0.92 * w)
                y = rng.uniform(0.08 * h, 0.92 * h)
                if abs(x - w / 2) > 60 or abs(y - h / 2) > 60: break
            loc = np.array([x, y], dtype=np.float32)
            food_items.append(FoodPellet(location=loc, calories=Config.FOOD_CALORIES))

        class _FoodGroup:
            def __init__(self, items): self.items = items; self.world = None
            def begin_assessment(self):
                for f in self.items: f.eaten = False; f.dead = False
            def end_assessment(self): pass
            def begin_generation(self): pass
            def end_generation(self): pass
            def begin_run(self): pass
            def end_run(self): pass
            def add_to_world(self):
                for f in self.items:
                    f.world = self.world; self.world.add_object(f)

        self.add("food", _FoodGroup(food_items))

    def begin_simulation(self):
        print("=" * 60)
        print("  CPG Hexapod – All 5 Bio-Inspired Features")
        print("  Features active:")
        print("    1. Hebbian/STDP synaptic plasticity")
        print("    2. Proprioceptive sensory feedback reflex")
        print("    3. Morphological computation (leg_length, stiffness)")
        print("    4. Foraging / energy-based fitness")
        print("    5. Neuromodulation (fatigue-driven frequency/amplitude)")
        print(f"  Genome length  : {CPGHexapod.GENOME_LENGTH}")
        print(f"  Population     : {Config.POPULATION_SIZE}")
        print(f"  Generations    : {Config.GENERATIONS}")
        print("=" * 60)
        super().begin_simulation()

    def end_simulation(self):
        super().end_simulation()
        if Config.SAVE_BEST_GENOME:
            self._ga.save_best(Config.GENOME_SAVE_PATH)
        print("=" * 60)
        print("  Evolution complete.")
        if self._ga._best_ever_genome is not None:
            g  = self._ga._best_ever_genome
            lo_l, hi_l = Config.LEG_LENGTH_MIN, Config.LEG_LENGTH_MAX
            lo_s, hi_s = Config.STIFFNESS_MIN,  Config.STIFFNESS_MAX
            print(f"  Best fitness    : {self._ga._best_ever_fitness:.4f}")
            print(f"  Best leg_length : {lo_l+(hi_l-lo_l)*g[10]:.3f}")
            print(f"  Best stiffness  : {lo_s+(hi_s-lo_s)*g[11]:.3f}")
            print(f"  Best genome     : {np.round(g, 4).tolist()}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_genome(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        data = json.load(f)
    genome = np.array(data['genome'], dtype=np.float64)
    print(f"[Loader] Loaded genome (fitness={data.get('best_fitness', '?'):.4f})")
    return genome


if __name__ == "__main__":
    sim = CPGHexapodSimulation()
    sim._run_simulation_no_render(parallel=False)
