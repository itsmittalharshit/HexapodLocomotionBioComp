"""
v2_hebbian.py  –  CPG Hexapod + Foraging + Synaptic Plasticity (Point 1 + 4)
=============================================================================
Variant 2: Adds Hebbian / STDP-inspired learning to the Foraging base.

NEW vs v1_foraging:
  - CPGNetwork.coupling_weights : 6×6 matrix of dynamic coupling strengths
  - CPGNetwork.step()           : Hebbian update — pairs that fire in-phase
                                  strengthen, out-of-phase weaken
  - Config.HEBBIAN_*            : learning rate, decay, min/max weight bounds
  - CPGHexapod                  : coupling_weights reset between assessments
                                  but persist within a lifetime (plasticity)

Biological rationale
  "Cells that fire together, wire together."
  Oscillator pairs that maintain their desired phase relationship
  reinforce each other; chaotic pairs weaken. The robot can refine
  its inter-leg coordination during each lifetime trial.
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
DEMO_NAME  = "CPG Hexapod – Hebbian + Foraging"
CLASS_NAME = "CPGHexapodSimulation"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    POPULATION_SIZE  = 20
    GENERATIONS      = 100
    ASSESSMENTS      = 3
    TIMESTEPS        = 600

    CROSSOVER_RATE   = 0.70
    MUTATION_RATE    = 0.05
    ELITISM          = 2
    MUTATION_SIGMA   = 0.10

    FITNESS_MODE     = 'FORAGING'
    DEFAULT_GAIT     = 'TRIPOD'
    ENVIRONMENT      = 'FLAT'

    CPG_FREQ         = 1.2
    CPG_COUPLING     = 0.65
    CPG_DUTY         = 0.60
    CPG_AMPLITUDE    = 0.80

    DRAW_CPG_NETWORK  = True
    DRAW_TRAILS       = True
    DRAW_FORCE_ARROWS = True
    SAVE_BEST_GENOME  = True
    GENOME_SAVE_PATH  = "best_hexapod_hebbian.json"

    TIMESTEP_DT      = 0.05
    MAX_SPEED        = 120.0
    MIN_SPEED        = 0.0

    # Foraging
    FOOD_COUNT        = 12
    FOOD_CALORIES     = 25.0
    AGENT_START_ENERGY = 40.0
    ENERGY_PER_STEP   = 0.04

    # ── Hebbian / STDP parameters ─────────────────────────────────────────
    HEBBIAN_LR        = 0.002   # learning rate η
    HEBBIAN_DECAY     = 0.0001  # passive weight decay (forgetting)
    HEBBIAN_W_MIN     = 0.05    # minimum coupling weight
    HEBBIAN_W_MAX     = 2.0     # maximum coupling weight
    # Plasticity threshold: |phase_error| < this → Hebbian potentiation
    HEBBIAN_SYNC_THR  = 0.25    # radians


GAIT_PHASES = {
    'TRIPOD': np.array([0, np.pi, 0, np.pi, 0, np.pi], dtype=np.float32),
    'WAVE':   np.array([0, np.pi/3, 2*np.pi/3,
                        np.pi, 4*np.pi/3, 5*np.pi/3], dtype=np.float32),
    'RIPPLE': np.array([0, 2*np.pi/3, 4*np.pi/3,
                        np.pi/3, np.pi, 5*np.pi/3], dtype=np.float32),
}

LEG_NAMES = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']


# ══════════════════════════════════════════════════════════════════════════════
# CPG NETWORK  (with Hebbian plasticity)
# ══════════════════════════════════════════════════════════════════════════════

class CPGNetwork:
    """
    Kuramoto network with dynamic coupling weights.

    Weight update rule (Hebbian / STDP-inspired):
      If |φⱼ − φᵢ − Δᵢⱼ| < threshold  →  potentiate  (cells fire together)
      Else                              →  depress     (chaotic pair)
      Plus a passive decay term so unused connections shrink.

    dWᵢⱼ/dt = η * [sync_signal(i,j) − δ * Wᵢⱼ]
    where sync_signal = +1 if in-phase, −1 if anti-phase.
    """

    def __init__(self, freq, coupling, duty, amplitude, phase_offsets):
        self.freq      = float(freq)
        self.base_coupling = float(coupling)   # genomic baseline
        self.duty      = float(duty)
        self.amplitude = float(amplitude)
        self.phi       = np.array(phase_offsets, dtype=np.float64)
        self.desired   = np.array(phase_offsets, dtype=np.float64)

        # Initialise weight matrix to genomic coupling
        self.coupling_weights = np.full((6, 6), coupling, dtype=np.float64)
        np.fill_diagonal(self.coupling_weights, 0.0)

    def reset_weights(self):
        """Call between assessments to start each lifetime from baseline."""
        self.coupling_weights[:] = self.base_coupling
        np.fill_diagonal(self.coupling_weights, 0.0)

    def step(self, dt):
        omega = 2.0 * np.pi * self.freq
        dphi  = np.zeros(6)

        for i in range(6):
            cs = 0.0
            for j in range(6):
                if i == j:
                    continue
                delta = self.desired[j] - self.desired[i]
                cs   += self.coupling_weights[i, j] * math.sin(
                    self.phi[j] - self.phi[i] - delta)
            dphi[i] = omega + cs

        self.phi += dphi * dt

        # Hebbian weight update
        self._update_weights(dt)

    def _update_weights(self, dt):
        lr    = Config.HEBBIAN_LR
        decay = Config.HEBBIAN_DECAY
        thr   = Config.HEBBIAN_SYNC_THR
        wmin  = Config.HEBBIAN_W_MIN
        wmax  = Config.HEBBIAN_W_MAX

        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                delta = self.desired[j] - self.desired[i]
                phase_err = abs((self.phi[j] - self.phi[i] - delta)
                                % (2 * math.pi))
                # Normalise to [0, π]
                if phase_err > math.pi:
                    phase_err = 2 * math.pi - phase_err

                # In-phase → potentiate; out-of-phase → depress
                sync = 1.0 if phase_err < thr else -0.5

                dW = lr * (sync - decay * self.coupling_weights[i, j])
                self.coupling_weights[i, j] = max(wmin,
                    min(wmax, self.coupling_weights[i, j] + dW * dt))

    def output(self, leg_idx):
        phi_norm = self.phi[leg_idx] % (2.0 * np.pi)
        x = math.sin(phi_norm) * self.amplitude
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
# WORLD OBJECTS
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
    def __init__(self, location=None, calories=None):
        super().__init__(location=location, radius=10.0, solid=False)
        self.calories = calories if calories is not None else Config.FOOD_CALORIES
        self.eaten    = False
        self._pulse   = 0.0

    def draw(self):
        if self.eaten: return
        self._pulse = (self._pulse + 0.05) % (2 * math.pi)
        glow = 0.7 + 0.3 * math.sin(self._pulse)
        glColor4fv([0.9 * glow, 0.8 * glow, 0.1, 0.35])
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for a in range(20):
            ang = a / 20.0 * 2 * math.pi
            glVertex2d(14 * math.cos(ang), 14 * math.sin(ang))
        glEnd()
        glColor4fv([1.0 * glow, 0.85 * glow, 0.1, 0.85])
        q = gluNewQuadric()
        gluDisk(q, 0, self.radius, 20, 1)
        gluDeleteQuadric(q)
        glColor4fv([1.0, 1.0, 0.6, 0.9])
        glPointSize(4.0)
        glBegin(GL_POINTS); glVertex2d(0, 0); glEnd()

    def __del__(self): pass


# ══════════════════════════════════════════════════════════════════════════════
# HEXAPOD AGENT
# ══════════════════════════════════════════════════════════════════════════════

class CPGHexapod(Agent, Evolver):
    GENOME_LENGTH = 11
    GENE_SCALE = [
        (0.3, 3.0), (0.0, 1.0), (0.3, 0.85), (0.1, 1.0),
    ] + [(0.0, 2 * np.pi)] * 6

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

    def _scale_gene(self, raw, idx):
        lo, hi = self.GENE_SCALE[idx]
        return lo + (hi - lo) * max(0.0, min(1.0, raw))

    def set_genotype(self, genome):
        assert len(genome) == self.GENOME_LENGTH
        g = np.asarray(genome, dtype=np.float64)
        freq     = self._scale_gene(g[0], 0)
        coupling = self._scale_gene(g[1], 1)
        duty     = self._scale_gene(g[2], 2)
        amp      = self._scale_gene(g[3], 3)
        phases   = np.array([self._scale_gene(g[4+i], 4+i) for i in range(6)],
                            dtype=np.float64)
        self.cpg = CPGNetwork(freq, coupling, duty, amp, phases)

    def get_genotype(self):
        g = np.zeros(self.GENOME_LENGTH, dtype=np.float64)
        lo0, hi0 = self.GENE_SCALE[0]; g[0] = (self.cpg.freq           - lo0) / (hi0 - lo0)
        lo1, hi1 = self.GENE_SCALE[1]; g[1] = (self.cpg.base_coupling  - lo1) / (hi1 - lo1)
        lo2, hi2 = self.GENE_SCALE[2]; g[2] = (self.cpg.duty           - lo2) / (hi2 - lo2)
        lo3, hi3 = self.GENE_SCALE[3]; g[3] = (self.cpg.amplitude      - lo3) / (hi3 - lo3)
        for i in range(6):
            lo, hi = self.GENE_SCALE[4+i]
            g[4+i] = (self.cpg.desired[i] - lo) / (hi - lo)
        return g

    def get_fitness(self):
        if self._step_count == 0: return 0.0
        base    = self._calories_gathered
        explore = self._distance_travelled * 0.01
        penalty = self._collision_penalty * 5.0
        # Small bonus for high average coupling weight (integrated plasticity)
        plasticity_bonus = self.cpg.avg_weight * 0.5
        return max(0.0, base + explore - penalty + plasticity_bonus)

    def reset(self):
        self._distance_travelled = 0.0
        self._energy_consumed    = 0.0
        self._calories_gathered  = 0.0
        self._energy_remaining   = Config.AGENT_START_ENERGY
        self._coordination_score = 0.0
        self._step_count         = 0
        self._collision_penalty  = 0.0
        self._foods_eaten        = 0
        self._trail.clear()
        # Reset Hebbian weights to genomic baseline each assessment
        self.cpg.reset_weights()
        super().reset()

    def on_collision(self, other):
        if isinstance(other, Obstacle):
            self._collision_penalty += 1.0
        elif isinstance(other, FoodPellet) and not other.eaten:
            other.eaten              = True
            other.dead               = True
            self._calories_gathered += other.calories
            self._energy_remaining  += other.calories
            self._foods_eaten       += 1

    def control(self):
        dt = self._timestep
        if self._energy_remaining <= 0:
            self.controls['left'] = self.controls['right'] = 0.0
            return

        self.cpg.step(dt)

        left_drive = right_drive = energy_tick = 0.0
        for i, name in enumerate(LEG_NAMES):
            disp, swing = self.cpg.output(i)
            self._leg_states[i] = swing
            is_left = name.startswith('L')
            if not swing:
                drive = self.cpg.duty * disp
                if is_left: left_drive  += drive / 3.0
                else:       right_drive += drive / 3.0
                energy_tick += abs(drive)
            else:
                energy_tick += 0.02

        self.controls['left']  = max(-1.0, min(1.0, left_drive))
        self.controls['right'] = max(-1.0, min(1.0, right_drive))

        movement_cost = energy_tick * Config.ENERGY_PER_STEP * dt
        self._energy_consumed  += movement_cost
        self._energy_remaining -= movement_cost
        self._step_count       += 1

        spd = math.hypot(*self.velocity) if hasattr(self, 'velocity') else 0.0
        self._distance_travelled += spd * dt

        pair_diff = (abs(math.sin(self.cpg.phi[0] - self.cpg.phi[2])) +
                     abs(math.sin(self.cpg.phi[2] - self.cpg.phi[4])) +
                     abs(math.sin(self.cpg.phi[1] - self.cpg.phi[3])) +
                     abs(math.sin(self.cpg.phi[3] - self.cpg.phi[5])))
        self._coordination_score += (1.0 - pair_diff / 4.0)

        if Config.DRAW_TRAILS and hasattr(self, 'location'):
            self._trail.append(tuple(self.location))
            if len(self._trail) > 200:
                self._trail.pop(0)

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

        if Config.DRAW_CPG_NETWORK:
            self._draw_cpg_network_hebbian()

        for i in range(6):
            self._draw_leg(i)
        self._draw_body()

        if Config.DRAW_FORCE_ARROWS:
            self._draw_force_arrows()
        self._draw_energy_bar()
        glDisable(GL_LINE_SMOOTH)

    def _draw_cpg_network_hebbian(self):
        """Draw coupling lines; thickness encodes the learned weight."""
        radius = 28.0
        for i in range(6):
            ai = i / 6.0 * 2 * math.pi - math.pi / 2
            xi = radius * math.cos(ai)
            yi = radius * math.sin(ai)
            for j in range(i + 1, 6):
                aj = j / 6.0 * 2 * math.pi - math.pi / 2
                xj = radius * math.cos(aj)
                yj = radius * math.sin(aj)
                w = self.cpg.coupling_weights[i, j]
                norm_w = (w - Config.HEBBIAN_W_MIN) / (
                    Config.HEBBIAN_W_MAX - Config.HEBBIAN_W_MIN + 1e-9)
                alpha = 0.05 + norm_w * 0.6
                # Colour: blue=weak, gold=strong
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
        glVertex2d(x0+bar_w, y0+bar_h); glVertex2d(x0, y0+bar_h)
        glEnd()
        glColor4fv([1.0-ratio, ratio, 0.1, 0.9])
        glBegin(GL_QUADS)
        glVertex2d(x0, y0); glVertex2d(x0+bar_w*ratio, y0)
        glVertex2d(x0+bar_w*ratio, y0+bar_h); glVertex2d(x0, y0+bar_h)
        glEnd()

    def _leg_geometry(self, leg_idx):
        bw, bh = 14.0, 20.0
        return [(-bw,-bh*.6,-1),(-bw*1.2,0.,-1),(-bw,+bh*.6,-1),
                (+bw,-bh*.6,+1),(+bw*1.2,0.,+1),(+bw,+bh*.6,+1)][leg_idx]

    def _draw_leg(self, i):
        disp, swing = self.cpg.output(i)
        ax, ay, side = self._leg_geometry(i)
        sweep = disp * 8.0 * side
        lift  = max(0.0, disp) * 8.0 if swing else 0.0
        cx = ax + side * 10.0; cy = ay
        kx = ax + side * (10.0 + 14.0 * 0.5); ky = ay + 14.0 * 0.3 + lift * 0.4
        fx = ax + side * (10.0 + 14.0 * 0.6 + sweep); fy = ay + 14.0 * 0.5 + lift
        glColor4fv([0.9,0.45,0.1,0.9] if swing else [0.15,0.80,0.40,0.9])
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        glVertex2d(ax,ay); glVertex2d(cx,cy); glVertex2d(kx,ky); glVertex2d(fx,fy)
        glEnd()
        glColor4fv([0.0,1.0,0.5,1.0] if not swing else [1.0,0.6,0.1,0.8])
        glPointSize(5.0); glBegin(GL_POINTS); glVertex2d(fx,fy); glEnd()
        if hasattr(self, 'location'):
            self._foot_positions[i] = (self.location[0]+fx, self.location[1]+fy, not swing)

    def _draw_body(self):
        for colour, rx, ry, dy, n in [
            ([0.15,0.55,0.25,0.95], 10,  7, -16, 20),
            ([0.12,0.48,0.22,0.95], 14, 11,   0, 24),
            ([0.10,0.40,0.18,0.90], 11,  9,  15, 20),
        ]:
            glColor4fv(colour); glLineWidth(1.5); glBegin(GL_LINE_LOOP)
            for a in range(n):
                ang = a / n * 2 * math.pi
                glVertex2d(rx * math.cos(ang), ry * math.sin(ang) + dy)
            glEnd()
        for side, i in [(-6,0),(6,3)]:
            frac = self.cpg.phase_fractions[i]
            glColor4fv([0.2+0.8*frac, 0.8-0.6*frac, 0.5, 1.0])
            glPointSize(6.0); glBegin(GL_POINTS); glVertex2d(side,-18); glEnd()

    def _draw_force_arrows(self):
        for fp in self._foot_positions:
            if fp is None: continue
            fx, fy, contact = fp
            if not contact: continue
            lx = fx - self.location[0]; ly = fy - self.location[1]
            glColor4fv([0.0,0.9,0.5,0.6]); glLineWidth(1.5)
            glBegin(GL_LINES); glVertex2d(lx,ly); glVertex2d(lx,ly-6); glEnd()


# ══════════════════════════════════════════════════════════════════════════════
# GA + SIMULATION
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
        data = {"best_fitness": self._best_ever_fitness,
                "genome": self._best_ever_genome.tolist(),
                "fitness_mode": Config.FITNESS_MODE,
                "generations_run": self.generations,
                "history": self._generation_log}
        with open(path, 'w') as f: json.dump(data, f, indent=2)
        print(f"[GA] Best genome saved → {path}")


class CPGHexapodSimulation(Simulation):
    def __init__(self):
        super().__init__("CPGHexapod_Hebbian")
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
                x = rng.uniform(0.08*w, 0.92*w)
                y = rng.uniform(0.08*h, 0.92*h)
                if abs(x - w/2) > 60 or abs(y - h/2) > 60: break
            loc = np.array([x, y], dtype=np.float32)
            food_items.append(FoodPellet(location=loc, calories=Config.FOOD_CALORIES))

        class _FoodGroup:
            def __init__(self, items): self.items=items; self.world=None
            def begin_assessment(self):
                for f in self.items: f.eaten=False; f.dead=False
            def end_assessment(self): pass
            def begin_generation(self): pass
            def end_generation(self): pass
            def begin_run(self): pass
            def end_run(self): pass
            def add_to_world(self):
                for f in self.items: f.world=self.world; self.world.add_object(f)

        self.add("food", _FoodGroup(food_items))

    def begin_simulation(self):
        print("=" * 60)
        print("  CPG Hexapod – Hebbian Plasticity + Foraging")
        print(f"  Hebbian LR   : {Config.HEBBIAN_LR}")
        print(f"  Weight range : [{Config.HEBBIAN_W_MIN}, {Config.HEBBIAN_W_MAX}]")
        print(f"  Food count   : {Config.FOOD_COUNT}")
        print("=" * 60)
        super().begin_simulation()

    def end_simulation(self):
        super().end_simulation()
        if Config.SAVE_BEST_GENOME: self._ga.save_best(Config.GENOME_SAVE_PATH)
        print("=" * 60)
        print("  Evolution complete.")
        if self._ga._best_ever_genome is not None:
            print(f"  Best fitness : {self._ga._best_ever_fitness:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    sim = CPGHexapodSimulation()
    sim._run_simulation_no_render(parallel=False)
