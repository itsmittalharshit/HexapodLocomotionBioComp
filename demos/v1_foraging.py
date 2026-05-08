"""
v1_foraging.py  –  CPG Hexapod + Foraging Behaviour (Point 4)
==============================================================
Variant 1: Adds energy-based survival fitness via Food objects.

NEW vs original:
  - FoodPellet  : WorldObject that gives +calories on contact
  - CPGHexapod  : tracks energy_consumed vs calories_gathered;
                  dies (dead=True) when energy runs out
  - Fitness     : total calories gathered (survival-driven)
  - Config      : FOOD_COUNT, FOOD_CALORIES, AGENT_START_ENERGY
  - Simulation  : scatters FoodPellets; FITNESS_MODE forced to 'FORAGING'

All other behaviour (CPG, GA, rendering) identical to original.
"""

import math
import time
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np

from OpenGL.GL import (
    glBegin, glEnd, glVertex2d, glColor4fv, glLineWidth,
    GL_LINE_LOOP, GL_LINE_STRIP, GL_LINES, GL_QUADS,
    glPushMatrix, glPopMatrix, glTranslatef, glRotatef,
    glEnable, glDisable, GL_LINE_SMOOTH, GL_BLEND,
    glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    glPointSize, GL_POINTS, glVertex2f
)
from OpenGL.GLU import gluNewQuadric, gluDisk, gluDeleteQuadric, GLU_FILL

from core.agent.agent import Agent
from core.evolve.evolver import Evolver
from core.evolve.base import NormalMutator
from core.evolve.genetic_algorithm import GeneticAlgorithm
from core.evolve.population import Population
from core.simulation import Simulation
from core.utils import (
    Vec2, AgentSettings as AS,
    ColourPalette, ColourType as CT,
    length_angle_to_vector, normalise_vector,
    WORLD_DISPLAY_PARAMETERS as WDP
)
from core.world.drawable import Drawable
from core.world.world_object import WorldObject

IS_DEMO    = True
DEMO_NAME  = "CPG Hexapod – Foraging"
CLASS_NAME = "CPGHexapodSimulation"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    POPULATION_SIZE   = 20
    GENERATIONS       = 100
    ASSESSMENTS       = 3
    TIMESTEPS         = 600

    CROSSOVER_RATE    = 0.70
    MUTATION_RATE     = 0.05
    ELITISM           = 2
    MUTATION_SIGMA    = 0.10

    # Foraging is the only fitness mode in this variant
    FITNESS_MODE      = 'FORAGING'

    DEFAULT_GAIT      = 'TRIPOD'
    ENVIRONMENT       = 'FLAT'

    CPG_FREQ          = 1.2
    CPG_COUPLING      = 0.65
    CPG_DUTY          = 0.60
    CPG_AMPLITUDE     = 0.80

    DRAW_CPG_NETWORK  = True
    DRAW_TRAILS       = True
    DRAW_LEG_PHASES   = True
    DRAW_FORCE_ARROWS = True
    SAVE_BEST_GENOME  = True
    GENOME_SAVE_PATH  = "best_hexapod_foraging.json"

    TIMESTEP_DT       = 0.05
    MAX_SPEED         = 120.0
    MIN_SPEED         = 0.0

    # ── Foraging parameters ───────────────────────────────────────────────
    FOOD_COUNT        = 12      # food pellets scattered in the world
    FOOD_CALORIES     = 25.0    # energy gained per pellet
    AGENT_START_ENERGY = 40.0   # initial energy budget
    ENERGY_PER_STEP   = 0.04    # energy drained per timestep of movement


GAIT_PHASES = {
    'TRIPOD': np.array([0, np.pi, 0, np.pi, 0, np.pi], dtype=np.float32),
    'WAVE':   np.array([0, np.pi/3, 2*np.pi/3,
                        np.pi, 4*np.pi/3, 5*np.pi/3], dtype=np.float32),
    'RIPPLE': np.array([0, 2*np.pi/3, 4*np.pi/3,
                        np.pi/3, np.pi, 5*np.pi/3], dtype=np.float32),
}

LEG_NAMES = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']


# ══════════════════════════════════════════════════════════════════════════════
# CPG NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class CPGNetwork:
    def __init__(self, freq, coupling, duty, amplitude, phase_offsets):
        self.freq      = float(freq)
        self.coupling  = float(coupling)
        self.duty      = float(duty)
        self.amplitude = float(amplitude)
        self.phi       = np.array(phase_offsets, dtype=np.float64)
        self.desired   = np.array(phase_offsets, dtype=np.float64)

    def step(self, dt):
        omega = 2.0 * np.pi * self.freq
        dphi  = np.zeros(6)
        for i in range(6):
            cs = 0.0
            for j in range(6):
                if i != j:
                    delta = self.desired[j] - self.desired[i]
                    cs += math.sin(self.phi[j] - self.phi[i] - delta)
            dphi[i] = omega + self.coupling * cs
        self.phi += dphi * dt

    def output(self, leg_idx):
        phi_norm = self.phi[leg_idx] % (2.0 * np.pi)
        x = math.sin(phi_norm) * self.amplitude
        is_swing = math.sin(phi_norm) > 0.0
        return x, is_swing

    @property
    def phase_fractions(self):
        return (self.phi % (2.0 * np.pi)) / (2.0 * np.pi)

    def coupling_strength(self, i, j):
        delta = self.desired[j] - self.desired[i]
        return math.sin(self.phi[j] - self.phi[i] - delta)


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
    """
    A food source. When a hexapod overlaps this, it gains calories.
    Becomes 'eaten' (dead=True) after collection so the simulation
    can remove it from the world on the next tick.
    """

    def __init__(self, location=None, calories=None):
        super().__init__(location=location, radius=10.0, solid=False)
        self.calories = calories if calories is not None else Config.FOOD_CALORIES
        self.eaten    = False
        self._pulse   = 0.0  # animation counter

    def draw(self):
        if self.eaten:
            return
        self._pulse = (self._pulse + 0.05) % (2 * math.pi)
        glow = 0.7 + 0.3 * math.sin(self._pulse)

        # Outer glow ring
        glColor4fv([0.9 * glow, 0.8 * glow, 0.1, 0.35])
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for a in range(20):
            ang = a / 20.0 * 2 * math.pi
            glVertex2d(14 * math.cos(ang), 14 * math.sin(ang))
        glEnd()

        # Filled disc
        glColor4fv([1.0 * glow, 0.85 * glow, 0.1, 0.85])
        q = gluNewQuadric()
        gluDisk(q, 0, self.radius, 20, 1)
        gluDeleteQuadric(q)

        # Star highlight
        glColor4fv([1.0, 1.0, 0.6, 0.9])
        glPointSize(4.0)
        glBegin(GL_POINTS)
        glVertex2d(0, 0)
        glEnd()

    def __del__(self): pass


# ══════════════════════════════════════════════════════════════════════════════
# HEXAPOD AGENT
# ══════════════════════════════════════════════════════════════════════════════

class CPGHexapod(Agent, Evolver):
    GENOME_LENGTH = 11
    GENE_SCALE = [
        (0.3, 3.0),
        (0.0, 1.0),
        (0.3, 0.85),
        (0.1, 1.0),
    ] + [(0.0, 2 * np.pi)] * 6

    def __init__(self):
        Agent.__init__(
            self,
            min_speed=Config.MIN_SPEED,
            max_speed=Config.MAX_SPEED,
            timestep=Config.TIMESTEP_DT,
            random_colour=False,
            solid=False,
        )
        Evolver.__init__(self)

        self.radius = 18.0
        self.colour = ColourPalette[CT.GREEN]

        default_phases = GAIT_PHASES.get(Config.DEFAULT_GAIT,
                                         GAIT_PHASES['TRIPOD']).copy()
        self.cpg = CPGNetwork(
            freq=Config.CPG_FREQ,
            coupling=Config.CPG_COUPLING,
            duty=Config.CPG_DUTY,
            amplitude=Config.CPG_AMPLITUDE,
            phase_offsets=default_phases,
        )

        self._start_location     = None
        self._distance_travelled = 0.0
        self._energy_consumed    = 0.0   # energy spent moving
        self._calories_gathered  = 0.0   # energy gained from food
        self._energy_remaining   = Config.AGENT_START_ENERGY
        self._coordination_score = 0.0
        self._step_count         = 0
        self._collision_penalty  = 0.0
        self._foods_eaten        = 0

        self._trail         = []
        self._leg_states    = [False] * 6
        self._foot_positions = [None] * 6
        self._slope_angle   = 0.0

    # ── Genome ────────────────────────────────────────────────────────────

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
        lo0, hi0 = self.GENE_SCALE[0]; g[0] = (self.cpg.freq     - lo0) / (hi0 - lo0)
        lo1, hi1 = self.GENE_SCALE[1]; g[1] = (self.cpg.coupling - lo1) / (hi1 - lo1)
        lo2, hi2 = self.GENE_SCALE[2]; g[2] = (self.cpg.duty     - lo2) / (hi2 - lo2)
        lo3, hi3 = self.GENE_SCALE[3]; g[3] = (self.cpg.amplitude- lo3) / (hi3 - lo3)
        for i in range(6):
            lo, hi = self.GENE_SCALE[4+i]
            g[4+i] = (self.cpg.desired[i] - lo) / (hi - lo)
        return g

    # ── Fitness ───────────────────────────────────────────────────────────

    def get_fitness(self):
        """
        Foraging fitness:
          - Primary: total calories gathered (incentivises finding food)
          - Secondary: small distance bonus (incentivises exploration)
          - Penalty: collision count
        """
        if self._step_count == 0:
            return 0.0
        base    = self._calories_gathered
        explore = self._distance_travelled * 0.01
        penalty = self._collision_penalty * 5.0
        return max(0.0, base + explore - penalty)

    # ── Reset ─────────────────────────────────────────────────────────────

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
        super().reset()

    # ── Collisions ────────────────────────────────────────────────────────

    def on_collision(self, other):
        if isinstance(other, Obstacle):
            self._collision_penalty += 1.0
        elif isinstance(other, FoodPellet) and not other.eaten:
            # Consume food: gain calories, mark pellet as eaten
            other.eaten               = True
            other.dead                = True
            self._calories_gathered  += other.calories
            self._energy_remaining   += other.calories
            self._foods_eaten        += 1

    # ── Control ───────────────────────────────────────────────────────────

    def control(self):
        dt = self._timestep

        # If out of energy → stop (agent is effectively dead)
        if self._energy_remaining <= 0:
            self.controls['left']  = 0.0
            self.controls['right'] = 0.0
            return

        self.cpg.step(dt)

        left_drive  = 0.0
        right_drive = 0.0
        energy_tick = 0.0

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

        left_drive  = max(-1.0, min(1.0, left_drive))
        right_drive = max(-1.0, min(1.0, right_drive))

        self.controls['left']  = left_drive
        self.controls['right'] = right_drive

        # Drain energy
        movement_cost = energy_tick * Config.ENERGY_PER_STEP * dt
        self._energy_consumed   += movement_cost
        self._energy_remaining  -= movement_cost
        self._step_count        += 1

        spd = math.hypot(*self.velocity) if hasattr(self, 'velocity') else 0.0
        self._distance_travelled += spd * dt

        fracs = self.cpg.phase_fractions
        pair_diff = (abs(math.sin(self.cpg.phi[0] - self.cpg.phi[2])) +
                     abs(math.sin(self.cpg.phi[2] - self.cpg.phi[4])) +
                     abs(math.sin(self.cpg.phi[1] - self.cpg.phi[3])) +
                     abs(math.sin(self.cpg.phi[3] - self.cpg.phi[5])))
        self._coordination_score += (1.0 - pair_diff / 4.0)

        if Config.DRAW_TRAILS and hasattr(self, 'location'):
            self._trail.append(tuple(self.location))
            if len(self._trail) > 200:
                self._trail.pop(0)

    # ── Draw ──────────────────────────────────────────────────────────────

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
            self._draw_cpg_network()

        for i in range(6):
            self._draw_leg(i)
        self._draw_body()

        if Config.DRAW_FORCE_ARROWS:
            self._draw_force_arrows()

        # Energy bar above body
        self._draw_energy_bar()

        glDisable(GL_LINE_SMOOTH)

    def _draw_energy_bar(self):
        """Show a small energy/satiation bar above the hexapod."""
        ratio = max(0.0, min(1.0,
            self._energy_remaining / (Config.AGENT_START_ENERGY + self._calories_gathered + 0.01)))
        bar_w = 30.0
        bar_h = 4.0
        x0 = -bar_w / 2
        y0 = -28.0

        # Background
        glColor4fv([0.3, 0.0, 0.0, 0.7])
        glBegin(GL_QUADS)
        glVertex2d(x0,          y0)
        glVertex2d(x0 + bar_w,  y0)
        glVertex2d(x0 + bar_w,  y0 + bar_h)
        glVertex2d(x0,          y0 + bar_h)
        glEnd()

        # Filled portion
        r = 1.0 - ratio
        g_ = ratio
        glColor4fv([r, g_, 0.1, 0.9])
        glBegin(GL_QUADS)
        glVertex2d(x0,                 y0)
        glVertex2d(x0 + bar_w * ratio, y0)
        glVertex2d(x0 + bar_w * ratio, y0 + bar_h)
        glVertex2d(x0,                 y0 + bar_h)
        glEnd()

    def _leg_geometry(self, leg_idx):
        bw, bh = 14.0, 20.0
        positions = [
            (-bw, -bh * 0.6, -1),
            (-bw * 1.2, 0.0,  -1),
            (-bw, +bh * 0.6,  -1),
            (+bw, -bh * 0.6,  +1),
            (+bw * 1.2, 0.0,  +1),
            (+bw, +bh * 0.6,  +1),
        ]
        return positions[leg_idx]

    def _draw_leg(self, i):
        disp, swing = self.cpg.output(i)
        ax, ay, side = self._leg_geometry(i)
        coxa_len, femur_len, tibia_len = 10.0, 14.0, 10.0
        sweep = disp * 8.0 * side
        lift  = max(0.0, disp) * 8.0 if swing else 0.0
        cx = ax + side * coxa_len
        cy = ay
        fx = ax + side * (coxa_len + femur_len * 0.6 + sweep)
        fy = ay + femur_len * 0.5 + lift
        kx = ax + side * (coxa_len + femur_len * 0.5)
        ky = ay + femur_len * 0.3 + lift * 0.4

        glColor4fv([0.9, 0.45, 0.1, 0.9] if swing else [0.15, 0.80, 0.40, 0.9])
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        glVertex2d(ax, ay); glVertex2d(cx, cy)
        glVertex2d(kx, ky); glVertex2d(fx, fy)
        glEnd()
        glColor4fv([0.0, 1.0, 0.5, 1.0] if not swing else [1.0, 0.6, 0.1, 0.8])
        glPointSize(5.0)
        glBegin(GL_POINTS); glVertex2d(fx, fy); glEnd()
        if hasattr(self, 'location'):
            self._foot_positions[i] = (
                self.location[0] + fx, self.location[1] + fy, not swing)

    def _draw_body(self):
        glColor4fv([0.15, 0.55, 0.25, 0.95])
        glLineWidth(1.5)
        glBegin(GL_LINE_LOOP)
        for a in range(20):
            ang = a / 20.0 * 2 * math.pi
            glVertex2d(10 * math.cos(ang), 7 * math.sin(ang) - 16)
        glEnd()
        glColor4fv([0.12, 0.48, 0.22, 0.95])
        glBegin(GL_LINE_LOOP)
        for a in range(24):
            ang = a / 24.0 * 2 * math.pi
            glVertex2d(14 * math.cos(ang), 11 * math.sin(ang))
        glEnd()
        glColor4fv([0.10, 0.40, 0.18, 0.9])
        glBegin(GL_LINE_LOOP)
        for a in range(20):
            ang = a / 20.0 * 2 * math.pi
            glVertex2d(11 * math.cos(ang), 9 * math.sin(ang) + 15)
        glEnd()
        for side, i in [(-6, 0), (6, 3)]:
            frac = self.cpg.phase_fractions[i]
            r = 0.2 + 0.8 * frac
            g_ = 0.8 - 0.6 * frac
            glColor4fv([r, g_, 0.5, 1.0])
            glPointSize(6.0)
            glBegin(GL_POINTS); glVertex2d(side, -18); glEnd()
        glColor4fv([0.5, 0.3, 0.8, 0.6])
        glLineWidth(1.0)
        for side in (-1, 1):
            glBegin(GL_LINE_STRIP)
            glVertex2d(side * 6, -18); glVertex2d(side * 12, -28); glVertex2d(side * 9, -38)
            glEnd()
        if Config.DRAW_CPG_NETWORK:
            for i in range(6):
                frac = self.cpg.phase_fractions[i]
                _, swing = self.cpg.output(i)
                color = ([0.2, 1.0, 0.5, 0.5] if not swing else [1.0, 0.5, 0.1, 0.5])
                glColor4fv(color)
                glLineWidth(2.5)
                arc_start = (i / 6.0) * 2 * math.pi
                arc_end   = arc_start + frac * 2 * math.pi / 6
                glBegin(GL_LINE_STRIP)
                for step in range(12):
                    a = arc_start + (arc_end - arc_start) * step / 11
                    glVertex2d(22 * math.cos(a), 22 * math.sin(a))
                glEnd()

    def _draw_cpg_network(self):
        radius = 28.0
        glLineWidth(0.8)
        for i in range(6):
            ai = i / 6.0 * 2 * math.pi - math.pi / 2
            xi = radius * math.cos(ai)
            yi = radius * math.sin(ai)
            for j in range(i + 1, 6):
                aj = j / 6.0 * 2 * math.pi - math.pi / 2
                xj = radius * math.cos(aj)
                yj = radius * math.sin(aj)
                strength = abs(self.cpg.coupling_strength(i, j))
                alpha = 0.05 + strength * self.cpg.coupling * 0.35
                glColor4fv([0.5, 0.3, 0.9, alpha])
                glBegin(GL_LINES)
                glVertex2d(xi, yi); glVertex2d(xj, yj)
                glEnd()

    def _draw_force_arrows(self):
        for i, fp in enumerate(self._foot_positions):
            if fp is None: continue
            fx, fy, contact = fp
            if not contact: continue
            lx = fx - self.location[0]
            ly = fy - self.location[1]
            glColor4fv([0.0, 0.9, 0.5, 0.6])
            glLineWidth(1.5)
            glBegin(GL_LINES)
            glVertex2d(lx, ly); glVertex2d(lx, ly - 6)
            glEnd()


# ══════════════════════════════════════════════════════════════════════════════
# GENETIC ALGORITHM
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
        if self._best_ever_genome is None:
            print("[GA] No genome to save yet."); return
        data = {
            "best_fitness": self._best_ever_fitness,
            "genome": self._best_ever_genome.tolist(),
            "fitness_mode": Config.FITNESS_MODE,
            "generations_run": self.generations,
            "history": self._generation_log,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[GA] Best genome saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class CPGHexapodSimulation(Simulation):
    def __init__(self):
        super().__init__("CPGHexapod_Foraging")

        self.generations = Config.GENERATIONS
        self.assessments = Config.ASSESSMENTS
        self.timesteps   = Config.TIMESTEPS

        mutator = NormalMutator(mu=0.0, sigma=Config.MUTATION_SIGMA)
        self._ga = CPGGeneticAlgorithm(
            crossover=Config.CROSSOVER_RATE,
            mutation=Config.MUTATION_RATE,
            elitism=Config.ELITISM,
            mutator=mutator,
        )

        pop = Population(Config.POPULATION_SIZE, CPGHexapod, self._ga)
        self.add("hexapods", pop)
        self._build_food()

    # ── Food placement ────────────────────────────────────────────────────

    def _build_food(self):
        """Scatter FoodPellets around the world, avoiding spawn centre."""
        rng = np.random.default_rng(7)
        w, h = WDP.width, WDP.height
        food_items = []
        for _ in range(Config.FOOD_COUNT):
            for _attempt in range(100):
                x = rng.uniform(0.08 * w, 0.92 * w)
                y = rng.uniform(0.08 * h, 0.92 * h)
                if abs(x - w / 2) > 60 or abs(y - h / 2) > 60:
                    break
            loc = np.array([x, y], dtype=np.float32)
            food_items.append(FoodPellet(location=loc,
                                         calories=Config.FOOD_CALORIES))

        class _FoodGroup:
            def __init__(self, items):
                self.items = items
                self.world = None
            def begin_assessment(self):
                # Respawn eaten pellets
                for f in self.items:
                    f.eaten = False
                    f.dead  = False
            def end_assessment(self): pass
            def begin_generation(self): pass
            def end_generation(self): pass
            def begin_run(self): pass
            def end_run(self): pass
            def add_to_world(self):
                for f in self.items:
                    f.world = self.world
                    self.world.add_object(f)

        self.add("food", _FoodGroup(food_items))

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def begin_simulation(self):
        print("=" * 60)
        print("  CPG Hexapod – Foraging Variant")
        print(f"  Food count     : {Config.FOOD_COUNT}")
        print(f"  Start energy   : {Config.AGENT_START_ENERGY}")
        print(f"  Calories/pellet: {Config.FOOD_CALORIES}")
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
            print(f"  Best fitness : {self._ga._best_ever_fitness:.4f}")
            g = self._ga._best_ever_genome
            print(f"  Best genome  : {np.round(g, 4).tolist()}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_genome(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        data = json.load(f)
    genome = np.array(data['genome'], dtype=np.float64)
    print(f"[Loader] Loaded genome from '{path}'  "
          f"(fitness={data.get('best_fitness', '?'):.4f})")
    return genome


if __name__ == "__main__":
    sim = CPGHexapodSimulation()
    sim._run_simulation_no_render(parallel=False)
