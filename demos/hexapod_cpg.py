"""
hexapod_cpg.py  –  CPG Hexapod Locomotion Demo for PyBeast++
=============================================================
Place this file in the pybeastpp/demos/ folder.

Biology → Bytes: Central Pattern Generator (CPG) oscillators
are coupled via a Kuramoto-style network and evolved with a
Genetic Algorithm to produce stable, efficient hexapod gaits.

The genome encodes:
  [freq, coupling, duty_factor, amplitude,
   phase_0 … phase_5]   (11 genes total)

Fitness functions (selectable):
  DISTANCE   – distance travelled / time  (default)
  EFFICIENCY – distance / energy consumed
  STABILITY  – regularity of leg coordination

Gaits available for manual inspection:
  TRIPOD, WAVE, RIPPLE, EVOLVED

Environment presets:
  FLAT, ROUGH, SLOPE

How to use
----------
1.  Run via PyBeast++ GUI:  Demo → CPG Hexapod
2.  Run headless from the repo root:
        python -c "
        import sys; sys.path.insert(0, '.')
        from demos.hexapod_cpg import CPGHexapodSimulation
        sim = CPGHexapodSimulation()
        sim.run_simulation(render=False)
        "

All log output goes to stdout.  Fitness per generation is
printed so you can pipe it to a file for offline plotting.
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import math
import time
import json
import os
from copy import deepcopy
from pathlib import Path

# ─── numpy ───────────────────────────────────────────────────────────────────
import numpy as np

# ─── OpenGL (only used in draw()) ────────────────────────────────────────────
from OpenGL.GL import (
    glBegin, glEnd, glVertex2d, glColor4fv, glLineWidth,
    GL_LINE_LOOP, GL_LINE_STRIP, GL_LINES, GL_QUADS,
    glPushMatrix, glPopMatrix, glTranslatef, glRotatef,
    glEnable, glDisable, GL_LINE_SMOOTH, GL_BLEND,
    glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    glPointSize, GL_POINTS, glVertex2f
)
from OpenGL.GLU import gluNewQuadric, gluDisk, gluDeleteQuadric, GLU_FILL

# ─── PyBeast++ core ──────────────────────────────────────────────────────────
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

# ─── GUI registration ────────────────────────────────────────────────────────
IS_DEMO    = True
DEMO_NAME  = "CPG Hexapod"
CLASS_NAME = "CPGHexapodSimulation"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION  (edit these to change behaviour)
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Simulation structure ──────────────────────────────────────────────
    POPULATION_SIZE   = 20       # agents per generation
    GENERATIONS       = 100       # GA generations
    ASSESSMENTS       = 3        # evaluations per individual (averaged)
    TIMESTEPS         = 600      # steps per assessment

    # ── GA hyper-parameters ───────────────────────────────────────────────
    CROSSOVER_RATE    = 0.70
    MUTATION_RATE     = 0.05
    ELITISM           = 2        # top-N preserved unchanged
    MUTATION_SIGMA    = 0.10     # std-dev of Gaussian mutation

    # ── Fitness mode: 'DISTANCE' | 'EFFICIENCY' | 'STABILITY' ────────────
    FITNESS_MODE      = 'DISTANCE'

    # ── Gait preset for manual mode (overridden by GA in evolved mode) ────
    # Options: 'TRIPOD' | 'WAVE' | 'RIPPLE'
    DEFAULT_GAIT      = 'TRIPOD'

    # ── Environment preset ────────────────────────────────────────────────
    # Options: 'FLAT' | 'ROUGH' | 'SLOPE'
    ENVIRONMENT       = 'FLAT'

    # ── CPG defaults (used when no genome is provided) ────────────────────
    CPG_FREQ          = 1.2      # Hz
    CPG_COUPLING      = 0.65     # κ  coupling strength
    CPG_DUTY          = 0.60     # stance duty factor
    CPG_AMPLITUDE     = 0.80     # oscillator amplitude

    # ── Visualisation ─────────────────────────────────────────────────────
    DRAW_CPG_NETWORK  = True     # show inter-oscillator coupling lines
    DRAW_TRAILS       = True     # show body trail
    DRAW_LEG_PHASES   = True     # colour legs by phase
    DRAW_FORCE_ARROWS = True     # show ground-reaction force arrows
    SAVE_BEST_GENOME  = True     # write best genome to JSON on completion
    GENOME_SAVE_PATH  = "best_hexapod_genome.json"

    # ── Physics ───────────────────────────────────────────────────────────
    TIMESTEP_DT       = 0.05     # seconds per simulation step
    MAX_SPEED         = 120.0
    MIN_SPEED         = 0.0

# ══════════════════════════════════════════════════════════════════════════════
# 2.  GAIT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# Phase offsets for each of 6 legs (L1,L2,L3,R1,R2,R3).
# 0 = in-phase with reference oscillator, π = antiphase.
GAIT_PHASES = {
    # Two tripods alternate: (L1,R2,L3) and (R1,L2,R3)
    'TRIPOD': np.array([0, np.pi, 0, np.pi, 0, np.pi], dtype=np.float32),
    # One leg at a time, rear→front, left then right
    'WAVE':   np.array([0, np.pi/3, 2*np.pi/3,
                        np.pi, 4*np.pi/3, 5*np.pi/3], dtype=np.float32),
    # Intermediate: two overlapping metachronal waves
    'RIPPLE': np.array([0, 2*np.pi/3, 4*np.pi/3,
                        np.pi/3, np.pi, 5*np.pi/3], dtype=np.float32),
}

LEG_NAMES = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

# ══════════════════════════════════════════════════════════════════════════════
# 3.  CPG OSCILLATOR ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CPGNetwork:
    """
    Kuramoto-coupled oscillator network: 6 nodes, one per leg.

    Each oscillator i integrates:
        dφᵢ/dt = ω + κ Σⱼ sin(φⱼ − φᵢ − Δᵢⱼ)

    where Δᵢⱼ = desired phase difference from the gait pattern.
    The output drives stance/swing decision and foot displacement.
    """

    def __init__(self, freq: float, coupling: float,
                 duty: float, amplitude: float,
                 phase_offsets: np.ndarray):
        self.freq       = float(freq)          # Hz
        self.coupling   = float(coupling)      # κ
        self.duty       = float(duty)          # stance duty factor
        self.amplitude  = float(amplitude)     # peak oscillator output
        self.phi        = np.array(phase_offsets, dtype=np.float64)  # current phases
        self.desired    = np.array(phase_offsets, dtype=np.float64)  # target offsets

    def step(self, dt: float) -> None:
        omega = 2.0 * np.pi * self.freq
        dphi  = np.zeros(6)
        for i in range(6):
            coupling_sum = 0.0
            for j in range(6):
                if i != j:
                    delta_desired = self.desired[j] - self.desired[i]
                    coupling_sum += math.sin(
                        self.phi[j] - self.phi[i] - delta_desired
                    )
            dphi[i] = omega + self.coupling * coupling_sum
        self.phi += dphi * dt

    def output(self, leg_idx: int):
        """Return (normalised_displacement, is_swing)."""
        phi_norm = self.phi[leg_idx] % (2.0 * np.pi)
        x = math.sin(phi_norm) * self.amplitude
        # Swing when sin > 0 (first half of cycle), stance when sin ≤ 0
        is_swing = math.sin(phi_norm) > 0.0
        return x, is_swing

    @property
    def phase_fractions(self) -> np.ndarray:
        """Phase fraction [0,1) for each leg."""
        return (self.phi % (2.0 * np.pi)) / (2.0 * np.pi)

    def coupling_strength(self, i: int, j: int) -> float:
        """Instantaneous coupling influence i→j."""
        delta = self.desired[j] - self.desired[i]
        return math.sin(self.phi[j] - self.phi[i] - delta)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  OBSTACLE (for ROUGH environment)
# ══════════════════════════════════════════════════════════════════════════════

class Obstacle(WorldObject):
    """Static round pebble obstacle for the rough-terrain preset."""

    def __init__(self, location=None, radius: float = 12.0):
        super().__init__(location=location, radius=radius, solid=True)
        self.colour = ColourPalette[CT.DARK_GREY]

    def draw(self) -> None:
        glColor4fv([0.35, 0.33, 0.30, 0.9])
        q = gluNewQuadric()
        gluDisk(q, 0, self.radius, 16, 1)
        gluDeleteQuadric(q)

    def __del__(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HEXAPOD AGENT
# ══════════════════════════════════════════════════════════════════════════════

class CPGHexapod(Agent, Evolver):
    """
    Six-legged agent driven by a CPGNetwork.

    Genome layout (11 genes, all in [0,1] before scaling):
        gene 0  → frequency   [0.3 … 3.0] Hz
        gene 1  → coupling    [0.0 … 1.0]
        gene 2  → duty factor [0.30 … 0.85]
        gene 3  → amplitude   [0.10 … 1.0]
        genes 4…9 → phase offsets for legs 0…5, scaled to [0, 2π]

    Fitness is scored by Config.FITNESS_MODE.
    """

    GENOME_LENGTH = 11

    # Scaling bounds for each gene
    GENE_SCALE = [
        (0.3, 3.0),   # freq
        (0.0, 1.0),   # coupling
        (0.3, 0.85),  # duty
        (0.1, 1.0),   # amplitude
    ] + [(0.0, 2 * np.pi)] * 6  # 6 phase offsets

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

        # Initialise CPG with default gait
        default_phases = GAIT_PHASES.get(Config.DEFAULT_GAIT,
                                         GAIT_PHASES['TRIPOD']).copy()
        self.cpg = CPGNetwork(
            freq=Config.CPG_FREQ,
            coupling=Config.CPG_COUPLING,
            duty=Config.CPG_DUTY,
            amplitude=Config.CPG_AMPLITUDE,
            phase_offsets=default_phases,
        )

        # Metrics accumulated during an assessment
        self._start_location: Vec2 = None
        self._distance_travelled: float = 0.0
        self._energy_consumed: float = 0.0
        self._coordination_score: float = 0.0
        self._step_count: int = 0
        self._collision_penalty: float = 0.0

        # For visualisation
        self._trail: list = []           # body trail positions
        self._leg_states: list = [False] * 6  # True = swing
        self._foot_positions: list = [None] * 6

        # Environment modifier (set by simulation)
        self._slope_angle: float = 0.0   # radians, uphill direction

    # ── Genome interface (Evolver) ────────────────────────────────────────

    def _scale_gene(self, raw: float, idx: int) -> float:
        lo, hi = self.GENE_SCALE[idx]
        return lo + (hi - lo) * max(0.0, min(1.0, raw))

    def set_genotype(self, genome) -> None:
        assert len(genome) == self.GENOME_LENGTH, (
            f"Expected {self.GENOME_LENGTH} genes, got {len(genome)}"
        )
        g = np.asarray(genome, dtype=np.float64)
        freq     = self._scale_gene(g[0], 0)
        coupling = self._scale_gene(g[1], 1)
        duty     = self._scale_gene(g[2], 2)
        amp      = self._scale_gene(g[3], 3)
        phases   = np.array([self._scale_gene(g[4 + i], 4 + i)
                              for i in range(6)], dtype=np.float64)
        self.cpg = CPGNetwork(freq, coupling, duty, amp, phases)

    def get_genotype(self) -> np.ndarray:
        g = np.zeros(self.GENOME_LENGTH, dtype=np.float64)
        lo0, hi0 = self.GENE_SCALE[0]; g[0] = (self.cpg.freq - lo0) / (hi0 - lo0)
        lo1, hi1 = self.GENE_SCALE[1]; g[1] = (self.cpg.coupling - lo1) / (hi1 - lo1)
        lo2, hi2 = self.GENE_SCALE[2]; g[2] = (self.cpg.duty - lo2) / (hi2 - lo2)
        lo3, hi3 = self.GENE_SCALE[3]; g[3] = (self.cpg.amplitude - lo3) / (hi3 - lo3)
        for i in range(6):
            lo, hi = self.GENE_SCALE[4 + i]
            g[4 + i] = (self.cpg.desired[i] - lo) / (hi - lo)
        return g

    # ── Fitness ───────────────────────────────────────────────────────────

    def get_fitness(self) -> float:
        if self._step_count == 0:
            return 0.0
        penalty = self._collision_penalty
        if Config.FITNESS_MODE == 'DISTANCE':
            return max(0.0, self._distance_travelled - penalty * 20)
        elif Config.FITNESS_MODE == 'EFFICIENCY':
            if self._energy_consumed < 1e-6:
                return 0.0
            return max(0.0,
                (self._distance_travelled / self._energy_consumed) - penalty)
        elif Config.FITNESS_MODE == 'STABILITY':
            return max(0.0, self._coordination_score - penalty * 5)
        return self._distance_travelled

    # ── Reset between assessments ─────────────────────────────────────────

    def reset(self) -> None:
        self._distance_travelled  = 0.0
        self._energy_consumed     = 0.0
        self._coordination_score  = 0.0
        self._step_count          = 0
        self._collision_penalty   = 0.0
        self._trail.clear()
        super().reset()

    # ── on_collision ─────────────────────────────────────────────────────

    def on_collision(self, other: WorldObject) -> None:
        if isinstance(other, Obstacle):
            self._collision_penalty += 1.0

    # ── Control loop (called every timestep) ─────────────────────────────

    def control(self) -> None:
        dt = self._timestep
        self.cpg.step(dt)

        # Aggregate leg outputs into differential drive signals
        left_drive  = 0.0
        right_drive = 0.0
        energy_tick = 0.0
        coordination = 0.0

        for i, name in enumerate(LEG_NAMES):
            disp, swing = self.cpg.output(i)
            self._leg_states[i] = swing
            is_left = name.startswith('L')

            # Stance legs push, swing legs recover
            if not swing:
                drive = self.cpg.duty * disp
                if is_left:
                    left_drive  += drive / 3.0
                else:
                    right_drive += drive / 3.0
                energy_tick += abs(drive)
            else:
                energy_tick += 0.02  # swing costs some energy too

        # Apply slope resistance
        slope_drag = math.cos(self._slope_angle) if Config.ENVIRONMENT == 'SLOPE' else 1.0

        # Normalise and clamp to [-1, 1]
        left_drive  = max(-1.0, min(1.0, left_drive  * slope_drag))
        right_drive = max(-1.0, min(1.0, right_drive * slope_drag))

        # PyBeast++ agent controls
        self.controls['left']  = left_drive
        self.controls['right'] = right_drive

        # Accumulate metrics
        self._energy_consumed    += energy_tick * dt
        self._step_count         += 1

        # Distance accumulation (approx from speed)
        spd = math.hypot(*self.velocity) if hasattr(self, 'velocity') else 0.0
        self._distance_travelled += spd * dt

        # Coordination score: fraction of timesteps with expected tripod pairing
        fracs = self.cpg.phase_fractions
        # For tripod, legs 0,2,4 should be ~in-phase and 1,3,5 ~antiphase
        pair_diff = abs(math.sin(self.cpg.phi[0] - self.cpg.phi[2])) + \
                    abs(math.sin(self.cpg.phi[2] - self.cpg.phi[4])) + \
                    abs(math.sin(self.cpg.phi[1] - self.cpg.phi[3])) + \
                    abs(math.sin(self.cpg.phi[3] - self.cpg.phi[5]))
        self._coordination_score += (1.0 - pair_diff / 4.0)

        # Trail
        if Config.DRAW_TRAILS and hasattr(self, 'location'):
            self._trail.append(tuple(self.location))
            if len(self._trail) > 200:
                self._trail.pop(0)

    # ── OpenGL Rendering ─────────────────────────────────────────────────

    def draw(self) -> None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)

        # Body trail
        if Config.DRAW_TRAILS and len(self._trail) > 1:
            glColor4fv([0.4, 0.8, 0.4, 0.25])
            glLineWidth(1.0)
            glBegin(GL_LINE_STRIP)
            for px, py in self._trail:
                # Trail is in world coords; draw() is already translated to agent
                glVertex2d(px - self.location[0], py - self.location[1])
            glEnd()

        # CPG coupling network visualisation
        if Config.DRAW_CPG_NETWORK:
            self._draw_cpg_network()

        # Six legs
        for i in range(6):
            self._draw_leg(i)

        # Body segments
        self._draw_body()

        # Ground reaction force arrows
        if Config.DRAW_FORCE_ARROWS:
            self._draw_force_arrows()

        glDisable(GL_LINE_SMOOTH)

    def _leg_geometry(self, leg_idx: int):
        """Return (attach_x, attach_y, side) in local body frame."""
        bw, bh = 14.0, 20.0
        positions = [
            (-bw, -bh * 0.6, -1),   # L1 front-left
            (-bw * 1.2, 0.0,  -1),  # L2 mid-left
            (-bw, +bh * 0.6,  -1),  # L3 rear-left
            (+bw, -bh * 0.6,  +1),  # R1 front-right
            (+bw * 1.2, 0.0,  +1),  # R2 mid-right
            (+bw, +bh * 0.6,  +1),  # R3 rear-right
        ]
        return positions[leg_idx]

    def _draw_leg(self, i: int) -> None:
        disp, swing = self.cpg.output(i)
        ax, ay, side = self._leg_geometry(i)

        coxa_len = 10.0
        femur_len = 14.0
        tibia_len = 10.0

        sweep  = disp * 8.0 * side
        lift   = max(0.0, disp) * 8.0 if swing else 0.0

        # Coxa end
        cx = ax + side * coxa_len
        cy = ay

        # Tibia tip (foot)
        fx = ax + side * (coxa_len + femur_len * 0.6 + sweep)
        fy = ay + femur_len * 0.5 + lift

        # Knee
        kx = ax + side * (coxa_len + femur_len * 0.5)
        ky = ay + femur_len * 0.3 + lift * 0.4

        if swing:
            glColor4fv([0.9, 0.45, 0.1, 0.9])   # orange swing
        else:
            glColor4fv([0.15, 0.80, 0.40, 0.9])  # green stance
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        glVertex2d(ax, ay)
        glVertex2d(cx, cy)
        glVertex2d(kx, ky)
        glVertex2d(fx, fy)
        glEnd()

        # Foot dot
        if not swing:
            # Stance: bright green with contact flash
            glColor4fv([0.0, 1.0, 0.5, 1.0])
        else:
            glColor4fv([1.0, 0.6, 0.1, 0.8])
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glVertex2d(fx, fy)
        glEnd()

        # Store foot position in world coords for arrows
        if hasattr(self, 'location'):
            self._foot_positions[i] = (
                self.location[0] + fx,
                self.location[1] + fy,
                not swing  # is_contact
            )

    def _draw_body(self) -> None:
        # Head capsule
        glColor4fv([0.15, 0.55, 0.25, 0.95])
        glLineWidth(1.5)
        glBegin(GL_LINE_LOOP)
        for a in range(20):
            ang = a / 20.0 * 2 * math.pi
            glVertex2d(10 * math.cos(ang), 7 * math.sin(ang) - 16)
        glEnd()

        # Thorax
        glColor4fv([0.12, 0.48, 0.22, 0.95])
        glBegin(GL_LINE_LOOP)
        for a in range(24):
            ang = a / 24.0 * 2 * math.pi
            glVertex2d(14 * math.cos(ang), 11 * math.sin(ang))
        glEnd()

        # Abdomen
        glColor4fv([0.10, 0.40, 0.18, 0.9])
        glBegin(GL_LINE_LOOP)
        for a in range(20):
            ang = a / 20.0 * 2 * math.pi
            glVertex2d(11 * math.cos(ang), 9 * math.sin(ang) + 15)
        glEnd()

        # Eyes (phase-coloured by oscillator state)
        for side, i in [(-6, 0), (6, 3)]:
            frac = self.cpg.phase_fractions[i]
            r = 0.2 + 0.8 * frac
            g_ = 0.8 - 0.6 * frac
            glColor4fv([r, g_, 0.5, 1.0])
            glPointSize(6.0)
            glBegin(GL_POINTS)
            glVertex2d(side, -18)
            glEnd()

        # Antennae
        glColor4fv([0.5, 0.3, 0.8, 0.6])
        glLineWidth(1.0)
        for side in (-1, 1):
            glBegin(GL_LINE_STRIP)
            glVertex2d(side * 6, -18)
            glVertex2d(side * 12, -28)
            glVertex2d(side * 9, -38)
            glEnd()

        # Phase indicator arcs around body (CPG phase visualisation)
        if Config.DRAW_CPG_NETWORK:
            for i in range(6):
                frac = self.cpg.phase_fractions[i]
                _, swing = self.cpg.output(i)
                color = ([0.2, 1.0, 0.5, 0.5] if not swing
                         else [1.0, 0.5, 0.1, 0.5])
                glColor4fv(color)
                glLineWidth(2.5)
                arc_start = (i / 6.0) * 2 * math.pi
                arc_end   = arc_start + frac * 2 * math.pi / 6
                glBegin(GL_LINE_STRIP)
                for step in range(12):
                    a = arc_start + (arc_end - arc_start) * step / 11
                    glVertex2d(22 * math.cos(a), 22 * math.sin(a))
                glEnd()

    def _draw_cpg_network(self) -> None:
        """Draw coupling lines between oscillator nodes (in body frame)."""
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
                glVertex2d(xi, yi)
                glVertex2d(xj, yj)
                glEnd()

    def _draw_force_arrows(self) -> None:
        """Draw ground-reaction force arrows for stance legs."""
        for i, fp in enumerate(self._foot_positions):
            if fp is None:
                continue
            fx, fy, contact = fp
            if not contact:
                continue
            lx = fx - self.location[0]
            ly = fy - self.location[1]
            glColor4fv([0.0, 0.9, 0.5, 0.6])
            glLineWidth(1.5)
            glBegin(GL_LINES)
            glVertex2d(lx, ly)
            glVertex2d(lx, ly - 6)
            glEnd()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ENVIRONMENT MARKERS
# ══════════════════════════════════════════════════════════════════════════════

class SlopeMarker(WorldObject):
    """Visual indicator of slope direction (drawn as gradient fill region)."""

    def __init__(self):
        super().__init__(radius=1.0)
        self.colour = [0.7, 0.7, 0.9, 0.15]

    def draw(self) -> None:
        w = WDP.width
        h = WDP.height
        glColor4fv([0.55, 0.55, 0.85, 0.08])
        glBegin(GL_QUADS)
        glVertex2d(-w / 2, -h / 2)
        glVertex2d(+w / 2, -h / 2)
        glColor4fv([0.30, 0.30, 0.65, 0.18])
        glVertex2d(+w / 2,  h / 2)
        glVertex2d(-w / 2,  h / 2)
        glEnd()

    def __del__(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 7.  GENETIC ALGORITHM SUBCLASS  (custom logging)
# ══════════════════════════════════════════════════════════════════════════════

class CPGGeneticAlgorithm(GeneticAlgorithm):
    """
    Extends GeneticAlgorithm with:
    - Gaussian mutation (NormalMutator is already in base; we set sigma here)
    - Best genome tracking and JSON export
    - Per-generation stdout logging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generation_log: list[dict] = []

    def generate(self) -> None:
        super().generate()
        gen_idx = self.generations
        avg = self._average_fitness_record[-1] if self._average_fitness_record else 0
        best = self._best_fitness_record[-1]    if self._best_fitness_record    else 0
        entry = {"generation": gen_idx, "avg": avg, "best": best}
        self._generation_log.append(entry)
        print(f"[GA] Gen {gen_idx:3d}  avg={avg:.4f}  best={best:.4f}")

    def save_best(self, path: str) -> None:
        if self._best_ever_genome is None:
            print("[GA] No genome to save yet.")
            return
        data = {
            "best_fitness": self._best_ever_fitness,
            "genome": self._best_ever_genome.tolist(),
            "fitness_mode": Config.FITNESS_MODE,
            "gait": Config.DEFAULT_GAIT,
            "generations_run": self.generations,
            "history": self._generation_log,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[GA] Best genome saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class CPGHexapodSimulation(Simulation):
    """
    Main simulation class.  Registered with PyBeast++ GUI via:
        IS_DEMO    = True
        DEMO_NAME  = "CPG Hexapod"
        CLASS_NAME = "CPGHexapodSimulation"

    To run headless:
        sim = CPGHexapodSimulation()
        sim.run_simulation(render=False)
    """

    def __init__(self):
        super().__init__("CPGHexapod")

        self.generations  = Config.GENERATIONS
        self.assessments  = Config.ASSESSMENTS
        self.timesteps    = Config.TIMESTEPS

        # Build GA
        mutator = NormalMutator(mu=0.0, sigma=Config.MUTATION_SIGMA)
        self._ga = CPGGeneticAlgorithm(
            crossover=Config.CROSSOVER_RATE,
            mutation=Config.MUTATION_RATE,
            elitism=Config.ELITISM,
            mutator=mutator,
        )

        # Population of hexapod agents
        pop = Population(
            Config.POPULATION_SIZE,
            CPGHexapod,
            self._ga,
        )
        self.add("hexapods", pop)

        # Environment objects
        self._build_environment()

    # ── Environment builders ─────────────────────────────────────────────

    def _build_environment(self) -> None:
        env = Config.ENVIRONMENT
        if env == 'ROUGH':
            self._place_obstacles()
        elif env == 'SLOPE':
            self._setup_slope()
        # FLAT needs nothing extra

    def _place_obstacles(self, count: int = 14) -> None:
        """Scatter obstacles avoiding the centre spawn zone."""
        from core.evolve.base import Group
        obstacles = []
        rng = np.random.default_rng(42)
        w, h = WDP.width, WDP.height
        for _ in range(count):
            # Keep obstacles away from the centre quarter
            while True:
                x = rng.uniform(0.05 * w, 0.95 * w)
                y = rng.uniform(0.05 * h, 0.95 * h)
                if abs(x - w / 2) > 80 or abs(y - h / 2) > 80:
                    break
            loc = np.array([x, y], dtype=np.float32)
            obstacles.append(Obstacle(location=loc, radius=rng.uniform(8, 18)))

        # Wrap in a minimal SimulationObject so the sim lifecycle works
        class _ObstacleGroup:
            def __init__(self, obs):
                self.obs = obs
                self.world = None
            def begin_assessment(self): pass
            def end_assessment(self): pass
            def begin_generation(self): pass
            def end_generation(self): pass
            def begin_run(self): pass
            def end_run(self): pass
            def add_to_world(self):
                for o in self.obs:
                    o.world = self.world
                    self.world.add_object(o)

        self.add("obstacles", _ObstacleGroup(obstacles))

    def _setup_slope(self) -> None:
        """Apply slope bias to all hexapod agents."""
        # Slope angle: 10° uphill toward top of screen
        slope_angle = math.radians(10)

        class _SlopeSetup:
            def __init__(self, angle):
                self.angle = angle
                self.world = None
            def begin_assessment(self):
                for obj in self.world._agents:
                    if isinstance(obj, CPGHexapod):
                        obj._slope_angle = self.angle
            def end_assessment(self): pass
            def begin_generation(self): pass
            def end_generation(self): pass
            def begin_run(self): pass
            def end_run(self): pass
            def add_to_world(self):
                marker = SlopeMarker()
                marker.world = self.world
                self.world.add_object(marker)

        self.add("slope", _SlopeSetup(slope_angle))

    # ── Lifecycle hooks ──────────────────────────────────────────────────

    def begin_simulation(self) -> None:
        print("=" * 60)
        print("  CPG Hexapod Locomotion  –  PyBeast++ Demo")
        print(f"  Fitness mode : {Config.FITNESS_MODE}")
        print(f"  Environment  : {Config.ENVIRONMENT}")
        print(f"  Gait preset  : {Config.DEFAULT_GAIT}")
        print(f"  Population   : {Config.POPULATION_SIZE}")
        print(f"  Generations  : {Config.GENERATIONS}")
        print(f"  Assessments  : {Config.ASSESSMENTS}")
        print("=" * 60)
        super().begin_simulation()

    def log_end_generation(self) -> None:
        pop = self.contents["hexapods"]
        averages = pop.average_member_fitness()
        if averages:
            avg = sum(averages) / len(averages)
            best = max(averages)
            self.log.info(
                f"Gen {self._generation:3d}/{self.generations}  "
                f"avg_fitness={avg:.4f}  best={best:.4f}"
            )

    def end_simulation(self) -> None:
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
# 9.  MANUAL / HEADLESS ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_genome(path: str) -> np.ndarray | None:
    """Load a previously saved best genome from JSON."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    genome = np.array(data['genome'], dtype=np.float64)
    print(f"[Loader] Loaded genome from '{path}'  "
          f"(fitness={data.get('best_fitness', '?'):.4f})")
    return genome


def run_headless(
    generations: int = Config.GENERATIONS,
    population: int  = Config.POPULATION_SIZE,
    fitness_mode: str = Config.FITNESS_MODE,
    environment: str  = Config.ENVIRONMENT,
    save_path: str    = Config.GENOME_SAVE_PATH,
) -> None:
    """Run evolution without GUI.  Results saved to save_path."""
    Config.GENERATIONS    = generations
    Config.POPULATION_SIZE = population
    Config.FITNESS_MODE   = fitness_mode
    Config.ENVIRONMENT    = environment
    Config.GENOME_SAVE_PATH = save_path

    sim = CPGHexapodSimulation()
    # Headless: call the private method directly (no wx App needed)
    sim._run_simulation_no_render(parallel=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CPG Hexapod – headless training")
    parser.add_argument("--generations", type=int, default=Config.GENERATIONS)
    parser.add_argument("--population",  type=int, default=Config.POPULATION_SIZE)
    parser.add_argument("--fitness",     type=str, default=Config.FITNESS_MODE,
                        choices=['DISTANCE', 'EFFICIENCY', 'STABILITY'])
    parser.add_argument("--environment", type=str, default=Config.ENVIRONMENT,
                        choices=['FLAT', 'ROUGH', 'SLOPE'])
    parser.add_argument("--save",        type=str, default=Config.GENOME_SAVE_PATH)
    args = parser.parse_args()

    run_headless(
        generations  = args.generations,
        population   = args.population,
        fitness_mode = args.fitness,
        environment  = args.environment,
        save_path    = args.save,
    )
