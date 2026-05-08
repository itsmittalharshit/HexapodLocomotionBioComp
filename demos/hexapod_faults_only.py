"""
hexapod_faults_only.py  –  Fault-Injecting CPG Hexapod (No Self-Healing)
=========================================================================
Run tests:
    python hexapod_faults_only.py --test
Run full evolution:
    python hexapod_faults_only.py --evolve

WHAT WAS REMOVED vs. hexapod_ais_selfhealing.py
─────────────────────────────────────────────────
• ArtificialImmuneSystem class (clonal selection, memory pool, micro-GA)
• AISMemoryCell dataclass and _hypermutate()
• All AIS Config keys (AIS_MEMORY_POOL_SIZE, AIS_CLONE_COUNT, etc.)
• All REPAIR_* Config keys (micro-evolution repair loop)
• CPGNetwork.rebalance() – no automatic drive compensation on leg loss
• CPGHexapod.ais attribute and all ais.* calls
• _run_ais_patrol() → replaced with _run_fault_patrol() (log only)
• inject_leg_loss() no longer calls AIS.patrol() or rebalances drives
• get_ais_report() → renamed get_fault_report() (AIS fields removed)
• AIS-bonus in get_fitness() (heal_events * 0.5)
• DRAW_AIS_STATUS / _draw_ais_indicator() rendering overlay
• All AIS-specific tests (ais_memory_learning, comparative_fitness,
  full_survival_run now reduced to full_fault_run without healing check)

WHAT IS KEPT
────────────────────────────────────────────────────────────────────────
• FaultType enum & FaultEvent dataclass
• FaultDetector – all 5 detectors run every 10 ticks, faults are LOGGED
• CPGNetwork – Kuramoto + Hebbian plasticity, disable_leg() still works
  (drive set to 0, but remaining legs are NOT rebalanced)
• CPGHexapod – full genome, fitness, neuromodulation, foraging, rendering
• inject_leg_loss(), inject_phase_noise(), inject_coupling_failure() APIs
• GA + Simulation layers (PyBeast++ path unchanged)
• Test suite rewritten to verify fault *injection and logging* rather than
  fault *recovery*
"""

import math, json, os, sys, argparse, copy
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import IntEnum, auto
import sys
import numpy as np

# Add project root to sys.path so we can import 'core'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PYBEAST_AVAILABLE = False
try:
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
    from core.evolve.base import NormalMutator, SimulationObject
    from core.evolve.genetic_algorithm import GeneticAlgorithm
    from core.evolve.population import Population
    from core.simulation import Simulation
    from core.utils import (Vec2, AgentSettings as AS, ColourPalette,
                            ColourType as CT, WORLD_DISPLAY_PARAMETERS as WDP)
    from core.world.world_object import WorldObject
    _PYBEAST_AVAILABLE = True
except ImportError:
    pass

IS_DEMO    = True
DEMO_NAME  = "CPG Hexapod – Fault Injection (No Healing)"
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
    FITNESS_MODE      = 'FORAGING'
    DEFAULT_GAIT      = 'TRIPOD'
    ENVIRONMENT       = 'FLAT'
    CPG_FREQ          = 1.2
    CPG_COUPLING      = 0.65
    CPG_DUTY          = 0.60
    CPG_AMPLITUDE     = 0.80
    TIMESTEP_DT       = 0.05
    MAX_SPEED         = 120.0
    MIN_SPEED         = 0.0
    FOOD_COUNT        = 12
    FOOD_CALORIES     = 25.0
    AGENT_START_ENERGY= 40.0
    ENERGY_PER_STEP   = 0.04
    HEBBIAN_LR        = 0.002
    HEBBIAN_DECAY     = 0.0001
    HEBBIAN_W_MIN     = 0.05
    HEBBIAN_W_MAX     = 2.0
    HEBBIAN_SYNC_THR  = 0.25
    FEEDBACK_THRESHOLD  = 0.5
    FEEDBACK_DECAY      = 0.85
    FEEDBACK_MAGNITUDE  = 1.0
    LEG_LENGTH_MIN    = 0.5
    LEG_LENGTH_MAX    = 2.0
    STIFFNESS_MIN     = 0.0
    STIFFNESS_MAX     = 1.0
    FATIGUE_GROWTH         = 0.0008
    FATIGUE_RECOVERY       = 0.0020
    FATIGUE_FOOD_RESET     = 0.40
    FATIGUE_MAX            = 1.0
    FATIGUE_FREQ_GAIN      = 0.60
    FATIGUE_AMPLITUDE_GAIN = 0.50
    FATIGUE_HEBBIAN_SCALE  = 0.30
    # Fault detection thresholds
    FAULT_SILENT_TICKS    = 20
    FAULT_PHASE_THR       = 1.2
    FAULT_WEIGHT_MIN      = 0.08
    FAULT_SYMMETRY_THR    = 0.60
    ENERGY_CRISIS_THR     = 5.0
    # Visualisation
    DRAW_CPG_NETWORK   = True
    DRAW_TRAILS        = True
    DRAW_FORCE_ARROWS  = True
    SAVE_BEST_GENOME   = True
    GENOME_SAVE_PATH   = "best_hexapod_faults.json"


GAIT_PHASES = {
    'TRIPOD': np.array([0, np.pi, 0, np.pi, 0, np.pi], dtype=np.float64),
    'WAVE':   np.array([0, np.pi/3, 2*np.pi/3,
                        np.pi, 4*np.pi/3, 5*np.pi/3], dtype=np.float64),
    'RIPPLE': np.array([0, 2*np.pi/3, 4*np.pi/3,
                        np.pi/3, np.pi, 5*np.pi/3], dtype=np.float64),
}
LEG_NAMES = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']


# ══════════════════════════════════════════════════════════════════════════════
# FAULT TYPES & EVENTS
# ══════════════════════════════════════════════════════════════════════════════

class FaultType(IntEnum):
    NONE          = 0
    LEG_LOSS      = auto()
    PHASE_DRIFT   = auto()
    COUPLING_FAIL = auto()
    ENERGY_CRISIS = auto()
    SYMMETRY_FAIL = auto()


@dataclass
class FaultEvent:
    fault_type: FaultType
    leg_mask:   np.ndarray
    severity:   float
    timestamp:  int
    resolved:   bool = False
    resolution_time: int = -1


# ══════════════════════════════════════════════════════════════════════════════
# FAULT DETECTION ENGINE  (logging only – no healing)
# ══════════════════════════════════════════════════════════════════════════════

class FaultDetector:
    def __init__(self):
        self._silent_ticks = np.zeros(6, dtype=int)
        self._history      = deque(maxlen=30)
        self.active_faults: List[FaultEvent] = []

    def reset(self):
        self._silent_ticks[:] = 0
        self._history.clear()
        self.active_faults.clear()

    def update(self, cpg, disabled_legs, left_drive, right_drive,
               energy, timestep) -> Optional[FaultEvent]:
        faults = []

        # 1. LEG_LOSS
        for i in range(6):
            disp, _ = cpg.output(i)
            if abs(disp) < 0.02 or disabled_legs[i]:
                self._silent_ticks[i] += 1
            else:
                self._silent_ticks[i] = 0
            if self._silent_ticks[i] >= Config.FAULT_SILENT_TICKS:
                mask = np.zeros(6, bool); mask[i] = True
                faults.append(FaultEvent(FaultType.LEG_LOSS, mask,
                    min(1.0, self._silent_ticks[i] / 60), timestep))

        # 2. PHASE_DRIFT
        phase_err = np.abs((cpg.phi - cpg.desired) % (2 * np.pi))
        phase_err = np.where(phase_err > np.pi, 2*np.pi - phase_err, phase_err)
        drifted = phase_err > Config.FAULT_PHASE_THR
        if drifted.any():
            faults.append(FaultEvent(FaultType.PHASE_DRIFT, drifted.copy(),
                min(1.0, float(np.max(phase_err[drifted]) / np.pi)), timestep))

        # 3. COUPLING_FAIL
        diag_mask = ~np.eye(6, dtype=bool)
        avg_w = float(np.mean(cpg.coupling_weights[diag_mask]))
        if avg_w < Config.FAULT_WEIGHT_MIN:
            faults.append(FaultEvent(FaultType.COUPLING_FAIL, np.ones(6, bool),
                min(1.0, 1.0 - avg_w / Config.FAULT_WEIGHT_MIN), timestep))

        # 4. ENERGY_CRISIS
        if energy < Config.ENERGY_CRISIS_THR:
            faults.append(FaultEvent(FaultType.ENERGY_CRISIS, np.ones(6, bool),
                1.0 - energy / Config.ENERGY_CRISIS_THR, timestep))

        # 5. SYMMETRY_FAIL
        total = abs(left_drive) + abs(right_drive) + 1e-9
        imbalance = abs(left_drive - right_drive) / total
        if imbalance > Config.FAULT_SYMMETRY_THR:
            mask = np.zeros(6, bool)
            mask[[0,1,2]] = left_drive < right_drive
            mask[[3,4,5]] = left_drive >= right_drive
            faults.append(FaultEvent(FaultType.SYMMETRY_FAIL, mask,
                min(1.0, imbalance), timestep))

        if not faults:
            return None
        faults.sort(key=lambda f: f.severity, reverse=True)
        return faults[0]


# ══════════════════════════════════════════════════════════════════════════════
# CPG NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class CPGNetwork:
    """
    Kuramoto CPG with Hebbian plasticity.
    disable_leg(i) silences a leg permanently with NO drive rebalancing
    (rebalance() has been removed along with the AIS that called it).
    """

    def __init__(self, freq, coupling, duty, amplitude, phase_offsets):
        self.freq          = float(freq)
        self.base_coupling = float(coupling)
        self.duty          = float(duty)
        self.amplitude     = float(amplitude)
        self.phi           = np.array(phase_offsets, dtype=np.float64)
        self.desired       = np.array(phase_offsets, dtype=np.float64)
        self.eff_freq      = self.freq
        self.eff_amplitude = self.amplitude
        self.coupling_weights = np.full((6,6), coupling, dtype=np.float64)
        np.fill_diagonal(self.coupling_weights, 0.0)
        self.drive_weights  = np.ones(6, dtype=np.float64)
        self._disabled_legs = np.zeros(6, dtype=bool)

    def disable_leg(self, i: int) -> None:
        """Silence leg i permanently. No compensation for other legs."""
        self._disabled_legs[i] = True
        self.drive_weights[i]  = 0.0

    def enable_leg(self, i: int) -> None:
        """Re-enable a leg (used for test resets only)."""
        self._disabled_legs[i] = False
        self.drive_weights[i]  = 1.0

    def reset_weights(self) -> None:
        self.coupling_weights[:] = self.base_coupling
        np.fill_diagonal(self.coupling_weights, 0.0)
        self.drive_weights[:]    = 1.0
        self._disabled_legs[:]   = False

    def step(self, dt, feedback=None, eff_freq=None, eff_amplitude=None,
             fatigue=0.0):
        self.eff_freq      = eff_freq      if eff_freq      is not None else self.freq
        self.eff_amplitude = eff_amplitude if eff_amplitude is not None else self.amplitude
        if feedback is not None:
            thr = Config.FEEDBACK_THRESHOLD
            for i in range(6):
                if feedback[i] > thr and not self._disabled_legs[i]:
                    self.phi[i] = self.desired[i] + 0.05
        omega = 2.0 * np.pi * self.eff_freq
        dphi  = np.zeros(6)
        for i in range(6):
            if self._disabled_legs[i]: continue
            cs = 0.0
            for j in range(6):
                if i != j and not self._disabled_legs[j]:
                    delta = self.desired[j] - self.desired[i]
                    cs   += self.coupling_weights[i,j] * math.sin(
                        self.phi[j] - self.phi[i] - delta)
            dphi[i] = omega + cs
        self.phi += dphi * dt
        self._update_hebbian(dt, fatigue)

    def _update_hebbian(self, dt, fatigue=0.0):
        lr    = Config.HEBBIAN_LR * (1.0 - Config.FATIGUE_HEBBIAN_SCALE * fatigue)
        decay = Config.HEBBIAN_DECAY
        thr   = Config.HEBBIAN_SYNC_THR
        for i in range(6):
            if self._disabled_legs[i]: continue
            for j in range(6):
                if i == j or self._disabled_legs[j]: continue
                delta    = self.desired[j] - self.desired[i]
                phase_err= abs((self.phi[j] - self.phi[i] - delta) % (2*np.pi))
                if phase_err > np.pi: phase_err = 2*np.pi - phase_err
                sync = 1.0 if phase_err < thr else -0.5
                dW   = lr * (sync - decay * self.coupling_weights[i,j])
                self.coupling_weights[i,j] = max(Config.HEBBIAN_W_MIN,
                    min(Config.HEBBIAN_W_MAX,
                        self.coupling_weights[i,j] + dW * dt))

    def output(self, leg_idx) -> Tuple[float, bool]:
        if self._disabled_legs[leg_idx]:
            return 0.0, False
        phi_norm = self.phi[leg_idx] % (2.0 * np.pi)
        x = math.sin(phi_norm) * self.eff_amplitude * self.drive_weights[leg_idx]
        return x, math.sin(phi_norm) > 0.0

    @property
    def phase_fractions(self):
        return (self.phi % (2.0 * np.pi)) / (2.0 * np.pi)

    @property
    def avg_weight(self):
        mask = ~np.eye(6, dtype=bool)
        return float(np.mean(self.coupling_weights[mask]))

    @property
    def disabled_legs(self):
        return self._disabled_legs.copy()


# ══════════════════════════════════════════════════════════════════════════════
# WORLD OBJECTS
# ══════════════════════════════════════════════════════════════════════════════

if _PYBEAST_AVAILABLE:
    class Obstacle(WorldObject):
        def __init__(self, location=None, radius=12.0):
            super().__init__(location=location, radius=radius, solid=True)
            self.colour = ColourPalette[CT.DARK_GREY]
        def draw(self):
            glColor4fv([0.35, 0.33, 0.30, 0.9])
            q = gluNewQuadric(); gluDisk(q, 0, self.radius, 16, 1); gluDeleteQuadric(q)
        def __del__(self): pass

    class FoodPellet(WorldObject):
        def __init__(self, location=None, calories=None):
            super().__init__(location=location, radius=10.0, solid=False)
            self.calories = calories if calories is not None else Config.FOOD_CALORIES
            self.eaten = False; self._pulse = 0.0
        def draw(self):
            if self.eaten: return
            self._pulse = (self._pulse + 0.05) % (2 * math.pi)
            glow = 0.7 + 0.3 * math.sin(self._pulse)
            glColor4fv([0.9*glow, 0.8*glow, 0.1, 0.35]); glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            for a in range(20):
                ang = a/20.0*2*math.pi
                glVertex2d(14*math.cos(ang), 14*math.sin(ang))
            glEnd()
            glColor4fv([1.0*glow, 0.85*glow, 0.1, 0.85])
            q = gluNewQuadric(); gluDisk(q, 0, self.radius, 20, 1); gluDeleteQuadric(q)
            glColor4fv([1.0, 1.0, 0.6, 0.9]); glPointSize(4.0)
            glBegin(GL_POINTS); glVertex2d(0,0); glEnd()
        def __del__(self): pass


# ══════════════════════════════════════════════════════════════════════════════
# HEXAPOD AGENT
# ══════════════════════════════════════════════════════════════════════════════

if _PYBEAST_AVAILABLE:
    _AgentBase   = Agent
    _EvolverBase = Evolver
else:
    class _AgentBase:
        def __init__(self, **kw): self.location = np.array([0., 0.])
        def reset(self): pass
    class _EvolverBase:
        def __init__(self): pass


class CPGHexapod(_AgentBase, _EvolverBase):
    """
    Hexapod with fault detection (logging only).  No self-healing.

    Genome (13 genes):
      0   freq           [0.3, 3.0]
      1   coupling       [0.0, 1.0]
      2   duty           [0.3, 0.85]
      3   amplitude      [0.1, 1.0]
      4-9 phase offsets  [0, 2π] × 6
      10  leg_length     [0.5, 2.0]
      11  joint_stiffness[0.0, 1.0]
    """
    GENOME_LENGTH = 13
    GENE_SCALE = [(0.3,3.0),(0.0,1.0),(0.3,0.85),(0.1,1.0)] \
               + [(0.0, 2*np.pi)]*6 \
               + [(Config.LEG_LENGTH_MIN, Config.LEG_LENGTH_MAX),
                  (Config.STIFFNESS_MIN,  Config.STIFFNESS_MAX)]
    BASE_COXA  = 10.0
    BASE_FEMUR = 14.0

    def __init__(self):
        if _PYBEAST_AVAILABLE:
            _AgentBase.__init__(self, min_speed=Config.MIN_SPEED,
                max_speed=Config.MAX_SPEED, timestep=Config.TIMESTEP_DT,
                random_colour=False, solid=False)
            _EvolverBase.__init__(self)
            self.radius = 18.0
            self.colour = ColourPalette[CT.GREEN]
        else:
            _AgentBase.__init__(self)
            _EvolverBase.__init__(self)

        default_phases = GAIT_PHASES.get(Config.DEFAULT_GAIT,
                                         GAIT_PHASES['TRIPOD']).copy()
        self.cpg = CPGNetwork(Config.CPG_FREQ, Config.CPG_COUPLING,
                              Config.CPG_DUTY, Config.CPG_AMPLITUDE,
                              default_phases)
        self.fault_detector     = FaultDetector()
        self._current_fault: Optional[FaultEvent] = None
        self._fault_log: List[FaultEvent] = []
        self._fault_patrol_tick = 0
        self._timestep_count    = 0

        self.leg_length      = 1.0
        self.joint_stiffness = 0.3
        self._spring_energy  = 0.0
        self._fatigue        = 0.0
        self._feedback       = np.zeros(6, dtype=np.float64)
        self._reflex_count   = 0

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
        self._left_drive         = 0.0
        self._right_drive        = 0.0

    # ── Genome ────────────────────────────────────────────────────────────

    def _scale_gene(self, raw, idx):
        lo, hi = self.GENE_SCALE[idx]
        return lo + (hi - lo) * max(0.0, min(1.0, raw))

    def set_genotype(self, genome):
        assert len(genome) == self.GENOME_LENGTH
        g = np.asarray(genome, dtype=np.float64)
        freq   = self._scale_gene(g[0], 0)
        coupling = self._scale_gene(g[1], 1)
        duty   = self._scale_gene(g[2], 2)
        amp    = self._scale_gene(g[3], 3)
        phases = np.array([self._scale_gene(g[4+i], 4+i) for i in range(6)])
        self.cpg = CPGNetwork(freq, coupling, duty, amp, phases)
        self.leg_length      = self._scale_gene(g[10], 10)
        self.joint_stiffness = self._scale_gene(g[11], 11)

    def get_genotype(self):
        g = np.zeros(self.GENOME_LENGTH)
        for idx, (lo, hi) in enumerate(self.GENE_SCALE[:4]):
            vals = [self.cpg.freq, self.cpg.base_coupling,
                    self.cpg.duty, self.cpg.amplitude]
            g[idx] = (vals[idx] - lo) / (hi - lo)
        for i in range(6):
            lo, hi = self.GENE_SCALE[4+i]
            g[4+i] = (self.cpg.desired[i] - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[10]; g[10] = (self.leg_length - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[11]; g[11] = (self.joint_stiffness - lo) / (hi - lo)
        return g

    # ── Fitness (no AIS heal bonus) ───────────────────────────────────────

    def get_fitness(self):
        if self._step_count == 0: return 0.0
        base         = self._calories_gathered
        explore      = self._distance_travelled * 0.01
        plasticity   = self.cpg.avg_weight * 0.3
        reflex_bonus = self._reflex_count * 0.3
        penalty      = self._collision_penalty * 5.0
        fatigue_pen  = self._fatigue * 3.0
        morph_cost   = (self.leg_length - 1.0)**2 * self.joint_stiffness * 1.5
        return max(0.0, base + explore + plasticity + reflex_bonus
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
        self._current_fault      = None
        self._fault_patrol_tick  = 0
        self.fault_detector.reset()
        self.cpg.reset_weights()
        super().reset() if _PYBEAST_AVAILABLE else None

    # ── Collisions ────────────────────────────────────────────────────────

    def on_collision(self, other):
        if not _PYBEAST_AVAILABLE: return
        if isinstance(other, Obstacle):
            self._collision_penalty += 1.0
            self._feedback[:] = Config.FEEDBACK_MAGNITUDE
            self._reflex_count += 1
        elif isinstance(other, FoodPellet) and not other.eaten:
            other.eaten              = True
            other.dead               = True
            self._calories_gathered += other.calories
            self._energy_remaining  += other.calories
            self._foods_eaten       += 1
            self._fatigue = max(0.0,
                self._fatigue - self._fatigue * Config.FATIGUE_FOOD_RESET)

    # ── Control ───────────────────────────────────────────────────────────

    def control(self):
        dt = Config.TIMESTEP_DT
        self._timestep_count += 1

        if self._energy_remaining <= 0:
            if _PYBEAST_AVAILABLE:
                self.controls['left'] = self.controls['right'] = 0.0
            self._fatigue = max(0.0, self._fatigue - Config.FATIGUE_RECOVERY * dt)
            return

        eff_freq = self.cpg.freq * (1.0 - Config.FATIGUE_FREQ_GAIN * self._fatigue)
        eff_amp  = self.cpg.amplitude * (1.0 - Config.FATIGUE_AMPLITUDE_GAIN * self._fatigue)

        self.cpg.step(dt, feedback=self._feedback,
                      eff_freq=eff_freq, eff_amplitude=eff_amp,
                      fatigue=self._fatigue)
        self._feedback *= Config.FEEDBACK_DECAY

        left_drive = right_drive = energy_tick = 0.0
        for i, name in enumerate(LEG_NAMES):
            disp, swing = self.cpg.output(i)
            self._leg_states[i] = swing
            is_left = name.startswith('L')
            dw = self.cpg.drive_weights[i]
            if not swing:
                drive = self.cpg.duty * disp * self.leg_length * dw
                if is_left: left_drive  += drive / 3.0
                else:       right_drive += drive / 3.0
                energy_tick += abs(drive)
                self._spring_energy += 0.5 * self.joint_stiffness * (disp**2) * dt
            else:
                spring_boost = self._spring_energy * 0.3 * self.leg_length * dw
                if is_left: left_drive  += spring_boost / 3.0
                else:       right_drive += spring_boost / 3.0
                self._spring_energy = max(0.0, self._spring_energy - spring_boost*2)
                energy_tick += 0.02

        self._left_drive  = max(-1.0, min(1.0, left_drive))
        self._right_drive = max(-1.0, min(1.0, right_drive))
        if _PYBEAST_AVAILABLE:
            self.controls['left']  = self._left_drive
            self.controls['right'] = self._right_drive

        movement_cost = energy_tick * Config.ENERGY_PER_STEP * dt
        self._energy_consumed  += movement_cost
        self._energy_remaining -= movement_cost
        self._step_count       += 1
        self._fatigue = max(0.0, min(Config.FATIGUE_MAX,
            self._fatigue + Config.FATIGUE_GROWTH * energy_tick
            - Config.FATIGUE_RECOVERY * dt))

        if _PYBEAST_AVAILABLE and hasattr(self, 'velocity'):
            spd = math.hypot(*self.velocity)
        else:
            spd = abs(left_drive + right_drive) * 10.0
        self._distance_travelled += spd * dt

        phi = self.cpg.phi
        pair_diff = (abs(math.sin(phi[0]-phi[2])) + abs(math.sin(phi[2]-phi[4])) +
                     abs(math.sin(phi[1]-phi[3])) + abs(math.sin(phi[3]-phi[5])))
        self._coordination_score += 1.0 - pair_diff / 4.0

        if Config.DRAW_TRAILS and _PYBEAST_AVAILABLE and hasattr(self, 'location'):
            self._trail.append(tuple(self.location))
            if len(self._trail) > 200: self._trail.pop(0)

        # Fault patrol: detect and log, no healing
        self._fault_patrol_tick += 1
        if self._fault_patrol_tick >= 10:
            self._fault_patrol_tick = 0
            self._run_fault_patrol()

    def _run_fault_patrol(self):
        """Detect faults and log them.  No healing applied."""
        fault = self.fault_detector.update(
            self.cpg, self.cpg.disabled_legs,
            self._left_drive, self._right_drive,
            self._energy_remaining, self._timestep_count)
        if fault is not None:
            if (self._current_fault is None or
                    fault.fault_type != self._current_fault.fault_type):
                self._fault_log.append(fault)
                self._current_fault = fault
        else:
            if self._current_fault is not None:
                self._current_fault.resolved      = True
                self._current_fault.resolution_time = self._timestep_count
                self._current_fault = None

    # ── Fault injection API ───────────────────────────────────────────────

    def inject_leg_loss(self, leg_idx: int) -> None:
        """
        Permanently break a leg (hardware failure simulation).
        The robot receives NO compensatory drive adjustment — it limps
        on however many legs remain.
        """
        self.cpg.disable_leg(leg_idx)
        mask = np.zeros(6, bool); mask[leg_idx] = True
        fault = FaultEvent(FaultType.LEG_LOSS, mask, 1.0, self._timestep_count)
        self._fault_log.append(fault)
        self._current_fault = fault

    def inject_phase_noise(self, sigma: float = 0.8) -> None:
        """Perturb all oscillator phases."""
        self.cpg.phi += np.random.normal(0, sigma, 6)

    def inject_coupling_failure(self) -> None:
        """Collapse all Hebbian coupling weights."""
        self.cpg.coupling_weights[:] = Config.FAULT_WEIGHT_MIN * 0.5
        np.fill_diagonal(self.cpg.coupling_weights, 0.0)

    def get_fault_report(self) -> dict:
        return {
            "faults_seen":   len(self._fault_log),
            "fault_types":   [f.fault_type.name for f in self._fault_log],
            "fault_times":   [f.timestamp for f in self._fault_log],
            "resolved":      [f.resolved for f in self._fault_log],
            "fitness":       self.get_fitness(),
            "distance":      self._distance_travelled,
            "energy_left":   self._energy_remaining,
            "coord_score":   self._coordination_score / max(1, self._step_count),
            "disabled_legs": self.cpg.disabled_legs.tolist(),
        }

    # ── Rendering ─────────────────────────────────────────────────────────

    if _PYBEAST_AVAILABLE:
        def draw(self):
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LINE_SMOOTH)
            if Config.DRAW_TRAILS and len(self._trail) > 1:
                glColor4fv([0.4, 0.8, 0.4, 0.25]); glLineWidth(1.0)
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
            self._draw_energy_bar()
            self._draw_fatigue_ring()
            self._draw_feedback_halo()
            glDisable(GL_LINE_SMOOTH)

        def _draw_cpg_network(self):
            radius = 28.0
            for i in range(6):
                ai = i/6.0*2*math.pi - math.pi/2
                xi, yi = radius*math.cos(ai), radius*math.sin(ai)
                for j in range(i+1, 6):
                    aj = j/6.0*2*math.pi - math.pi/2
                    xj, yj = radius*math.cos(aj), radius*math.sin(aj)
                    w = self.cpg.coupling_weights[i,j]
                    norm_w = (w - Config.HEBBIAN_W_MIN) / (Config.HEBBIAN_W_MAX - Config.HEBBIAN_W_MIN + 1e-9)
                    if (self.cpg.disabled_legs[i] or self.cpg.disabled_legs[j]):
                        glColor4fv([0.9, 0.1, 0.1, 0.25])
                    else:
                        glColor4fv([norm_w*0.9, norm_w*0.7, 1.0-norm_w, 0.05+norm_w*0.6])
                    glLineWidth(max(0.5, norm_w * 3.0))
                    glBegin(GL_LINES); glVertex2d(xi,yi); glVertex2d(xj,yj); glEnd()
            glLineWidth(1.0)

        def _draw_energy_bar(self):
            ratio = max(0.0, min(1.0,
                self._energy_remaining / (Config.AGENT_START_ENERGY + self._calories_gathered + 0.01)))
            bar_w = 30.0; bar_h = 4.0; x0 = -15.0; y0 = -28.0
            glColor4fv([0.3, 0.0, 0.0, 0.7]); glBegin(GL_QUADS)
            glVertex2d(x0,y0); glVertex2d(x0+bar_w,y0)
            glVertex2d(x0+bar_w,y0+bar_h); glVertex2d(x0,y0+bar_h); glEnd()
            glColor4fv([1.0-ratio, ratio, 0.1, 0.9]); glBegin(GL_QUADS)
            glVertex2d(x0,y0); glVertex2d(x0+bar_w*ratio,y0)
            glVertex2d(x0+bar_w*ratio,y0+bar_h); glVertex2d(x0,y0+bar_h); glEnd()

        def _draw_fatigue_ring(self):
            if self._fatigue < 0.02: return
            r_ring = 20.0 + self._fatigue * 8.0
            glColor4fv([1.0, 0.4*(1-self._fatigue), 0.0, self._fatigue*0.7])
            glLineWidth(2.0); glBegin(GL_LINE_LOOP)
            for a in range(24):
                ang = a/24.0*2*math.pi
                glVertex2d(r_ring*math.cos(ang), r_ring*math.sin(ang))
            glEnd(); glLineWidth(1.0)

        def _draw_feedback_halo(self):
            max_fb = float(np.max(self._feedback))
            if max_fb < 0.05: return
            glColor4fv([1.0, 0.1, 0.1, min(0.6, max_fb*0.6)])
            glLineWidth(3.0); glBegin(GL_LINE_LOOP)
            for a in range(24):
                ang = a/24.0*2*math.pi
                glVertex2d(24*math.cos(ang), 24*math.sin(ang))
            glEnd(); glLineWidth(1.0)

        def _leg_geometry(self, leg_idx):
            bw = 14.0*self.leg_length; bh = 20.0
            return [(-bw,-bh*.6,-1),(-bw*1.2,0.,-1),(-bw,+bh*.6,-1),
                    (+bw,-bh*.6,+1),(+bw*1.2,0.,+1),(+bw,+bh*.6,+1)][leg_idx]

        def _draw_leg(self, i):
            disp, swing = self.cpg.output(i)
            ax, ay, side = self._leg_geometry(i)
            coxa  = self.BASE_COXA  * self.leg_length
            femur = self.BASE_FEMUR * self.leg_length
            sweep = disp * 8.0 * side * self.leg_length
            lift  = max(0.0, disp) * 8.0 * self.leg_length if swing else 0.0
            cx = ax + side*coxa; cy = ay
            kx = ax + side*(coxa+femur*0.5); ky = ay + femur*0.3 + lift*0.4
            fx = ax + side*(coxa+femur*0.6+sweep); fy = ay + femur*0.5 + lift
            if self.cpg.disabled_legs[i]:
                glColor4fv([0.5, 0.0, 0.0, 0.5])   # dark red = broken leg
            elif self._feedback[i] > Config.FEEDBACK_THRESHOLD:
                glColor4fv([1.0, 0.1, 0.1, 0.9])
            elif swing:
                glColor4fv([0.9, 0.45, 0.1, 0.9])
            else:
                r = 0.15 + 0.7*(1-self.joint_stiffness) + 0.15*self._fatigue
                b = 0.2  + 0.7*self.joint_stiffness
                glColor4fv([r, 0.80*(1-0.5*self._fatigue), b, 0.9])
            glLineWidth(1.5 + self.joint_stiffness*2.0)
            glBegin(GL_LINE_STRIP)
            glVertex2d(ax,ay); glVertex2d(cx,cy); glVertex2d(kx,ky); glVertex2d(fx,fy)
            glEnd()
            glColor4fv([0.0,1.0,0.5,1.0] if not swing else [1.0,0.6,0.1,0.8])
            glPointSize(5.0 + self.leg_length*2.0)
            glBegin(GL_POINTS); glVertex2d(fx,fy); glEnd()
            if hasattr(self, 'location'):
                self._foot_positions[i] = (self.location[0]+fx, self.location[1]+fy, not swing)

        def _draw_body(self):
            s = 0.8 + 0.4*(self.leg_length-Config.LEG_LENGTH_MIN)/(Config.LEG_LENGTH_MAX-Config.LEG_LENGTH_MIN)
            for colour, rx, ry, dy, n in [
                ([0.15,0.55,0.25,0.95],int(10*s),int(7*s),-16,20),
                ([0.12,0.48,0.22,0.95],int(14*s),int(11*s),0,24),
                ([0.10,0.40,0.18,0.90],int(11*s),int(9*s),15,20),
            ]:
                glColor4fv(colour); glLineWidth(1.5); glBegin(GL_LINE_LOOP)
                for a in range(n):
                    ang = a/n*2*math.pi
                    glVertex2d(rx*math.cos(ang), ry*math.sin(ang)+dy)
                glEnd()
            for side, idx in [(-6,0),(6,3)]:
                frac = self.cpg.phase_fractions[idx]
                glColor4fv([0.2+0.8*frac, 0.8-0.6*frac, 0.5, 1.0])
                glPointSize(6.0); glBegin(GL_POINTS); glVertex2d(side,-18); glEnd()
            glColor4fv([0.5,0.3,0.8,0.6]); glLineWidth(1.0)
            for side in (-1,1):
                glBegin(GL_LINE_STRIP)
                glVertex2d(side*6,-18); glVertex2d(side*12,-28); glVertex2d(side*9,-38)
                glEnd()

        def _draw_force_arrows(self):
            for fp in self._foot_positions:
                if fp is None: continue
                fx, fy, contact = fp
                if not contact: continue
                lx = fx-self.location[0]; ly = fy-self.location[1]
                glColor4fv([0.0,0.9,0.5,0.6]); glLineWidth(1.5)
                glBegin(GL_LINES); glVertex2d(lx,ly); glVertex2d(lx,ly-6); glEnd()


# ══════════════════════════════════════════════════════════════════════════════
# GA + SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

if _PYBEAST_AVAILABLE:
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
                    "genome_length": CPGHexapod.GENOME_LENGTH,
                    "generations_run": self.generations,
                    "history": self._generation_log}
            with open(path, 'w') as f: json.dump(data, f, indent=2)
            print(f"[GA] Best genome saved → {path}")

    class CPGHexapodSimulation(Simulation):
        def __init__(self):
            super().__init__("CPGHexapod_FaultsOnly")
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
                    if abs(x-w/2) > 60 or abs(y-h/2) > 60: break
                food_items.append(FoodPellet(
                    location=np.array([x,y], dtype=np.float32),
                    calories=Config.FOOD_CALORIES))
            class _FoodGroup(SimulationObject):
                def __init__(self, items):
                    super().__init__(); self.items = items
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
            print("=" * 65)
            print("  CPG Hexapod – Fault Injection (No Self-Healing)")
            print("  Active: fault detector (log only), Hebbian plasticity,")
            print("          proprioception, morphology, foraging, fatigue")
            print(f"  Genome: {CPGHexapod.GENOME_LENGTH}  Pop: {Config.POPULATION_SIZE}"
                  f"  Gens: {Config.GENERATIONS}")
            print("=" * 65)
            super().begin_simulation()

        def end_simulation(self):
            super().end_simulation()
            if Config.SAVE_BEST_GENOME:
                self._ga.save_best(Config.GENOME_SAVE_PATH)
            print("=" * 65)
            if self._ga._best_ever_genome is not None:
                print(f"  Best fitness: {self._ga._best_ever_fitness:.4f}")
            print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ══════════════════════════════════════════════════════════════════════════════

class _TestResult:
    def __init__(self, name):
        self.name = name; self.passed = False; self.message = ""; self.metrics = {}
    def ok(self, msg="", **metrics):
        self.passed = True; self.message = msg; self.metrics = metrics; return self
    def fail(self, msg="", **metrics):
        self.passed = False; self.message = msg; self.metrics = metrics; return self
    def __str__(self):
        s = f"  {'✓ PASS' if self.passed else '✗ FAIL'}  {self.name}"
        if self.message: s += f"\n         {self.message}"
        for k, v in self.metrics.items(): s += f"\n         {k}: {v}"
        return s


def _run_steps(agent, n):
    for _ in range(n): agent.control()


def test_leg_loss_permanent() -> _TestResult:
    """Break one leg; confirm permanent disablement and fault logging."""
    r = _TestResult("Leg-loss: permanent disable + fault logged")
    agent = CPGHexapod()
    _run_steps(agent, 60)
    coord_before = agent._coordination_score / max(1, agent._step_count)

    agent.inject_leg_loss(2)   # L3 breaks
    _run_steps(agent, 120)
    coord_after = agent._coordination_score / max(1, agent._step_count)
    report = agent.get_fault_report()

    leg_broken   = agent.cpg.disabled_legs[2]
    fault_logged = any(t == "LEG_LOSS" for t in report["fault_types"])

    if leg_broken and fault_logged:
        return r.ok("L3 permanently broken; fault correctly logged.",
            coord_before=f"{coord_before:.3f}",
            coord_after=f"{coord_after:.3f}",
            coord_delta=f"{coord_after - coord_before:+.3f}",
            faults_seen=report["faults_seen"],
            disabled_legs=report["disabled_legs"])
    return r.fail("Leg not disabled or fault not logged.",
        leg_broken=leg_broken, fault_logged=fault_logged)


def test_two_leg_loss() -> _TestResult:
    """Break two legs simultaneously; verify both stay disabled."""
    r = _TestResult("Two-leg loss: L1 + R3 permanently broken")
    agent = CPGHexapod()
    _run_steps(agent, 50)
    agent.inject_leg_loss(0)   # L1
    agent.inject_leg_loss(5)   # R3
    _run_steps(agent, 150)
    report = agent.get_fault_report()
    n = int(np.sum(agent.cpg.disabled_legs))
    if n == 2:
        return r.ok("Two legs broken; hexapod limping on 4.",
            disabled_legs=report["disabled_legs"],
            coord=f"{report['coord_score']:.3f}",
            distance=f"{report['distance']:.1f}")
    return r.fail("Unexpected disabled leg count.", n_disabled=n)


def test_phase_drift_no_repair() -> _TestResult:
    """Inject phase noise; confirm coordination drops with no correction."""
    r = _TestResult("Phase-drift: degradation confirmed, no repair")
    agent = CPGHexapod()
    _run_steps(agent, 60)
    coord_before = agent._coordination_score / max(1, agent._step_count)
    agent.inject_phase_noise(sigma=1.5)
    _run_steps(agent, 100)
    coord_after = agent._coordination_score / max(1, agent._step_count)
    report = agent.get_fault_report()
    if coord_after < coord_before or report["faults_seen"] > 0:
        return r.ok("Phase noise degraded coordination as expected.",
            coord_before=f"{coord_before:.3f}",
            coord_after=f"{coord_after:.3f}",
            faults_seen=report["faults_seen"])
    return r.fail("No measurable degradation after phase noise.",
        coord_before=f"{coord_before:.3f}", coord_after=f"{coord_after:.3f}")


def test_coupling_failure_no_repair() -> _TestResult:
    """Collapse coupling weights; confirm they stay collapsed."""
    r = _TestResult("Coupling failure: weights stay collapsed (no AIS repair)")
    agent = CPGHexapod()
    _run_steps(agent, 60)
    agent.inject_coupling_failure()
    _run_steps(agent, 100)
    avg_w = float(np.mean(agent.cpg.coupling_weights[~np.eye(6, dtype=bool)]))
    report = agent.get_fault_report()
    if avg_w < Config.FAULT_WEIGHT_MIN * 5:
        return r.ok("Coupling weights remain collapsed.",
            avg_coupling=f"{avg_w:.4f}", faults_seen=report["faults_seen"])
    return r.fail("Coupling weights recovered unexpectedly.", avg_coupling=f"{avg_w:.4f}")


def test_full_fault_run() -> _TestResult:
    """Four sequential faults; all logged, none healed."""
    r = _TestResult("Full fault run: 4 faults injected, none repaired")
    agent = CPGHexapod()
    agent._energy_remaining = 9999.0
    schedule = [
        (80,  lambda: agent.inject_leg_loss(0)),
        (160, lambda: agent.inject_phase_noise(1.0)),
        (240, lambda: agent.inject_coupling_failure()),
        (320, lambda: agent.inject_leg_loss(4)),
    ]
    fi = 0
    for step in range(450):
        if fi < len(schedule) and step >= schedule[fi][0]:
            schedule[fi][1](); fi += 1
        agent.control()
    report = agent.get_fault_report()
    n_disabled = int(np.sum(agent.cpg.disabled_legs))
    if n_disabled >= 2 and report["faults_seen"] >= 3:
        return r.ok("All faults injected and logged; no healing.",
            faults_seen=report["faults_seen"],
            fault_types=report["fault_types"],
            disabled_legs=report["disabled_legs"],
            distance=f"{report['distance']:.1f}",
            coord=f"{report['coord_score']:.4f}")
    return r.fail("Fewer faults logged than expected.",
        faults_seen=report["faults_seen"], n_disabled=n_disabled)


def test_headless_pybeast() -> _TestResult:
    r = _TestResult("PyBeast++ headless smoke test")
    if not _PYBEAST_AVAILABLE:
        return r.ok("PyBeast++ not installed – skipped.")
    try:
        Config.GENERATIONS = 2; Config.POPULATION_SIZE = 4
        Config.ASSESSMENTS = 1; Config.TIMESTEPS = 30
        CPGHexapodSimulation()._run_simulation_no_render(parallel=False)
        return r.ok("Simulation ran without error.")
    except Exception as e:
        return r.fail(f"Raised: {e}")


def run_tests(verbose=True) -> bool:
    suite = [test_leg_loss_permanent, test_two_leg_loss,
             test_phase_drift_no_repair, test_coupling_failure_no_repair,
             test_full_fault_run, test_headless_pybeast]
    print("\n" + "═"*65)
    print("  Fault-Injection CPG Hexapod – Test Suite (No Healing)")
    print("═"*65)
    results = [fn() for fn in suite]
    if verbose:
        for res in results: print(str(res))
    passed = sum(1 for r in results if r.passed)
    print("─"*65)
    print(f"  Results: {passed}/{len(results)} passed")
    print("═"*65 + "\n")
    return passed == len(results)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_genome(path):
    if not os.path.exists(path): return None
    with open(path) as f: data = json.load(f)
    return np.array(data['genome'], dtype=np.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fault-Injection CPG Hexapod (No Self-Healing)")
    parser.add_argument("--test",       action="store_true")
    parser.add_argument("--evolve",     action="store_true")
    parser.add_argument("--generations",type=int, default=Config.GENERATIONS)
    parser.add_argument("--population", type=int, default=Config.POPULATION_SIZE)
    args = parser.parse_args()

    if args.test:
        sys.exit(0 if run_tests(verbose=True) else 1)
    elif args.evolve:
        if not _PYBEAST_AVAILABLE:
            print("PyBeast++ not found."); sys.exit(1)
        Config.GENERATIONS = args.generations
        Config.POPULATION_SIZE = args.population
        CPGHexapodSimulation()._run_simulation_no_render(parallel=False)
    else:
        run_tests(verbose=True)
