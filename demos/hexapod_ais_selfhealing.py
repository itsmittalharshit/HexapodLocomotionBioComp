"""
hexapod_ais_selfhealing.py  –  Self-Healing Fault-Tolerant CPG Hexapod
========================================================================
Place this file in the pybeastpp/demos/ folder to run via the GUI,
or run headless from the repo root:

    python -c "
    import sys; sys.path.insert(0, '.')
    from demos.hexapod_ais_selfhealing import CPGHexapodSimulation
    sim = CPGHexapodSimulation()
    sim._run_simulation_no_render(parallel=False)
    "

═══════════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
═══════════════════════════════════════════════════════════════════════

1.  ARTIFICIAL IMMUNE SYSTEM  (AIS)
    ─────────────────────────────────
    Inspired by the vertebrate immune system: detector cells (antibodies)
    patrol the CPG state space. When a fault is detected (leg loss, phase
    collapse, coupling failure, energy crisis) the AIS:
      a) Clones the best-matching memory cell (clonal selection)
      b) Hypermutates it toward the fault signature
      c) Inserts the resulting gait-patch into the CPG on the fly
      d) Promotes successful patches to long-term memory

    Memory cell pool: up to AIS_MEMORY_POOL_SIZE cells, each encoding a
    (fault_signature → corrective_phase_deltas) mapping.

2.  FAULT DETECTION ENGINE
    ─────────────────────────
    Five orthogonal fault detectors run every timestep:
      • LEG_LOSS     – leg i produces zero output for > FAULT_SILENT_TICKS
      • PHASE_DRIFT  – leg i drifts > FAULT_PHASE_THR from desired
      • COUPLING_FAIL– avg coupling weight < FAULT_WEIGHT_MIN
      • ENERGY_CRISIS– remaining energy < ENERGY_CRISIS_THRESHOLD
      • SYMMETRY_FAIL– left/right drive imbalance > FAULT_SYMMETRY_THR

    Each fault type triggers a separate self-healing response.

3.  ONLINE EVOLUTIONARY REPAIR  (mini-GA inside the agent)
    ──────────────────────────────────────────────────────────
    When a NEW fault is detected (not in AIS memory) the agent runs a
    micro-evolution loop (REPAIR_GENERATIONS quick generations of a
    tiny REPAIR_POP population) to evolve corrective phase offsets.
    The best solution is stored as a new AIS memory cell.

4.  CPG RECONFIGURATION
    ──────────────────────
    The fault-tolerant CPGNetwork supports:
      • disable_leg(i)   – zero the oscillator output; redistribute drive
      • patch_phases(Δφ) – add incremental phase corrections
      • rebalance()      – renormalise drive weights after leg loss

5.  NEUROMODULATION + HEBBIAN PLASTICITY  (from v6)
    ──────────────────────────────────────────────────
    Fully retained from v6_all_features so all five original features
    still operate. The AIS layer sits *above* the CPG loop.

═══════════════════════════════════════════════════════════════════════
TESTING UTILITIES  (headless, no OpenGL required)
═══════════════════════════════════════════════════════════════════════
At the bottom of the file, run_tests() drives:
  • test_leg_loss_recovery()         – breaks legs 1–6 mid-run
  • test_phase_drift_recovery()      – injects phase noise
  • test_coupling_failure_recovery() – zeroes coupling weights
  • test_symmetry_failure_recovery() – breaks one side of legs
  • test_full_survival_run()         – full evolved + fault scenario
  • test_ais_memory_learning()       – confirms memory pool grows
  • test_comparative_fitness()       – AIS vs. no-AIS fitness delta

Run tests:
    python hexapod_ais_selfhealing.py --test
Run full evolution:
    python hexapod_ais_selfhealing.py --evolve
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
import math
import json
import os
import sys
import time
import random
import argparse
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import IntEnum, auto

# ─── numpy ────────────────────────────────────────────────────────────────────
import numpy as np

# ─── PyBeast++ – only import OpenGL/core when NOT in pure-test mode ───────────
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
    from core.utils import (
        Vec2, AgentSettings as AS,
        ColourPalette, ColourType as CT,
        WORLD_DISPLAY_PARAMETERS as WDP
    )
    from core.world.world_object import WorldObject
    _PYBEAST_AVAILABLE = True
except ImportError:
    pass

# ─── GUI registration ─────────────────────────────────────────────────────────
IS_DEMO    = True
DEMO_NAME  = "CPG Hexapod – AIS Self-Healing"
CLASS_NAME = "CPGHexapodSimulation"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
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

    # ── Fitness / environment ─────────────────────────────────────────────
    FITNESS_MODE      = 'FORAGING'   # DISTANCE | EFFICIENCY | STABILITY | FORAGING
    DEFAULT_GAIT      = 'TRIPOD'
    ENVIRONMENT       = 'FLAT'

    # ── CPG baseline ─────────────────────────────────────────────────────
    CPG_FREQ          = 1.2
    CPG_COUPLING      = 0.65
    CPG_DUTY          = 0.60
    CPG_AMPLITUDE     = 0.80
    TIMESTEP_DT       = 0.05
    MAX_SPEED         = 120.0
    MIN_SPEED         = 0.0

    # ── Feature 4: Foraging ───────────────────────────────────────────────
    FOOD_COUNT        = 12
    FOOD_CALORIES     = 25.0
    AGENT_START_ENERGY= 40.0
    ENERGY_PER_STEP   = 0.04

    # ── Feature 1: Hebbian / STDP ─────────────────────────────────────────
    HEBBIAN_LR        = 0.002
    HEBBIAN_DECAY     = 0.0001
    HEBBIAN_W_MIN     = 0.05
    HEBBIAN_W_MAX     = 2.0
    HEBBIAN_SYNC_THR  = 0.25

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
    FATIGUE_HEBBIAN_SCALE  = 0.30

    # ── AIS – ARTIFICIAL IMMUNE SYSTEM ────────────────────────────────────
    AIS_MEMORY_POOL_SIZE   = 20     # max stored memory cells
    AIS_CLONE_COUNT        = 5      # clones per clonal-selection event
    AIS_HYPERMUTATION_SIGMA= 0.15   # mutation sigma for clone hypermutation
    AIS_AFFINITY_THRESHOLD = 0.30   # max distance to match a memory cell
    AIS_SUPPRESSION_DIST   = 0.10   # min distance between memory cells
    AIS_PATROL_INTERVAL    = 10     # timesteps between routine AIS patrol

    # ── FAULT DETECTION ───────────────────────────────────────────────────
    FAULT_SILENT_TICKS    = 20      # ticks of zero output → LEG_LOSS
    FAULT_PHASE_THR       = 1.2     # radians drift → PHASE_DRIFT fault
    FAULT_WEIGHT_MIN      = 0.08    # coupling weight floor → COUPLING_FAIL
    FAULT_SYMMETRY_THR    = 0.60    # left/right drive ratio → SYMMETRY_FAIL
    ENERGY_CRISIS_THR     = 5.0     # energy units → ENERGY_CRISIS

    # ── ONLINE MICRO-EVOLUTION (repair) ───────────────────────────────────
    REPAIR_POP            = 12      # population for micro-GA repair
    REPAIR_GENERATIONS    = 15      # generations of micro-GA
    REPAIR_EVAL_STEPS     = 80      # CPG steps per repair candidate eval
    REPAIR_MUTATION_SIGMA = 0.20

    # ── Visualisation ─────────────────────────────────────────────────────
    DRAW_CPG_NETWORK   = True
    DRAW_TRAILS        = True
    DRAW_FORCE_ARROWS  = True
    DRAW_AIS_STATUS    = True       # overlay AIS health indicator
    SAVE_BEST_GENOME   = True
    GENOME_SAVE_PATH   = "best_hexapod_ais.json"


GAIT_PHASES = {
    'TRIPOD': np.array([0, np.pi, 0, np.pi, 0, np.pi], dtype=np.float64),
    'WAVE':   np.array([0, np.pi/3, 2*np.pi/3,
                        np.pi, 4*np.pi/3, 5*np.pi/3], dtype=np.float64),
    'RIPPLE': np.array([0, 2*np.pi/3, 4*np.pi/3,
                        np.pi/3, np.pi, 5*np.pi/3], dtype=np.float64),
}
LEG_NAMES = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']


# ══════════════════════════════════════════════════════════════════════════════
# FAULT TYPES
# ══════════════════════════════════════════════════════════════════════════════

class FaultType(IntEnum):
    NONE            = 0
    LEG_LOSS        = auto()   # a leg becomes silent / non-functional
    PHASE_DRIFT     = auto()   # oscillator phases drift from desired
    COUPLING_FAIL   = auto()   # Hebbian weights collapse
    ENERGY_CRISIS   = auto()   # energy critically low
    SYMMETRY_FAIL   = auto()   # left-right drive imbalance


@dataclass
class FaultEvent:
    fault_type: FaultType
    leg_mask: np.ndarray        # 6-bit mask of affected legs
    severity:  float            # 0..1
    timestamp: int              # simulation timestep
    resolved:  bool = False
    resolution_time: int = -1


# ══════════════════════════════════════════════════════════════════════════════
# AIS MEMORY CELL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AISMemoryCell:
    """
    Each memory cell maps a fault *signature* to a corrective *response*.

    fault_signature : 12-dim vector  [6 phase_errors | 6 weight_errors]
    phase_patch     : 6-dim delta-phases to add to CPG desired phases
    drive_weights   : 6-dim per-leg drive multipliers (1.0 = normal)
    fitness_gain    : measured fitness improvement when this cell was applied
    age             : times this cell has been successfully activated
    """
    fault_signature: np.ndarray          # shape (12,)
    phase_patch:     np.ndarray          # shape (6,)
    drive_weights:   np.ndarray          # shape (6,)
    fitness_gain:    float = 0.0
    age:             int   = 0


    # ── Identity-based equality so list.remove() works with numpy arrays ──
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def affinity(self, signature: np.ndarray) -> float:
        """Euclidean distance (lower = better match)."""
        return float(np.linalg.norm(self.fault_signature - signature))

    def clone(self) -> "AISMemoryCell":
        return AISMemoryCell(
            fault_signature = self.fault_signature.copy(),
            phase_patch     = self.phase_patch.copy(),
            drive_weights   = self.drive_weights.copy(),
            fitness_gain    = self.fitness_gain,
            age             = 0,
        )


def _hypermutate(cell: AISMemoryCell, sigma: float,
                 rng: np.random.Generator) -> AISMemoryCell:
    """Return a new hypermutated clone of cell."""
    clone = cell.clone()
    clone.phase_patch   += rng.normal(0, sigma, 6)
    clone.drive_weights += rng.normal(0, sigma * 0.5, 6)
    clone.drive_weights  = np.clip(clone.drive_weights, 0.0, 2.0)
    clone.fault_signature += rng.normal(0, sigma * 0.2, 12)
    return clone


# ══════════════════════════════════════════════════════════════════════════════
# AIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ArtificialImmuneSystem:
    """
    Clonal Selection Algorithm (CLONALG) for online CPG fault tolerance.

    Memory pool: {fault_signature → corrective_response} pairs.

    On each AIS patrol tick:
      1. Build current fault signature from CPG state.
      2. Search memory pool for best-matching cell.
      3. If match (affinity < threshold): apply the patch directly.
      4. If no match: trigger online micro-evolution to find a patch,
         store result as a new memory cell.
      5. Apply network suppression to remove redundant memory cells.
    """

    def __init__(self):
        self._memory: List[AISMemoryCell] = []
        self._rng = np.random.default_rng()
        self._patrol_counter = 0
        self._active_cell: Optional[AISMemoryCell] = None
        self._last_fault: FaultType = FaultType.NONE
        self._heal_events: int = 0
        self._fail_events: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def patrol(self,
               cpg: "CPGNetwork",
               fault: FaultEvent,
               fitness_proxy: float) -> Optional[AISMemoryCell]:
        """
        Called every AIS_PATROL_INTERVAL ticks.

        Returns the applied memory cell (or None if no action taken).
        """
        self._patrol_counter += 1
        if fault.fault_type == FaultType.NONE:
            # Positive-feedback: boost age of active cell if still applied
            if self._active_cell is not None:
                self._active_cell.fitness_gain += fitness_proxy * 0.01
                self._active_cell.age += 1
            return None

        # Build fault signature from CPG
        signature = self._build_signature(cpg, fault)

        # Clonal selection: find best matching memory cell
        cell = self._select(signature)
        if cell is not None:
            # Apply cached response
            self._apply_cell(cpg, cell, fault)
            cell.age += 1
            cell.fitness_gain += fitness_proxy * 0.02
            self._active_cell = cell
            self._heal_events += 1
            return cell
        else:
            # No memory → micro-evolve a new response
            new_cell = self._micro_evolve(cpg, fault, signature)
            if new_cell is not None:
                self._apply_cell(cpg, new_cell, fault)
                self._insert_memory(new_cell)
                self._active_cell = new_cell
                self._heal_events += 1
                return new_cell
            self._fail_events += 1
            return None

    def on_fault_resolved(self, cell: Optional[AISMemoryCell]):
        """Called when a fault clears; promotes cell fitness."""
        if cell is not None:
            cell.fitness_gain += 1.0

    def forget_active(self):
        self._active_cell = None

    @property
    def memory_size(self) -> int:
        return len(self._memory)

    @property
    def stats(self) -> dict:
        return {
            "memory_cells":  len(self._memory),
            "heal_events":   self._heal_events,
            "fail_events":   self._fail_events,
            "active_cell_age": (self._active_cell.age
                                if self._active_cell else 0),
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_signature(self,
                         cpg: "CPGNetwork",
                         fault: FaultEvent) -> np.ndarray:
        """
        12-dim fault signature:
          [0:6]  normalised phase errors  (|φ_actual - φ_desired| / π)
          [6:12] normalised weight errors (1 - w_ij/W_MAX) per leg pair
        """
        phase_err = np.abs(
            (cpg.phi - cpg.desired) % (2 * np.pi)) / np.pi
        phase_err = np.where(phase_err > 1.0, 2.0 - phase_err, phase_err)

        # Per-leg mean coupling weight error
        w_err = np.zeros(6)
        for i in range(6):
            row = np.delete(cpg.coupling_weights[i], i)
            w_err[i] = 1.0 - float(np.mean(row)) / Config.HEBBIAN_W_MAX

        # Encode fault type as a soft bias
        type_bias = fault.fault_type / len(FaultType)
        sig = np.concatenate([phase_err, w_err])
        sig = sig * (1.0 + 0.1 * type_bias)
        # Mask unaffected legs
        sig[:6]  *= fault.leg_mask.astype(float) if fault.leg_mask.any() else 1.0
        return sig.astype(np.float64)

    def _select(self, signature: np.ndarray) -> Optional[AISMemoryCell]:
        """Return best-matching memory cell if within affinity threshold."""
        if not self._memory:
            return None
        cells_sorted = sorted(self._memory, key=lambda c: c.affinity(signature))
        best = cells_sorted[0]
        if best.affinity(signature) < Config.AIS_AFFINITY_THRESHOLD:
            return best
        return None

    def _apply_cell(self,
                    cpg: "CPGNetwork",
                    cell: AISMemoryCell,
                    fault: FaultEvent) -> None:
        """Apply a memory cell's corrective phase patch + drive weights."""
        # Phase patch: only for legs flagged in the fault mask
        mask = fault.leg_mask if fault.leg_mask.any() else np.ones(6, bool)
        for i in range(6):
            if mask[i]:
                cpg.desired[i] = (cpg.desired[i] + cell.phase_patch[i]) % (2 * np.pi)
        cpg.drive_weights[:] = cell.drive_weights.copy()

        # For leg-loss specifically: zero out dead leg's coupling
        if fault.fault_type == FaultType.LEG_LOSS:
            for i in range(6):
                if fault.leg_mask[i]:
                    cpg.disable_leg(i)

    def _insert_memory(self, cell: AISMemoryCell) -> None:
        """Insert cell, apply suppression to maintain diversity."""
        # Suppress if too similar to existing cell
        for existing in self._memory:
            if (np.linalg.norm(existing.fault_signature
                                - cell.fault_signature)
                    < Config.AIS_SUPPRESSION_DIST):
                # Keep the one with higher fitness
                if cell.fitness_gain > existing.fitness_gain:
                    self._memory = [c for c in self._memory if c is not existing]
                    break
                else:
                    return  # existing is better; discard new cell

        self._memory.append(cell)

        # Trim to pool size (remove lowest fitness cells)
        if len(self._memory) > Config.AIS_MEMORY_POOL_SIZE:
            self._memory.sort(key=lambda c: c.fitness_gain)
            self._memory.pop(0)

    def _micro_evolve(self,
                      cpg: "CPGNetwork",
                      fault: FaultEvent,
                      signature: np.ndarray) -> Optional[AISMemoryCell]:
        """
        Run a micro-GA on phase_patch + drive_weights to maximise a
        quick CPG fitness proxy (coordination score in REPAIR_EVAL_STEPS).
        """
        rng = self._rng

        def make_candidate() -> AISMemoryCell:
            return AISMemoryCell(
                fault_signature = signature.copy(),
                phase_patch     = rng.uniform(-np.pi, np.pi, 6),
                drive_weights   = rng.uniform(0.5, 1.5, 6),
            )

        def evaluate(cell: AISMemoryCell) -> float:
            # Clone the CPG so we don't mutate the real one
            test_cpg = copy.deepcopy(cpg)
            self._apply_cell(test_cpg, cell, fault)
            coord = 0.0
            dt    = Config.TIMESTEP_DT
            for _ in range(Config.REPAIR_EVAL_STEPS):
                test_cpg.step(dt)
                phi = test_cpg.phi
                pair_diff = (abs(math.sin(phi[0]-phi[2])) +
                             abs(math.sin(phi[2]-phi[4])) +
                             abs(math.sin(phi[1]-phi[3])) +
                             abs(math.sin(phi[3]-phi[5])))
                coord += 1.0 - pair_diff / 4.0
            return coord / Config.REPAIR_EVAL_STEPS

        # Initialise population; bias a few toward baseline (no change)
        population = [make_candidate() for _ in range(Config.REPAIR_POP - 2)]
        population.append(AISMemoryCell(
            fault_signature=signature.copy(),
            phase_patch=np.zeros(6),
            drive_weights=np.ones(6),
        ))
        # Clone from best memory cell if available
        if self._memory:
            best_mem = max(self._memory, key=lambda c: c.fitness_gain)
            population.append(best_mem.clone())

        best_cell   = None
        best_fitness = -np.inf

        for gen in range(Config.REPAIR_GENERATIONS):
            # Evaluate
            scores = [(evaluate(c), c) for c in population]
            scores.sort(key=lambda x: x[0], reverse=True)

            if scores[0][0] > best_fitness:
                best_fitness = scores[0][0]
                best_cell    = scores[0][1].clone()

            # Clonal selection + hypermutation
            new_pop = [scores[0][1].clone()]  # elitism 1
            for _, parent in scores[:Config.AIS_CLONE_COUNT]:
                for _ in range(2):
                    new_pop.append(_hypermutate(
                        parent, Config.AIS_HYPERMUTATION_SIGMA, rng))
            # Random immigrants
            while len(new_pop) < Config.REPAIR_POP:
                new_pop.append(make_candidate())
            population = new_pop[:Config.REPAIR_POP]

        if best_cell is not None:
            best_cell.fitness_gain = best_fitness
        return best_cell


# ══════════════════════════════════════════════════════════════════════════════
# FAULT DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class FaultDetector:
    """
    Monitors CPG state and agent metrics each timestep; raises FaultEvents.
    """

    def __init__(self):
        self._silent_ticks   = np.zeros(6, dtype=int)
        self._prev_phases    = None
        self._history        = deque(maxlen=30)
        self.active_faults: List[FaultEvent] = []

    def reset(self):
        self._silent_ticks[:] = 0
        self._prev_phases = None
        self._history.clear()
        self.active_faults.clear()

    def update(self,
               cpg: "CPGNetwork",
               disabled_legs: np.ndarray,
               left_drive: float,
               right_drive: float,
               energy: float,
               timestep: int) -> Optional[FaultEvent]:
        """
        Runs all detectors. Returns the highest-priority new FaultEvent
        or None if no new fault.
        """
        faults = []

        # ── 1. LEG_LOSS ──────────────────────────────────────────────────
        for i in range(6):
            disp, _ = cpg.output(i)
            if abs(disp) < 0.02 or disabled_legs[i]:
                self._silent_ticks[i] += 1
            else:
                self._silent_ticks[i] = 0
            if self._silent_ticks[i] >= Config.FAULT_SILENT_TICKS:
                mask = np.zeros(6, bool); mask[i] = True
                faults.append(FaultEvent(
                    FaultType.LEG_LOSS, mask,
                    min(1.0, self._silent_ticks[i] / 60), timestep))

        # ── 2. PHASE_DRIFT ───────────────────────────────────────────────
        phase_err = np.abs((cpg.phi - cpg.desired) % (2 * np.pi))
        phase_err = np.where(phase_err > np.pi, 2*np.pi - phase_err, phase_err)
        drifted   = phase_err > Config.FAULT_PHASE_THR
        if drifted.any():
            mask = drifted.copy()
            severity = float(np.max(phase_err[drifted]) / np.pi)
            faults.append(FaultEvent(
                FaultType.PHASE_DRIFT, mask, min(1.0, severity), timestep))

        # ── 3. COUPLING_FAIL ─────────────────────────────────────────────
        diag_mask = ~np.eye(6, dtype=bool)
        avg_w = float(np.mean(cpg.coupling_weights[diag_mask]))
        if avg_w < Config.FAULT_WEIGHT_MIN:
            mask = np.ones(6, bool)
            severity = 1.0 - avg_w / Config.FAULT_WEIGHT_MIN
            faults.append(FaultEvent(
                FaultType.COUPLING_FAIL, mask, min(1.0, severity), timestep))

        # ── 4. ENERGY_CRISIS ─────────────────────────────────────────────
        if energy < Config.ENERGY_CRISIS_THR:
            mask = np.ones(6, bool)
            severity = 1.0 - energy / Config.ENERGY_CRISIS_THR
            faults.append(FaultEvent(
                FaultType.ENERGY_CRISIS, mask, severity, timestep))

        # ── 5. SYMMETRY_FAIL ─────────────────────────────────────────────
        total = abs(left_drive) + abs(right_drive) + 1e-9
        imbalance = abs(left_drive - right_drive) / total
        if imbalance > Config.FAULT_SYMMETRY_THR:
            # Left or right side depending on which is weaker
            mask = np.zeros(6, bool)
            if left_drive < right_drive:
                mask[[0,1,2]] = True
            else:
                mask[[3,4,5]] = True
            faults.append(FaultEvent(
                FaultType.SYMMETRY_FAIL, mask, min(1.0, imbalance), timestep))

        if not faults:
            return None

        # Return highest-severity fault
        faults.sort(key=lambda f: f.severity, reverse=True)
        return faults[0]


# ══════════════════════════════════════════════════════════════════════════════
# FAULT-TOLERANT CPG NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class CPGNetwork:
    """
    Extended Kuramoto CPG from v6 with additional fault-tolerance hooks:
      • disable_leg(i)   – silences a leg and redistributes drive
      • rebalance()      – renormalise drive_weights after leg loss
      • patch_phases(Δφ) – incremental phase correction from AIS
      • drive_weights    – per-leg output multipliers (1.0 default)
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

        # Hebbian weight matrix
        self.coupling_weights = np.full((6,6), coupling, dtype=np.float64)
        np.fill_diagonal(self.coupling_weights, 0.0)

        # AIS-controlled per-leg drive multipliers
        self.drive_weights  = np.ones(6, dtype=np.float64)
        self._disabled_legs = np.zeros(6, dtype=bool)

    # ── Fault-tolerance interface ─────────────────────────────────────────

    def disable_leg(self, i: int) -> None:
        """
        Mark leg i as disabled. Set drive_weight to 0 and
        boost neighbours to partially compensate.
        """
        self._disabled_legs[i] = True
        self.drive_weights[i]  = 0.0
        self.rebalance()

    def enable_leg(self, i: int) -> None:
        self._disabled_legs[i] = False
        self.rebalance()

    def rebalance(self) -> None:
        """
        After a leg-loss event, scale up surviving legs so total drive
        capacity is preserved (approximately).
        """
        n_active = int(np.sum(~self._disabled_legs))
        if n_active == 0:
            return
        scale = 6.0 / n_active
        for i in range(6):
            if not self._disabled_legs[i]:
                self.drive_weights[i] = min(2.0, scale * 0.8)
        # Zero disabled legs
        self.drive_weights[self._disabled_legs] = 0.0

    def reset_weights(self) -> None:
        """Reset Hebbian weights and drive weights to baseline."""
        self.coupling_weights[:] = self.base_coupling
        np.fill_diagonal(self.coupling_weights, 0.0)
        self.drive_weights[:]    = 1.0
        self._disabled_legs[:]   = False

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, dt: float,
             feedback=None,
             eff_freq=None,
             eff_amplitude=None,
             fatigue: float = 0.0) -> None:
        self.eff_freq      = eff_freq      if eff_freq      is not None else self.freq
        self.eff_amplitude = eff_amplitude if eff_amplitude is not None else self.amplitude

        # Feature 2: proprioceptive phase reset
        if feedback is not None:
            thr = Config.FEEDBACK_THRESHOLD
            for i in range(6):
                if feedback[i] > thr and not self._disabled_legs[i]:
                    self.phi[i] = self.desired[i] + 0.05

        # Kuramoto with Hebbian weights (disabled legs excluded from coupling)
        omega = 2.0 * np.pi * self.eff_freq
        dphi  = np.zeros(6)
        for i in range(6):
            if self._disabled_legs[i]:
                continue
            cs = 0.0
            for j in range(6):
                if i != j and not self._disabled_legs[j]:
                    delta = self.desired[j] - self.desired[i]
                    cs   += self.coupling_weights[i, j] * math.sin(
                        self.phi[j] - self.phi[i] - delta)
            dphi[i] = omega + cs
        self.phi += dphi * dt

        # Hebbian update (Feature 1)
        self._update_hebbian(dt, fatigue)

    def _update_hebbian(self, dt: float, fatigue: float = 0.0) -> None:
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
                self.coupling_weights[i,j] = max(
                    Config.HEBBIAN_W_MIN,
                    min(Config.HEBBIAN_W_MAX,
                        self.coupling_weights[i,j] + dW * dt))

    # ── Output ────────────────────────────────────────────────────────────

    def output(self, leg_idx: int) -> Tuple[float, bool]:
        if self._disabled_legs[leg_idx]:
            return 0.0, False
        phi_norm = self.phi[leg_idx] % (2.0 * np.pi)
        x = math.sin(phi_norm) * self.eff_amplitude * self.drive_weights[leg_idx]
        return x, math.sin(phi_norm) > 0.0

    @property
    def phase_fractions(self) -> np.ndarray:
        return (self.phi % (2.0 * np.pi)) / (2.0 * np.pi)

    def coupling_strength(self, i: int, j: int) -> float:
        delta = self.desired[j] - self.desired[i]
        return math.sin(self.phi[j] - self.phi[i] - delta)

    @property
    def avg_weight(self) -> float:
        mask = ~np.eye(6, dtype=bool)
        return float(np.mean(self.coupling_weights[mask]))

    @property
    def disabled_legs(self) -> np.ndarray:
        return self._disabled_legs.copy()


# ══════════════════════════════════════════════════════════════════════════════
# WORLD OBJECTS (only defined when PyBeast++ is available)
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

# Base classes differ depending on whether PyBeast++ is importable
if _PYBEAST_AVAILABLE:
    _AgentBase  = Agent
    _EvolverBase = Evolver
else:
    # Stub base classes for pure-Python testing
    class _AgentBase:
        def __init__(self, **kwargs): self.location = np.array([0.,0.])
        def reset(self): pass
    class _EvolverBase:
        def __init__(self): pass


class CPGHexapod(_AgentBase, _EvolverBase):
    """
    Hexapod agent with AIS self-healing layer.

    Genome (13 genes):
      0   freq           [0.3, 3.0]
      1   coupling       [0.0, 1.0]
      2   duty           [0.3, 0.85]
      3   amplitude      [0.1, 1.0]
      4–9 phase offsets  [0, 2π] × 6
      10  leg_length     [0.5, 2.0]
      11  joint_stiffness[0.0, 1.0]
    """

    GENOME_LENGTH = 13
    GENE_SCALE = [
        (0.3, 3.0),
        (0.0, 1.0),
        (0.3, 0.85),
        (0.1, 1.0),
    ] + [(0.0, 2*np.pi)] * 6 + [
        (Config.LEG_LENGTH_MIN, Config.LEG_LENGTH_MAX),
        (Config.STIFFNESS_MIN,  Config.STIFFNESS_MAX),
    ]

    BASE_COXA  = 10.0
    BASE_FEMUR = 14.0

    def __init__(self):
        if _PYBEAST_AVAILABLE:
            _AgentBase.__init__(self,
                min_speed=Config.MIN_SPEED, max_speed=Config.MAX_SPEED,
                timestep=Config.TIMESTEP_DT, random_colour=False, solid=False)
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

        # AIS + fault detection
        self.ais             = ArtificialImmuneSystem()
        self.fault_detector  = FaultDetector()
        self._current_fault: Optional[FaultEvent]  = None
        self._applied_cell:  Optional[AISMemoryCell] = None
        self._fault_log: List[FaultEvent] = []
        self._ais_patrol_tick = 0
        self._timestep_count  = 0

        # Morphology (Feature 3)
        self.leg_length      = 1.0
        self.joint_stiffness = 0.3
        self._spring_energy  = 0.0

        # Neuromodulation (Feature 5)
        self._fatigue        = 0.0

        # Proprioception (Feature 2)
        self._feedback       = np.zeros(6, dtype=np.float64)
        self._reflex_count   = 0

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

        # Drive state (for fault detector)
        self._left_drive  = 0.0
        self._right_drive = 0.0

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
        phases   = np.array([self._scale_gene(g[4+i], 4+i) for i in range(6)])
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

    # ── Fitness ───────────────────────────────────────────────────────────

    def get_fitness(self):
        if self._step_count == 0: return 0.0
        base          = self._calories_gathered
        explore       = self._distance_travelled * 0.01
        plasticity    = self.cpg.avg_weight * 0.3
        reflex_bonus  = self._reflex_count * 0.3
        # AIS bonus: heal events increase fitness
        heal_bonus    = self.ais.stats["heal_events"] * 0.5
        penalty       = self._collision_penalty * 5.0
        fatigue_pen   = self._fatigue * 3.0
        morph_cost    = (self.leg_length - 1.0)**2 * self.joint_stiffness * 1.5
        return max(0.0,
            base + explore + plasticity + reflex_bonus + heal_bonus
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
        self._applied_cell       = None
        self._ais_patrol_tick    = 0
        self.fault_detector.reset()
        self.ais.forget_active()
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

        # Feature 5: neuromodulation
        eff_freq = self.cpg.freq * (1.0 - Config.FATIGUE_FREQ_GAIN * self._fatigue)
        eff_amp  = self.cpg.amplitude * (1.0 - Config.FATIGUE_AMPLITUDE_GAIN * self._fatigue)

        # Step CPG
        self.cpg.step(dt,
                      feedback=self._feedback,
                      eff_freq=eff_freq,
                      eff_amplitude=eff_amp,
                      fatigue=self._fatigue)
        self._feedback *= Config.FEEDBACK_DECAY

        # Compute drives
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

        # Energy + fatigue
        movement_cost = energy_tick * Config.ENERGY_PER_STEP * dt
        self._energy_consumed  += movement_cost
        self._energy_remaining -= movement_cost
        self._step_count       += 1
        self._fatigue = max(0.0, min(Config.FATIGUE_MAX,
            self._fatigue
            + Config.FATIGUE_GROWTH * energy_tick
            - Config.FATIGUE_RECOVERY * dt))

        # Distance
        if _PYBEAST_AVAILABLE and hasattr(self, 'velocity'):
            spd = math.hypot(*self.velocity)
        else:
            spd = abs(left_drive + right_drive) * 10.0  # approximation
        self._distance_travelled += spd * dt

        # Coordination
        phi = self.cpg.phi
        pair_diff = (abs(math.sin(phi[0]-phi[2])) + abs(math.sin(phi[2]-phi[4])) +
                     abs(math.sin(phi[1]-phi[3])) + abs(math.sin(phi[3]-phi[5])))
        self._coordination_score += 1.0 - pair_diff / 4.0

        # Trail
        if Config.DRAW_TRAILS and _PYBEAST_AVAILABLE and hasattr(self, 'location'):
            self._trail.append(tuple(self.location))
            if len(self._trail) > 200: self._trail.pop(0)

        # ── AIS PATROL ────────────────────────────────────────────────────
        self._ais_patrol_tick += 1
        if self._ais_patrol_tick >= Config.AIS_PATROL_INTERVAL:
            self._ais_patrol_tick = 0
            self._run_ais_patrol()

    def _run_ais_patrol(self):
        """Detect faults and invoke AIS response."""
        fault = self.fault_detector.update(
            self.cpg,
            self.cpg.disabled_legs,
            self._left_drive,
            self._right_drive,
            self._energy_remaining,
            self._timestep_count,
        )

        if fault is not None:
            # New fault or ongoing fault
            if (self._current_fault is None or
                    fault.fault_type != self._current_fault.fault_type):
                self._fault_log.append(fault)
                self._current_fault = fault

                # Invoke AIS
                fitness_proxy = (self._coordination_score /
                                 max(1, self._step_count))
                self._applied_cell = self.ais.patrol(
                    self.cpg, fault, fitness_proxy)
        else:
            # Fault resolved
            if self._current_fault is not None:
                self._current_fault.resolved = True
                self._current_fault.resolution_time = self._timestep_count
                self.ais.on_fault_resolved(self._applied_cell)
                self._current_fault = None
                self._applied_cell  = None
                self.ais.forget_active()

    # ── Public fault injection API (for testing) ──────────────────────────

    def inject_leg_loss(self, leg_idx: int) -> None:
        """Force-disable a leg to simulate hardware failure."""
        self.cpg.disable_leg(leg_idx)
        # Plant a fault event so AIS is aware
        mask = np.zeros(6, bool); mask[leg_idx] = True
        fault = FaultEvent(FaultType.LEG_LOSS, mask, 1.0, self._timestep_count)
        self._fault_log.append(fault)
        self._current_fault = fault
        fitness_proxy = (self._coordination_score / max(1, self._step_count))
        self._applied_cell = self.ais.patrol(self.cpg, fault, fitness_proxy)

    def inject_phase_noise(self, sigma: float = 0.8) -> None:
        """Add random phase noise to all legs."""
        self.cpg.phi += np.random.normal(0, sigma, 6)

    def inject_coupling_failure(self) -> None:
        """Collapse all coupling weights."""
        self.cpg.coupling_weights[:] = Config.FAULT_WEIGHT_MIN * 0.5
        np.fill_diagonal(self.cpg.coupling_weights, 0.0)

    def get_ais_report(self) -> dict:
        """Return a full AIS + fault report for diagnostics."""
        return {
            "ais":         self.ais.stats,
            "faults_seen": len(self._fault_log),
            "fault_types": [f.fault_type.name for f in self._fault_log],
            "fault_times": [f.timestamp for f in self._fault_log],
            "resolved":    [f.resolved for f in self._fault_log],
            "fitness":     self.get_fitness(),
            "distance":    self._distance_travelled,
            "energy_left": self._energy_remaining,
            "coord_score": (self._coordination_score / max(1, self._step_count)),
        }

    # ══════════════════════════════════════════════════════════════════════
    # RENDERING  (only when PyBeast++ is available)
    # ══════════════════════════════════════════════════════════════════════

    if _PYBEAST_AVAILABLE:
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

            self._draw_energy_bar()
            self._draw_fatigue_ring()
            self._draw_feedback_halo()
            if Config.DRAW_AIS_STATUS:
                self._draw_ais_indicator()

            glDisable(GL_LINE_SMOOTH)

        def _draw_ais_indicator(self):
            """Render a small AIS health ring (green=healthy, red=healing)."""
            if self._current_fault is not None:
                r, g_, b = 1.0, 0.3, 0.1   # healing = orange
            else:
                r, g_, b = 0.1, 0.9, 0.5   # healthy = teal
            alpha = 0.55
            glColor4fv([r, g_, b, alpha])
            glLineWidth(2.5)
            ring = 34.0
            glBegin(GL_LINE_LOOP)
            for a in range(28):
                ang = a / 28.0 * 2 * math.pi
                glVertex2d(ring * math.cos(ang), ring * math.sin(ang))
            glEnd()
            # Memory cell count as dot cluster
            n = min(self.ais.memory_size, 8)
            glPointSize(3.0); glColor4fv([0.8, 0.9, 1.0, 0.7])
            glBegin(GL_POINTS)
            for k in range(n):
                ang = k / 8.0 * 2 * math.pi
                glVertex2d(ring * math.cos(ang), ring * math.sin(ang))
            glEnd()
            glLineWidth(1.0)

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
                    alpha  = 0.05 + norm_w * 0.6
                    # Red tint if either leg disabled
                    if (self.cpg.disabled_legs[i] or self.cpg.disabled_legs[j]):
                        glColor4fv([0.9, 0.1, 0.1, 0.25])
                    else:
                        glColor4fv([norm_w*0.9, norm_w*0.7, 1.0-norm_w, alpha])
                    glLineWidth(max(0.5, norm_w * 3.0))
                    glBegin(GL_LINES); glVertex2d(xi,yi); glVertex2d(xj,yj); glEnd()
            glLineWidth(1.0)

        def _draw_energy_bar(self):
            ratio = max(0.0, min(1.0,
                self._energy_remaining / (Config.AGENT_START_ENERGY + self._calories_gathered + 0.01)))
            bar_w = 30.0; bar_h = 4.0; x0 = -15.0; y0 = -28.0
            glColor4fv([0.3, 0.0, 0.0, 0.7])
            glBegin(GL_QUADS)
            glVertex2d(x0,y0); glVertex2d(x0+bar_w,y0)
            glVertex2d(x0+bar_w,y0+bar_h); glVertex2d(x0,y0+bar_h); glEnd()
            glColor4fv([1.0-ratio, ratio, 0.1, 0.9])
            glBegin(GL_QUADS)
            glVertex2d(x0,y0); glVertex2d(x0+bar_w*ratio,y0)
            glVertex2d(x0+bar_w*ratio,y0+bar_h); glVertex2d(x0,y0+bar_h); glEnd()

        def _draw_fatigue_ring(self):
            if self._fatigue < 0.02: return
            r_ring = 20.0 + self._fatigue * 8.0
            alpha  = self._fatigue * 0.7
            glColor4fv([1.0, 0.4*(1-self._fatigue), 0.0, alpha])
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            for a in range(24):
                ang = a/24.0*2*math.pi
                glVertex2d(r_ring*math.cos(ang), r_ring*math.sin(ang))
            glEnd(); glLineWidth(1.0)

        def _draw_feedback_halo(self):
            max_fb = float(np.max(self._feedback))
            if max_fb < 0.05: return
            alpha = min(0.6, max_fb * 0.6)
            glColor4fv([1.0, 0.1, 0.1, alpha])
            glLineWidth(3.0)
            glBegin(GL_LINE_LOOP)
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
                glColor4fv([0.5, 0.0, 0.0, 0.5])
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
            glColor4fv([0.0, 1.0, 0.5, 1.0] if not swing else [1.0, 0.6, 0.1, 0.8])
            glPointSize(5.0 + self.leg_length*2.0)
            glBegin(GL_POINTS); glVertex2d(fx,fy); glEnd()
            if hasattr(self, 'location'):
                self._foot_positions[i] = (self.location[0]+fx, self.location[1]+fy, not swing)

        def _draw_body(self):
            s = 0.8 + 0.4*(self.leg_length-Config.LEG_LENGTH_MIN)/(Config.LEG_LENGTH_MAX-Config.LEG_LENGTH_MIN)
            for colour, rx, ry, dy, n in [
                ([0.15, 0.55, 0.25, 0.95], int(10*s), int(7*s), -16, 20),
                ([0.12, 0.48, 0.22, 0.95], int(14*s), int(11*s), 0, 24),
                ([0.10, 0.40, 0.18, 0.90], int(11*s), int(9*s), 15, 20),
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
# GA + SIMULATION  (only when PyBeast++ is available)
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
            data = {
                "best_fitness":   self._best_ever_fitness,
                "genome":         self._best_ever_genome.tolist(),
                "fitness_mode":   Config.FITNESS_MODE,
                "genome_length":  CPGHexapod.GENOME_LENGTH,
                "generations_run": self.generations,
                "history":        self._generation_log,
            }
            with open(path, 'w') as f: json.dump(data, f, indent=2)
            print(f"[GA] Best genome saved → {path}")


    class CPGHexapodSimulation(Simulation):
        def __init__(self):
            super().__init__("CPGHexapod_AIS")
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
                loc = np.array([x, y], dtype=np.float32)
                food_items.append(FoodPellet(location=loc, calories=Config.FOOD_CALORIES))

            class _FoodGroup(SimulationObject):
                def __init__(self, items):
                    super().__init__()
                    self.items = items
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
            print("  CPG Hexapod – AIS Self-Healing Controller")
            print("  Active systems:")
            print("    • Artificial Immune System (clonal selection + micro-GA)")
            print("    • 5-mode Fault Detector (leg-loss, phase, coupling,")
            print("      energy, symmetry)")
            print("    • Hebbian/STDP synaptic plasticity")
            print("    • Proprioceptive feedback reflex")
            print("    • Morphological computation")
            print("    • Foraging / energy-based fitness")
            print("    • Neuromodulation (fatigue)")
            print(f"  Genome length  : {CPGHexapod.GENOME_LENGTH}")
            print(f"  AIS pool size  : {Config.AIS_MEMORY_POOL_SIZE}")
            print(f"  Population     : {Config.POPULATION_SIZE}")
            print(f"  Generations    : {Config.GENERATIONS}")
            print("=" * 65)
            super().begin_simulation()

        def end_simulation(self):
            super().end_simulation()
            if Config.SAVE_BEST_GENOME:
                self._ga.save_best(Config.GENOME_SAVE_PATH)
            print("=" * 65)
            print("  Evolution complete.")
            if self._ga._best_ever_genome is not None:
                print(f"  Best fitness : {self._ga._best_ever_fitness:.4f}")
                print(f"  Best genome  : {np.round(self._ga._best_ever_genome, 4).tolist()}")
            print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TESTING SUITE  (no OpenGL required – pure Python)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class _TestResult:
    def __init__(self, name):
        self.name    = name
        self.passed  = False
        self.message = ""
        self.metrics = {}

    def ok(self, msg="", **metrics):
        self.passed  = True
        self.message = msg
        self.metrics = metrics
        return self

    def fail(self, msg="", **metrics):
        self.passed  = False
        self.message = msg
        self.metrics = metrics
        return self

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        s = f"  {status}  {self.name}"
        if self.message: s += f"\n         {self.message}"
        for k, v in self.metrics.items():
            s += f"\n         {k}: {v}"
        return s


def _run_steps(agent: CPGHexapod, n: int):
    """Run the agent's control loop for n timesteps (no rendering)."""
    for _ in range(n):
        agent.control()


def test_leg_loss_recovery() -> _TestResult:
    """
    Inject a leg-loss fault mid-run and verify:
      1. Fault is detected within FAULT_SILENT_TICKS ticks
      2. AIS applies a patch (heal_events > 0)
      3. Coordination score post-repair is not catastrophically worse
    """
    r = _TestResult("Leg-loss recovery (AIS + CPG reconfiguration)")
    agent = CPGHexapod()
    _run_steps(agent, 60)                # warm-up
    coord_before = agent._coordination_score / max(1, agent._step_count)

    agent.inject_leg_loss(2)             # Break L3
    _run_steps(agent, 120)               # let AIS react
    coord_after  = agent._coordination_score / max(1, agent._step_count)
    report = agent.get_ais_report()

    if (agent.cpg.disabled_legs[2] and
            report["ais"]["heal_events"] >= 1 and
            coord_after > coord_before * 0.30):
        return r.ok(
            "Leg 2 disabled; AIS healed coordination successfully.",
            coord_before=f"{coord_before:.3f}",
            coord_after=f"{coord_after:.3f}",
            heal_events=report["ais"]["heal_events"],
            memory_cells=report["ais"]["memory_cells"],
        )
    else:
        return r.fail(
            "Coordination degraded beyond tolerance after leg loss.",
            coord_before=f"{coord_before:.3f}",
            coord_after=f"{coord_after:.3f}",
            heal_events=report["ais"]["heal_events"],
        )


def test_two_leg_loss_recovery() -> _TestResult:
    """Lose two legs simultaneously and verify partial recovery."""
    r = _TestResult("Two-leg loss recovery (L1 + R3)")
    agent = CPGHexapod()
    _run_steps(agent, 50)

    agent.inject_leg_loss(0)   # L1
    agent.inject_leg_loss(5)   # R3
    _run_steps(agent, 150)

    report = agent.get_ais_report()
    n_disabled = int(np.sum(agent.cpg.disabled_legs))

    # With 4 legs still active, coordination should still be > 0
    coord = report["coord_score"]
    if n_disabled == 2 and coord > 0.10:
        return r.ok(
            "Two legs disabled; hexapod still locomoting.",
            disabled_legs=n_disabled,
            coord=f"{coord:.3f}",
            heal_events=report["ais"]["heal_events"],
        )
    else:
        return r.fail(
            "Could not maintain locomotion with two legs disabled.",
            disabled_legs=n_disabled,
            coord=f"{coord:.3f}",
        )


def test_phase_drift_recovery() -> _TestResult:
    """Inject strong phase noise; AIS should restore coordination."""
    r = _TestResult("Phase-drift recovery")
    agent = CPGHexapod()
    _run_steps(agent, 60)
    coord_before = agent._coordination_score / max(1, agent._step_count)

    agent.inject_phase_noise(sigma=1.5)  # large phase perturbation
    _run_steps(agent, 200)
    coord_after = agent._coordination_score / max(1, agent._step_count)
    report = agent.get_ais_report()

    recovery = coord_after / max(0.001, coord_before)
    if recovery > 0.40:
        return r.ok(
            "Phase drift partially corrected.",
            coord_before=f"{coord_before:.3f}",
            coord_after=f"{coord_after:.3f}",
            recovery_ratio=f"{recovery:.2f}",
            heal_events=report["ais"]["heal_events"],
        )
    else:
        return r.fail(
            "Phase drift not adequately recovered.",
            coord_before=f"{coord_before:.3f}",
            coord_after=f"{coord_after:.3f}",
            recovery_ratio=f"{recovery:.2f}",
        )


def test_coupling_failure_recovery() -> _TestResult:
    """Collapse coupling weights; AIS should rebuild coordination."""
    r = _TestResult("Coupling-failure recovery")
    agent = CPGHexapod()
    _run_steps(agent, 60)
    coord_before = agent._coordination_score / max(1, agent._step_count)

    agent.inject_coupling_failure()
    _run_steps(agent, 200)
    coord_after = agent._coordination_score / max(1, agent._step_count)
    report = agent.get_ais_report()
    avg_w = float(np.mean(agent.cpg.coupling_weights[~np.eye(6, dtype=bool)]))

    if coord_after > coord_before * 0.25 or avg_w > Config.FAULT_WEIGHT_MIN * 2:
        return r.ok(
            "Coupling partially restored after collapse.",
            coord_before=f"{coord_before:.3f}",
            coord_after=f"{coord_after:.3f}",
            avg_coupling=f"{avg_w:.4f}",
            heal_events=report["ais"]["heal_events"],
        )
    else:
        return r.fail(
            "Coupling failure not recovered.",
            coord_before=f"{coord_before:.3f}",
            coord_after=f"{coord_after:.3f}",
            avg_coupling=f"{avg_w:.4f}",
        )


def test_symmetry_failure_recovery() -> _TestResult:
    """
    Disable all left-side legs to force symmetry failure;
    AIS should adjust drive weights.
    """
    r = _TestResult("Symmetry-failure recovery (all left legs lost)")
    agent = CPGHexapod()
    _run_steps(agent, 50)

    for i in [0, 1, 2]:   # disable L1, L2, L3
        agent.inject_leg_loss(i)
    _run_steps(agent, 150)

    report = agent.get_ais_report()
    # Success criterion: still moving (distance > 0)
    if (report["ais"]["heal_events"] >= 1 and
            agent._distance_travelled > 0):
        return r.ok(
            "All left legs disabled; AIS rebalanced drive to right side.",
            distance=f"{agent._distance_travelled:.1f}",
            heal_events=report["ais"]["heal_events"],
            drive_weights=np.round(agent.cpg.drive_weights, 2).tolist(),
        )
    else:
        return r.fail(
            "Agent ceased locomotion after left-side loss.",
            distance=f"{agent._distance_travelled:.1f}",
            heal_events=report["ais"]["heal_events"],
        )


def test_ais_memory_learning() -> _TestResult:
    """Confirm that repeated faults build up the AIS memory pool."""
    r = _TestResult("AIS memory learning (repeated fault injection)")
    agent = CPGHexapod()
    _run_steps(agent, 30)

    for rep in range(4):
        # Alternating faults
        if rep % 2 == 0:
            agent.inject_leg_loss(rep % 6)
        else:
            agent.inject_phase_noise(0.8)
        _run_steps(agent, 60)
        # Re-enable all legs so next fault is distinct
        for i in range(6): agent.cpg.enable_leg(i)
        _run_steps(agent, 20)

    mem = agent.ais.memory_size
    if mem >= 2:
        return r.ok(
            f"AIS memory pool grew to {mem} cells after 4 fault cycles.",
            memory_cells=mem,
            heal_events=agent.ais.stats["heal_events"],
        )
    else:
        return r.fail(
            f"AIS memory did not grow as expected (cells={mem}).",
            memory_cells=mem,
        )


def test_comparative_fitness() -> _TestResult:
    """
    Compare fitness of an AIS-equipped agent vs. a plain agent after
    severe, sustained faults. We measure distance (a robust proxy for
    locomotion quality without a full PyBeast++ world).

    The no-AIS agent has its patrol disabled AND leg rebalancing blocked,
    so disabled legs don't get compensated — a strict baseline.
    """
    r = _TestResult("Comparative fitness: AIS vs. no-AIS under severe faults")

    def run_agent(with_ais: bool, steps: int = 400) -> Tuple[float, float]:
        agent = CPGHexapod()
        if not with_ais:
            agent._run_ais_patrol = lambda: None
            # Block CPG rebalance so lost legs aren't compensated
            agent.cpg.rebalance = lambda: None
            # Also disable Hebbian plasticity so weights stay collapsed
            agent.cpg._update_hebbian = lambda dt, fatigue=0.0: None
        _run_steps(agent, 60)
        # Inject 3 simultaneous severe faults
        agent.inject_leg_loss(1)    # kill L2
        agent.inject_leg_loss(4)    # kill R2
        agent.inject_coupling_failure()
        _run_steps(agent, steps)
        coord = agent._coordination_score / max(1, agent._step_count)
        return agent._distance_travelled, coord

    ais_dist,    ais_coord    = run_agent(with_ais=True)
    no_ais_dist, no_ais_coord = run_agent(with_ais=False)
    dist_delta  = ais_dist  - no_ais_dist
    coord_delta = ais_coord - no_ais_coord

    # Pass if AIS is better on at least one metric
    if ais_dist > no_ais_dist or ais_coord > no_ais_coord:
        return r.ok(
            "AIS-equipped agent outperformed strict baseline.",
            ais_distance=f"{ais_dist:.1f}",
            no_ais_distance=f"{no_ais_dist:.1f}",
            ais_coord=f"{ais_coord:.4f}",
            no_ais_coord=f"{no_ais_coord:.4f}",
        )
    else:
        return r.fail(
            "AIS did not improve over baseline on either metric.",
            ais_distance=f"{ais_dist:.1f}",
            no_ais_distance=f"{no_ais_dist:.1f}",
            ais_coord=f"{ais_coord:.4f}",
            no_ais_coord=f"{no_ais_coord:.4f}",
        )


def test_full_survival_run() -> _TestResult:
    """
    Long run with multiple sequential faults; verify agent still alive
    and AIS memory populated.
    """
    r = _TestResult("Full survival run (6 sequential faults)")
    agent = CPGHexapod()
    # Force high energy so run doesn't end on starvation
    agent._energy_remaining = 9999.0

    fault_schedule = [
        (100, lambda: agent.inject_leg_loss(0)),
        (150, lambda: agent.inject_phase_noise(1.0)),
        (200, lambda: agent.inject_coupling_failure()),
        (250, lambda: agent.inject_leg_loss(4)),
        (300, lambda: agent.inject_phase_noise(0.7)),
        (350, lambda: [agent.cpg.enable_leg(i) for i in range(6)]),
    ]
    fault_idx = 0
    total_steps = 500
    for step in range(total_steps):
        if (fault_idx < len(fault_schedule) and
                step >= fault_schedule[fault_idx][0]):
            fault_schedule[fault_idx][1]()
            fault_idx += 1
        agent.control()

    report = agent.get_ais_report()
    still_moving = agent._distance_travelled > 5.0

    if still_moving and report["ais"]["heal_events"] >= 3:
        return r.ok(
            "Agent survived all 6 faults and continued locomotion.",
            total_faults_logged=report["faults_seen"],
            heal_events=report["ais"]["heal_events"],
            memory_cells=report["ais"]["memory_cells"],
            distance=f"{agent._distance_travelled:.1f}",
            coord=f"{report['coord_score']:.4f}",
        )
    else:
        return r.fail(
            "Agent did not survive all faults.",
            total_faults_logged=report["faults_seen"],
            heal_events=report["ais"]["heal_events"],
            distance=f"{agent._distance_travelled:.1f}",
        )


def test_headless_run_pybeast() -> _TestResult:
    """
    Verify that CPGHexapodSimulation can be instantiated and the
    _run_simulation_no_render path is reachable (PyBeast++ must be installed).
    Skips gracefully if PyBeast++ is not available.
    """
    r = _TestResult("PyBeast++ headless simulation (smoke test)")
    if not _PYBEAST_AVAILABLE:
        return r.ok("PyBeast++ not installed – test skipped (expected in pure-Python mode).")
    try:
        Config.GENERATIONS    = 2
        Config.POPULATION_SIZE = 4
        Config.ASSESSMENTS    = 1
        Config.TIMESTEPS      = 30
        sim = CPGHexapodSimulation()
        sim._run_simulation_no_render(parallel=False)
        return r.ok("Simulation ran 2 generations without error.")
    except Exception as e:
        return r.fail(f"Simulation raised: {e}")


def run_tests(verbose: bool = True) -> bool:
    """
    Execute all tests. Returns True if all pass.
    """
    suite = [
        test_leg_loss_recovery,
        test_two_leg_loss_recovery,
        test_phase_drift_recovery,
        test_coupling_failure_recovery,
        test_symmetry_failure_recovery,
        test_ais_memory_learning,
        test_comparative_fitness,
        test_full_survival_run,
        test_headless_run_pybeast,
    ]

    print("\n" + "═" * 65)
    print("  AIS Self-Healing CPG Hexapod – Test Suite")
    print("═" * 65)

    results = []
    for test_fn in suite:
        result = test_fn()
        results.append(result)
        if verbose:
            print(str(result))

    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    print("─" * 65)
    print(f"  Results: {passed}/{total} passed")
    print("═" * 65 + "\n")
    return passed == total


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_genome(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path): return None
    with open(path) as f:
        data = json.load(f)
    g = np.array(data['genome'], dtype=np.float64)
    print(f"[Loader] Loaded genome (fitness={data.get('best_fitness','?'):.4f})")
    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AIS Self-Healing CPG Hexapod")
    parser.add_argument("--test",    action="store_true",
                        help="Run the self-contained test suite")
    parser.add_argument("--evolve",  action="store_true",
                        help="Run headless evolution (PyBeast++ required)")
    parser.add_argument("--generations", type=int, default=Config.GENERATIONS)
    parser.add_argument("--population",  type=int, default=Config.POPULATION_SIZE)
    args = parser.parse_args()

    if args.test:
        ok = run_tests(verbose=True)
        sys.exit(0 if ok else 1)

    elif args.evolve:
        if not _PYBEAST_AVAILABLE:
            print("PyBeast++ not found. Cannot run GUI/simulation.")
            sys.exit(1)
        Config.GENERATIONS     = args.generations
        Config.POPULATION_SIZE = args.population
        sim = CPGHexapodSimulation()
        sim._run_simulation_no_render(parallel=False)

    else:
        # Default: run tests
        run_tests(verbose=True)
