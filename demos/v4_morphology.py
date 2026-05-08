"""
v4_morphology.py  –  CPG Hexapod + Foraging + Morphological Computation (Point 3 + 4)
=======================================================================================
Variant 4: Adds body-brain co-evolution (leg_length, joint_stiffness genes).

NEW vs v1_foraging:
  - Genome extended to 13 genes:
      gene 11 → leg_length    [0.5 … 2.0]  (multiplier on base leg segment lengths)
      gene 12 → joint_stiffness [0.0 … 1.0] (spring energy storage coefficient)
  - CPGHexapod.set_genotype() reads the extra morphological genes
  - CPGHexapod.control()      : longer legs → greater reach (drive scaled by leg_length);
                                 joint_stiffness stores and releases elastic energy
  - Fitness: distance-based bonus scaled by leg_length efficiency
  - Visualisation: leg segment lengths vary by genome

Biological rationale
  Body morphology and neural controller co-evolve. Long legs enable
  faster locomotion but reduce stability; stiff joints passively
  recycle kinetic energy (like a spring-mass system). The GA must
  find (body, brain) pairs that work together.
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
DEMO_NAME  = "CPG Hexapod – Morphology + Foraging"
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

    DRAW_CPG_NETWORK  = True
    DRAW_TRAILS       = True
    DRAW_FORCE_ARROWS = True
    SAVE_BEST_GENOME  = True
    GENOME_SAVE_PATH  = "best_hexapod_morphology.json"

    TIMESTEP_DT       = 0.05
    MAX_SPEED         = 120.0
    MIN_SPEED         = 0.0

    # Foraging
    FOOD_COUNT        = 12
    FOOD_CALORIES     = 25.0
    AGENT_START_ENERGY = 40.0
    ENERGY_PER_STEP   = 0.04

    # ── Morphological gene ranges ─────────────────────────────────────────
    LEG_LENGTH_MIN    = 0.5    # multiplier on base segment lengths
    LEG_LENGTH_MAX    = 2.0
    STIFFNESS_MIN     = 0.0    # joint spring coefficient
    STIFFNESS_MAX     = 1.0


GAIT_PHASES = {
    'TRIPOD': np.array([0, np.pi, 0, np.pi, 0, np.pi], dtype=np.float32),
    'WAVE':   np.array([0, np.pi/3, 2*np.pi/3,
                        np.pi, 4*np.pi/3, 5*np.pi/3], dtype=np.float32),
    'RIPPLE': np.array([0, 2*np.pi/3, 4*np.pi/3,
                        np.pi/3, np.pi, 5*np.pi/3], dtype=np.float32),
}

LEG_NAMES = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']


# ══════════════════════════════════════════════════════════════════════════════
# CPG NETWORK (unchanged from original)
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
                    cs   += math.sin(self.phi[j] - self.phi[i] - delta)
            dphi[i] = omega + self.coupling * cs
        self.phi += dphi * dt

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
        glColor4fv([0.9*glow, 0.8*glow, 0.1, 0.35])
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for a in range(20):
            ang = a/20.*2*math.pi
            glVertex2d(14*math.cos(ang), 14*math.sin(ang))
        glEnd()
        glColor4fv([1.*glow, 0.85*glow, 0.1, 0.85])
        q = gluNewQuadric(); gluDisk(q, 0, self.radius, 20, 1); gluDeleteQuadric(q)
        glColor4fv([1.,1.,0.6,0.9]); glPointSize(4.)
        glBegin(GL_POINTS); glVertex2d(0,0); glEnd()

    def __del__(self): pass


# ══════════════════════════════════════════════════════════════════════════════
# HEXAPOD AGENT
# ══════════════════════════════════════════════════════════════════════════════

class CPGHexapod(Agent, Evolver):
    """
    Extended genome: 11 original CPG genes + 2 morphological genes
      gene 11 : leg_length     (scaled to [LEG_LENGTH_MIN, LEG_LENGTH_MAX])
      gene 12 : joint_stiffness (scaled to [STIFFNESS_MIN, STIFFNESS_MAX])

    Morphological effects
    ─────────────────────
    leg_length scales every segment's visual length AND the drive force
    per stance leg (longer reach → more displacement per cycle).

    joint_stiffness models passive elastic energy storage:
      stance energy E_spring = 0.5 * k * displacement²
    which is added back as a drive bonus on the next swing → longer legs
    with stiff joints can "bounce" efficiently.

    Fitness integrates calories gathered PLUS a speed bonus normalised
    by the energetic cost of the morphology (long stiff legs are costly
    to build, penalising extreme values slightly).
    """

    GENOME_LENGTH = 13   # 11 CPG + 2 morphological

    GENE_SCALE = [
        (0.3, 3.0),    # 0 freq
        (0.0, 1.0),    # 1 coupling
        (0.3, 0.85),   # 2 duty
        (0.1, 1.0),    # 3 amplitude
    ] + [(0.0, 2 * np.pi)] * 6 + [   # 4-9 phase offsets
        (Config.LEG_LENGTH_MIN,   Config.LEG_LENGTH_MAX),   # 10 leg_length
        (Config.STIFFNESS_MIN,    Config.STIFFNESS_MAX),    # 11 joint_stiffness
    ]

    # Base segment lengths (multiplied by leg_length gene)
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

        # Morphological parameters (set by genome)
        self.leg_length       = 1.0   # multiplier
        self.joint_stiffness  = 0.3   # spring coefficient k

        # Elastic energy accumulator
        self._spring_energy   = 0.0

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

        # Read morphological genes
        self.leg_length      = self._scale_gene(g[10], 10)
        self.joint_stiffness = self._scale_gene(g[11], 11)

    def get_genotype(self):
        g = np.zeros(self.GENOME_LENGTH, dtype=np.float64)
        lo, hi = self.GENE_SCALE[0];  g[0]  = (self.cpg.freq      - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[1];  g[1]  = (self.cpg.coupling  - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[2];  g[2]  = (self.cpg.duty      - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[3];  g[3]  = (self.cpg.amplitude - lo) / (hi - lo)
        for i in range(6):
            lo, hi = self.GENE_SCALE[4+i]
            g[4+i] = (self.cpg.desired[i] - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[10]; g[10] = (self.leg_length      - lo) / (hi - lo)
        lo, hi = self.GENE_SCALE[11]; g[11] = (self.joint_stiffness - lo) / (hi - lo)
        return g

    def get_fitness(self):
        """
        Foraging fitness with morphological efficiency term.

        A morphology penalty discourages extreme leg_length AND
        extreme stiffness simultaneously (energy cost of heavy, stiff limbs).
        """
        if self._step_count == 0: return 0.0
        base     = self._calories_gathered
        explore  = self._distance_travelled * 0.01
        penalty  = self._collision_penalty * 5.0
        # Morphological cost: heavy long stiff legs waste energy
        morph_cost = (self.leg_length - 1.0) ** 2 * self.joint_stiffness * 2.0
        return max(0.0, base + explore - penalty - morph_cost)

    def reset(self):
        self._distance_travelled = 0.0
        self._energy_consumed    = 0.0
        self._calories_gathered  = 0.0
        self._energy_remaining   = Config.AGENT_START_ENERGY
        self._coordination_score = 0.0
        self._step_count         = 0
        self._collision_penalty  = 0.0
        self._foods_eaten        = 0
        self._spring_energy      = 0.0
        self._trail.clear()
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
                # Stance: drive scaled by leg_length (longer reach)
                drive = self.cpg.duty * disp * self.leg_length
                if is_left: left_drive  += drive / 3.0
                else:       right_drive += drive / 3.0
                energy_tick += abs(drive)

                # Accumulate spring energy (passive elastic storage)
                self._spring_energy += 0.5 * self.joint_stiffness * (disp ** 2) * dt
            else:
                # Swing: release stored spring energy as a boost
                spring_boost = self._spring_energy * 0.3 * self.leg_length
                if is_left: left_drive  += spring_boost / 3.0
                else:       right_drive += spring_boost / 3.0
                self._spring_energy = max(0.0, self._spring_energy - spring_boost * 2)
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
            if len(self._trail) > 200: self._trail.pop(0)

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

        if Config.DRAW_CPG_NETWORK: self._draw_cpg_network()
        for i in range(6): self._draw_leg(i)
        self._draw_body()
        if Config.DRAW_FORCE_ARROWS: self._draw_force_arrows()
        self._draw_energy_bar()
        glDisable(GL_LINE_SMOOTH)

    def _draw_energy_bar(self):
        ratio = max(0., min(1.,
            self._energy_remaining/(Config.AGENT_START_ENERGY
                                    + self._calories_gathered + 0.01)))
        bar_w=30.; bar_h=4.; x0=-15.; y0=-28.
        glColor4fv([0.3,0.,0.,0.7])
        glBegin(GL_QUADS)
        glVertex2d(x0,y0); glVertex2d(x0+bar_w,y0)
        glVertex2d(x0+bar_w,y0+bar_h); glVertex2d(x0,y0+bar_h); glEnd()
        glColor4fv([1.-ratio, ratio, 0.1, 0.9])
        glBegin(GL_QUADS)
        glVertex2d(x0,y0); glVertex2d(x0+bar_w*ratio,y0)
        glVertex2d(x0+bar_w*ratio,y0+bar_h); glVertex2d(x0,y0+bar_h); glEnd()

    def _leg_geometry(self, leg_idx):
        bw, bh = 14.0 * self.leg_length, 20.0
        return [(-bw,-bh*.6,-1),(-bw*1.2,0.,-1),(-bw,+bh*.6,-1),
                (+bw,-bh*.6,+1),(+bw*1.2,0.,+1),(+bw,+bh*.6,+1)][leg_idx]

    def _draw_leg(self, i):
        disp, swing = self.cpg.output(i)
        ax, ay, side = self._leg_geometry(i)

        # Segment lengths scaled by genome
        coxa  = self.BASE_COXA  * self.leg_length
        femur = self.BASE_FEMUR * self.leg_length

        sweep = disp * 8. * side * self.leg_length
        lift  = max(0., disp) * 8. * self.leg_length if swing else 0.
        cx = ax + side * coxa; cy = ay
        kx = ax + side * (coxa + femur * .5); ky = ay + femur * .3 + lift * .4
        fx = ax + side * (coxa + femur * .6 + sweep); fy = ay + femur * .5 + lift

        # Colour encodes stiffness: blue-tinted = stiff
        r = 0.15 + 0.7 * (1 - self.joint_stiffness)
        b = 0.2  + 0.7 * self.joint_stiffness
        glColor4fv([0.9,.45,.1,.9] if swing else [r, 0.80, b, 0.9])
        glLineWidth(1.5 + self.joint_stiffness * 2.0)
        glBegin(GL_LINE_STRIP)
        glVertex2d(ax,ay); glVertex2d(cx,cy); glVertex2d(kx,ky); glVertex2d(fx,fy)
        glEnd()
        glColor4fv([0.,1.,.5,1.] if not swing else [1.,.6,.1,.8])
        glPointSize(5.+self.leg_length*2.); glBegin(GL_POINTS); glVertex2d(fx,fy); glEnd()
        if hasattr(self, 'location'):
            self._foot_positions[i] = (self.location[0]+fx, self.location[1]+fy, not swing)

    def _draw_body(self):
        # Body size subtly reflects leg_length (larger body for long legs)
        s = 0.8 + 0.4 * (self.leg_length - Config.LEG_LENGTH_MIN) / (
            Config.LEG_LENGTH_MAX - Config.LEG_LENGTH_MIN)
        for colour, rx, ry, dy, n in [
            ([0.15,0.55,0.25,0.95], int(10*s), int(7*s), -16, 20),
            ([0.12,0.48,0.22,0.95], int(14*s), int(11*s),  0, 24),
            ([0.10,0.40,0.18,0.90], int(11*s), int(9*s),  15, 20),
        ]:
            glColor4fv(colour); glLineWidth(1.5); glBegin(GL_LINE_LOOP)
            for a in range(n):
                ang = a/n*2*math.pi
                glVertex2d(rx*math.cos(ang), ry*math.sin(ang)+dy)
            glEnd()
        for side, idx in [(-6,0),(6,3)]:
            frac = self.cpg.phase_fractions[idx]
            glColor4fv([0.2+0.8*frac, 0.8-0.6*frac, 0.5, 1.])
            glPointSize(6.); glBegin(GL_POINTS); glVertex2d(side,-18); glEnd()

    def _draw_cpg_network(self):
        radius = 28.
        glLineWidth(0.8)
        for i in range(6):
            ai = i/6.*2*math.pi-math.pi/2
            xi = radius*math.cos(ai); yi = radius*math.sin(ai)
            for j in range(i+1,6):
                aj = j/6.*2*math.pi-math.pi/2
                xj = radius*math.cos(aj); yj = radius*math.sin(aj)
                s = abs(self.cpg.coupling_strength(i,j))
                alpha = 0.05 + s*self.cpg.coupling*0.35
                glColor4fv([0.5,0.3,0.9,alpha])
                glBegin(GL_LINES); glVertex2d(xi,yi); glVertex2d(xj,yj); glEnd()

    def _draw_force_arrows(self):
        for fp in self._foot_positions:
            if fp is None: continue
            fx, fy, contact = fp
            if not contact: continue
            lx=fx-self.location[0]; ly=fy-self.location[1]
            glColor4fv([0.,0.9,0.5,0.6]); glLineWidth(1.5)
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
        super().__init__("CPGHexapod_Morphology")
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
            loc = np.array([x,y], dtype=np.float32)
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
        print("  CPG Hexapod – Morphological Computation + Foraging")
        print(f"  Genome length  : {CPGHexapod.GENOME_LENGTH} genes")
        print(f"  Leg length     : [{Config.LEG_LENGTH_MIN}, {Config.LEG_LENGTH_MAX}]")
        print(f"  Joint stiffness: [{Config.STIFFNESS_MIN}, {Config.STIFFNESS_MAX}]")
        print("=" * 60)
        super().begin_simulation()

    def end_simulation(self):
        super().end_simulation()
        if Config.SAVE_BEST_GENOME: self._ga.save_best(Config.GENOME_SAVE_PATH)
        print("=" * 60)
        print("  Evolution complete.")
        if self._ga._best_ever_genome is not None:
            g = self._ga._best_ever_genome
            lo_l, hi_l = Config.LEG_LENGTH_MIN, Config.LEG_LENGTH_MAX
            lo_s, hi_s = Config.STIFFNESS_MIN, Config.STIFFNESS_MAX
            best_ll = lo_l + (hi_l - lo_l) * g[10]
            best_st = lo_s + (hi_s - lo_s) * g[11]
            print(f"  Best fitness     : {self._ga._best_ever_fitness:.4f}")
            print(f"  Best leg_length  : {best_ll:.3f}")
            print(f"  Best stiffness   : {best_st:.3f}")
        print("=" * 60)


if __name__ == "__main__":
    sim = CPGHexapodSimulation()
    sim._run_simulation_no_render(parallel=False)
