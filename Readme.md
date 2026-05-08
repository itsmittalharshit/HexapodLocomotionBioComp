# 🦟 Enhancing Hexapod Versatility with Adaptive Central Pattern Generators

> **COMP5400 — Biological and Bio-Inspired Computation | Coursework 2 | Spring 2026**  
> University of Leeds · Module Leader: Netta Cohen

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyBEAST%2B%2B-green)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 👥 Group Members

| Name | Student ID |
|------|-----------|
| Harshit Mittal | zqnw0183 |
| Siddhi Tandel | strt0792 |
| Ganesh Kuppa | qdwh0212 |

---

## 📖 Overview

This project investigates bio-inspired locomotion control for a **simulated hexapod robot** using **Central Pattern Generator (CPG) neural networks** evolved by a **Genetic Algorithm (GA)**. The methodology bridges the gap between high-level biological principles and executable simulation code, building a progressive stack of eight simulation variants — each adding a distinct layer of biological realism.

The organisation of movement in six-legged biological systems such as stick insects, beetles, and cockroaches involves managing high-dimensional redundancy across 18 degrees of freedom. Modern research addresses this through two primary bio-inspired paradigms:

- **Central Pattern Generators (CPGs):** Networks of nonlinear oscillators (Kuramoto or Hopf models) that produce rhythmic motor patterns autonomously.
- **The Walknet Model:** A decentralised ANN based on behavioural rules derived from stick insect neuroscience, governing inter-leg coordination.

This project combines both paradigms, progressively enhancing them with Hebbian plasticity, proprioception, morphological co-evolution, neuromodulation, and an Artificial Immune System (AIS) for fault-tolerant self-healing. Ultimately, this research seeks to bridge the gap between biology and technical application — moving from rigid kinematic controllers toward adaptive, self-organising locomotion.

---

## 🏆 Key Results

| Variant | Best Fitness | Peak at Gen | 95% Conv. Gen | Final Average |
|---------|:-----------:|:-----------:|:-------------:|:-------------:|
| V1 Foraging *(baseline)* | 113.61 | 76 | 14 | 46.01 |
| V2 Hebbian Plasticity | 113.83 | 60 | 23 | 42.78 |
| V3 Proprioception | 113.61 | 90 | 39 | 42.24 |
| V4 Morphology ⭐ | **149.60** | 71 | 71 | 56.34 |
| V5 Neuromodulation | 132.79 | 12 | **12** | 38.56 |
| V6 All Features | 118.24 | 94 | 63 | 42.86 |
| V7 Fault Only *(no AIS)* | 117.08 | 100 | 33 | 43.61 |
| **V8 AIS Self-Healing** 🏅 | **155.34** | 44 | 44 | **68.95** |

> **Key findings:**
> - **V4 Morphology** delivers the largest single-mechanism gain (+31.7%) by co-evolving leg length and joint stiffness alongside the neural controller.
> - **V5 Neuromodulation** converges fastest (gen 12) — metabolic fatigue constraints simplify the fitness landscape early.
> - **V8 AIS Self-Healing** achieves the highest overall fitness (+36.7% over baseline), with ~32% of that advantage coming from active fault repair.
> - **V6 All Features** ("Apex controller") underperforms individual variants — the combined high-dimensional search space exceeds the GA's fixed budget: the **"complexity tax"**.

---

## 🧬 Biological Inspiration

### Central Pattern Generators (CPGs)

Hexapod insects walk without using their brain for rhythm. The beat comes from CPGs — small neural circuits in the thoracic ganglia, one per leg, loosely coupled to each other. This project models them with the **Kuramoto oscillator**:

```
dθᵢ/dt = ω + K × Σⱼ sin(θⱼ − θᵢ − Δᵢⱼ)
```

where `ω` is intrinsic frequency, `K` is coupling strength, and `Δᵢⱼ` is the desired phase offset encoding the gait pattern.

**Supported gait presets:**

| Gait | Phase pattern | Description |
|------|-------------|-------------|
| `TRIPOD` | `[0, π, 0, π, 0, π]` | Alternating groups of 3 legs — fastest |
| `WAVE` | `[0, π/3, 2π/3, π, 4π/3, 5π/3]` | One leg at a time — most stable |
| `RIPPLE` | `[0, 2π/3, 4π/3, π/3, π, 5π/3]` | Intermediate |
| `EVOLVED` | GA-discovered | Emergent from fitness optimisation |

### The Walknet Framework

Alongside CPGs, the **Walknet model** (Cruse et al., 1998) provides the "behavioural intelligence" — a decentralised ANN governing stance/swing transitions, inter-leg coordination, and context-dependent reflexes such as forward/backward walking and gap-crossing.

### Bio-Inspired Mechanisms by Variant

| Variant | Mechanism | Biological Basis | Reference |
|---------|-----------|-----------------|-----------|
| V2 | **Hebbian Plasticity** | "Cells that fire together wire together" — dynamic 6×6 coupling weight matrix | Munakata & Pfaffly, 2004 |
| V3 | **Proprioception** | Campaniform sensilla stumble-and-recover phase-reset reflex | Laskowski et al., 2000 |
| V4 | **Morphological Co-evolution** | Body-brain co-adaptation: leg length + joint stiffness genes | Aronoff & Fudeman, 2022 |
| V5 | **Neuromodulation / Fatigue** | Metabolic stress (ATP depletion) downregulates motor drive | Luan et al., 2014 |
| V8 | **Artificial Immune System** | Clonal selection, hypermutation, self-organised fault recovery | Djurdjanovic et al., 2010 |

---

## 🗂️ Project Structure

```
pybeastpp/
├── core/                          # PyBEAST++ framework core (unchanged)
│   ├── agent/                     # Agent base classes
│   ├── evolve/                    # GA, PSO, population management
│   ├── network/                   # Feed-forward network
│   ├── sensor/                    # Area, beam, touch sensors
│   ├── world/                     # World, collisions, drawables
│   ├── simulation.py              # Base Simulation class
│   └── utils.py                   # Shared utilities
│
├── demos/                         # ← Project demo files
│   ├── hexapod_cpg.py             # 🦟 Base CPG: Kuramoto + GA (DISTANCE fitness)
│   ├── v1_foraging.py             # 🍎 V1: Energy-based foraging + food pellets
│   ├── v2_hebbian.py              # 🧠 V2: Hebbian synaptic plasticity
│   ├── v3_proprioception.py       # 👣 V3: Proprioceptive reflex feedback
│   ├── v4_morphology.py           # 🦵 V4: Body-brain morphological co-evolution
│   ├── v5_neuromodulation.py      # 😴 V5: Fatigue-based neuromodulation
│   ├── v6_all_features.py         # 🔬 V6: All 5 mechanisms combined (Apex controller)
│   ├── hexapod_faults_only.py     # ⚠️  V7: Fault injection, no AIS repair (ablation)
│   ├── hexapod_ais_selfhealing.py # 🛡️ V8: AIS self-healing fault-tolerant controller
│   ├── plot_fitness.py            # 📊 Fitness evolution plotter
│   ├── braitenberg.py             # Existing PyBEAST++ demo
│   ├── chase.py                   # Existing PyBEAST++ demo
│   ├── evo_mouse.py               # Existing PyBEAST++ demo
│   └── mouse.py                   # Existing PyBEAST++ demo
│
├── gui/                           # PyBEAST++ GUI (Qt)
│   ├── canvas.py
│   ├── frame.py
│   └── utils.py
│
├── results/                       # Saved evolution results (JSON)
│   ├── best_hexapod_genome.json          # Base CPG
│   ├── best_hexapod_foraging.json        # V1
│   ├── best_hexapod_hebbian.json         # V2
│   ├── best_hexapod_proprioception.json  # V3
│   ├── best_hexapod_morphology.json      # V4
│   ├── best_hexapod_neuromodulation.json # V5
│   ├── best_hexapod_all_features.json    # V6
│   ├── best_hexapod_faults.json          # V7
│   └── best_hexapod_ais.json             # V8
│
├── main.py                        # PyBEAST++ GUI entry point
├── run.sh                         # Apptainer/Singularity launcher
├── environment.yml                # Conda environment specification
└── apptainer.def                  # Apptainer container definition
```

---

## ⚙️ Installation

### Option 1 — Conda (recommended, Windows / Linux / macOS)

> Tested on **School of Computer Science Linux machines** and **Windows 11**

**Requirements:** [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/cpg-hexapod-locomotion.git
cd cpg-hexapod-locomotion

# 2. Create and activate the environment (first time only)
conda env create -f environment.yml
conda activate pybeast

# 3. Launch the GUI
python main.py
```

On subsequent runs:
```bash
conda activate pybeast
python main.py
```

### Option 2 — Apptainer / Singularity (HPC / Linux)

```bash
chmod +x run.sh
./run.sh
```

Creates the container on first run, reuses it thereafter.

---

## 🚀 Running the Demos

### Via the GUI

1. Launch `python main.py`
2. Select **Demo** from the top menu
3. Choose any hexapod variant from the list

### Headless (no GUI — faster for batch experiments)

```bash
# V1 Foraging
python -c "
import sys; sys.path.insert(0, '.')
from demos.v1_foraging import CPGHexapodSimulation
sim = CPGHexapodSimulation()
sim._run_simulation_no_render(parallel=False)
"

# V4 Morphology
python -c "
import sys; sys.path.insert(0, '.')
from demos.v4_morphology import CPGHexapodSimulation
sim = CPGHexapodSimulation()
sim._run_simulation_no_render(parallel=False)
"

# V8 AIS Self-Healing
python -c "
import sys; sys.path.insert(0, '.')
from demos.hexapod_ais_selfhealing import CPGHexapodSimulation
sim = CPGHexapodSimulation()
sim._run_simulation_no_render(parallel=False)
"
```

### Command-line arguments (Base CPG)

```bash
python demos/hexapod_cpg.py --generations 200 --population 30 --fitness DISTANCE --environment FLAT
```

| Argument | Options | Default |
|----------|---------|---------|
| `--generations` | any int | `100` |
| `--population` | any int | `20` |
| `--fitness` | `DISTANCE`, `EFFICIENCY`, `STABILITY` | `DISTANCE` |
| `--environment` | `FLAT`, `ROUGH`, `SLOPE` | `FLAT` |
| `--save` | filename | `best_hexapod_genome.json` |

---

## 🧪 GA Configuration

All tunable parameters are in the `Config` class at the top of each demo file — no changes needed anywhere else:

```python
class Config:
    POPULATION_SIZE = 20       # agents per generation
    GENERATIONS     = 100      # GA generations
    ASSESSMENTS     = 3        # evaluations per individual (averaged)
    TIMESTEPS       = 600      # steps per assessment

    CROSSOVER_RATE  = 0.70
    MUTATION_SIGMA  = 0.10
    ELITISM         = 2        # top-N individuals preserved unchanged

    FITNESS_MODE    = 'DISTANCE'   # 'DISTANCE' | 'FORAGING' | 'EFFICIENCY'
    ENVIRONMENT     = 'FLAT'       # 'FLAT' | 'ROUGH' | 'SLOPE'
```

---

## 📊 Genome Encoding

### Base genome — 11 genes (all variants)

| Gene | Parameter | Scaled Range |
|------|-----------|-------------|
| 0 | Oscillator frequency (ω) | 0.3 – 3.0 Hz |
| 1 | Coupling strength (K) | 0.0 – 1.0 |
| 2 | Duty factor (stance %) | 0.30 – 0.85 |
| 3 | Oscillation amplitude | 0.10 – 1.0 |
| 4–9 | Per-leg phase offsets (L1, L2, L3, R1, R2, R3) | 0 – 2π rad |

### Morphology extension — 13 genes (V4, V6, V8)

| Gene | Parameter | Scaled Range |
|------|-----------|-------------|
| 11 | Leg length multiplier | 0.5 – 2.0 × |
| 12 | Joint stiffness coefficient | 0.0 – 1.0 |

---

## 🛡️ V8 AIS Self-Healing Architecture

The Artificial Immune System monitors the CPG every 10 timesteps for five fault "antigens":

| Fault | Trigger Condition |
|-------|-----------------|
| `LEG_LOSS` | Leg output = 0 for > 20 ticks (motor seizure) |
| `PHASE_DRIFT` | Phase error > 0.5 rad from desired |
| `COUPLING_FAIL` | Average coupling weight < 0.1 |
| `ENERGY_CRISIS` | Energy < 5% of maximum |
| `SYMMETRY_FAIL` | Left/right drive imbalance > threshold |

**Repair pipeline:**
1. Fault detected → compute fault signature vector
2. Search memory pool (≤ 64 cells) for closest matching antibody
3. **Known fault:** clone best match → hypermutate toward current signature → patch CPG phases on the fly
4. **Novel fault:** run micro-GA (12 agents × 15 generations) to evolve corrective phase offsets → store winner as new memory cell
5. Memory pool saturates by ~generation 30; subsequent known faults recover from memory in < 5 timesteps

---

## 📁 Results File Format

Each `results/*.json` file:

```json
{
  "best_fitness": 155.34,
  "genome": [0.42, 0.71, ...],
  "fitness_mode": "FORAGING",
  "gait": "EVOLVED",
  "generations_run": 100,
  "history": [
    { "generation": 1,  "avg": 58.40, "best": 116.90 },
    { "generation": 2,  "avg": 64.19, "best": 114.66 }
  ]
}
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Vector maths, genome arrays |
| `matplotlib` | Fitness plotting |
| `PyOpenGL` | Agent and world rendering |
| `PyQt5` | GUI framework |

Full list: [`environment.yml`](environment.yml)

---

## 🤖 AI Use Declaration

This project is classified **Amber** for AI tool use under COMP5400 guidelines.

- **Claude (Anthropic)** was used in an assistive capacity for code implementation support and structuring documentation.
- All written analysis, report content, and conceptual decisions are the original work of the group.
- All AI-assisted code was verified by running experiments and cross-checking against result JSON files.
- AI tools were **not** used to write the report or prepare the presentation.

---

## 📚 References

1. Campos, R., Matos, V., & Santos, C. (2010). Hexapod locomotion: A nonlinear dynamical systems approach. *IECON*.
2. Dürr, V., Schmitz, J., & Cruse, H. (2004). Behaviour-based modelling of hexapod locomotion. *Arthropod Structure & Development*.
3. Wang, Z. Y., Ding, X. L., & Rovetta, A. (2010). Analysis of typical locomotion of a symmetric hexapod robot. *Robotica*.
4. Ouyang, W., Chi, H., Pang, J., Liang, W., & Ren, Q. (2021). Adaptive locomotion control of a hexapod robot via bio-inspired learning. *IEEE Transactions*.
5. Namura, N., & Nakao, H. (2025). A CPG network for simple control of gait transitions in hexapod robots. *arXiv*.
6. Kar, A. K. (2016). Bio inspired computing — a review of algorithms and scope of applications. *Expert Systems with Applications*.
7. Darwish, A. (2018). Bio-inspired computing: Algorithms review, deep analysis, and scope of applications. *Future Computing and Informatics Journal*.
8. Djurdjanovic, D., Liu, J., Marko, K. A., & Ni, J. (2010). Immune systems inspired approach to anomaly detection, fault localisation and diagnosis. *IEEE Transactions*.
9. Stephens, D. W., & Krebs, J. R. (1986). *Foraging Theory*. Princeton University Press.
10. Munakata, Y., & Pfaffly, J. (2004). Hebbian learning and development. *Developmental Science, 7*(2), 141–148.
11. Laskowski, E. R., Newcomer-Aney, K., & Smith, J. (2000). Proprioception. *Physical Medicine and Rehabilitation Clinics*.
12. Aronoff, M., & Fudeman, K. (2022). *What is Morphology?* John Wiley & Sons.
13. Luan, S., Williams, I., Nikolic, K., & Constandinou, T. G. (2014). Neuromodulation: present and emerging methods. *Frontiers in Digital Health*.
14. Cruse, H., Kindermann, T., Schumm, M., Dean, J., & Schmitz, J. (1998). Walknet — a biologically inspired network to control a six-legged robot. *Neural Networks, 11*(7–8).

---

## 📄 License

This project is released under the [MIT License](LICENSE). The underlying PyBEAST++ framework retains its original license from the University of Leeds.