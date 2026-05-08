import logging
import sys
import time

from abc import ABC
from OpenGL.GL import glFinish
from core.world.world import World
from core.evolve.base import SimulationObject

class Simulation(ABC):
    def __init__(self, name):
        self.world: World = World(self)
        self.contents: dict[str, SimulationObject] = {}
        self.runs: int = 1
        self.generations: int = 1
        self.assessments: int = 1
        self.timesteps: int = 1000
        self.time_increment: int = 1
        self.sleep_betwen_logs: float = 0.0
        
        self._timestep: int = 0
        self._complete: bool = False
        self._run: int = 0
        self._generation: int = 0
        self._assessment: int = 0
        
        self.log: logging.Logger = logging.getLogger(name)
        self._initialise_logger()
        
        self.loaded = False
    
    def _initialise_logger(self) -> None:
        self.log.handlers = []
        self.log.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self.log.addHandler(console_handler)
    
    def _to_string(self, style: int) -> str:
        pass
    
    def run_simulation(self, render = False, parallel = False) -> None:
        if render:
            # Import here to avoid circular dependency
            from main import App
            app = App(simulation=self)
            sys.exit(app.exec_())
        else:
            self._run_simulation_no_render(parallel)
    
    def _run_simulation_no_render(self, parallel):
        self.initialise()
        self.begin_simulation()
        
        complete = False
        while not complete:
            complete = self.update()
    
    def parallel_runs(self):
        # TODO: perform runs in parallels
        pass
    
    def clean(self) -> None:
        # Clean up Qt application if running
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            for window in app.topLevelWindows():
                if window:
                    window.close()
            app.quit()
    
    def add(self, name: str, obj: SimulationObject) -> None:
        self.contents[name] = obj
    
    def initialise(self) -> None:
        for obj in self.contents.values():
            obj.world = self.world
    
    def display(self) -> None:
        self.world.display()
        glFinish()
    
    def update(self) -> bool:
        self.world.update()
        self._timestep += self.time_increment
        
        self.log_update()
        
        if self._timestep == self.timesteps:
            self.end_assessment()
        return self._complete
    
    def begin_simulation(self) -> None:
        self.log_begin_simulation()
        time.sleep(0.2)
        
        self._complete = False
        self._run = 0
        
        for obj in self.contents.values():
            obj.begin_generation()
            
        self.begin_assessment()
    
    def resume_simulation(self) -> None:
        self.log_resume_simulation()
        time.sleep(self.sleep_betwen_logs)
    
    def begin_run(self) -> None:
        self.log_begin_run()
        time.sleep(self.sleep_betwen_logs)
        self._generation = 0
        
        for obj in self.contents.values():
            obj.begin_run()
        
        self.begin_generation()
    
    def begin_generation(self) -> None:
        self.log_begin_generation()
        time.sleep(0.2)
        self._assessment = 0
        
        for obj in self.contents.values():
            obj.begin_generation()
        
        self.begin_assessment()
        
    def begin_assessment(self) -> None:
        self.log_begin_assessment()
        time.sleep(self.sleep_betwen_logs)
        
        self._timestep = 0
        
        for obj in self.contents.values():
            obj.begin_assessment()
            obj.add_to_world()
        
        self.world.initialise()
        
    
    def end_simulation(self) -> None:
        self.log_end_simulation()
        time.sleep(self.sleep_betwen_logs)
        self.world.clean()
        self._complete = True
    
    def end_run(self) -> None:
        self.log_end_run()
        time.sleep(self.sleep_betwen_logs)
        
        for obj in self.contents.values():
            obj.end_run()
        for name, obj in self.contents.items():
            if hasattr(obj, "_genetic_algorithm"):
                ga = obj._genetic_algorithm
                print("\n--- Fitness Summary ---")
                print("Generations:", len(ga._average_fitness_record))
                print("Mean fitness per generation:")
                print(ga._average_fitness_record)
                print("Best fitness per generation:")
                print(ga._best_fitness_record)
        self._run += 1
        if self._run == self.runs:
            self.end_simulation()
        else:
            self.begin_run()
    
    def end_generation(self) -> None:
        self.log_end_generation()
        time.sleep(self.sleep_betwen_logs)
        
        for obj in self.contents.values():
            obj.end_generation()
        self._generation += 1
        
        if self._generation == self.generations:
            self.end_run()
        else:
            self.begin_generation()
    
    def end_assessment(self) -> None:
        for obj in self.contents.values():
            obj.end_assessment()
        
        self.log_end_assessment()
        time.sleep(self.sleep_betwen_logs)
        self.world.clean()
        self._assessment += 1
        
        if self._assessment == self.assessments:
            self.end_generation()
        else:
            self.begin_assessment()
    
    def reset_run(self) -> None:
        self.world.clean()
        self._run -= 1
        self.begin_run()
    
    def reset_generation(self) -> None:
        self.world.clean()
        self.begin_generation()
    
    def reset_assessment(self) -> None:
        self.world.clean()
        self.begin_assessment()
    
    def log_begin_simulation(self) -> None:
        self.log.info(f"Simulation Started: {self.runs} runs, {self.generations} generations per run, "
                      f"{self.assessments} assessments per generation, {self.timesteps} timesteps.")
    
    def log_resume_simulation(self) -> None:
        pass
    
    def log_end_simulation(self) -> None:
        pass
    
    def log_begin_run(self) -> None:
        pass
    
    def log_end_run(self) -> None:
        pass
    
    def log_begin_generation(self) -> None:
        pass
    
    def log_end_generation(self) -> None:
        for name, obj in self.contents.items():
            if hasattr(obj, "_genetic_algorithm"):
                ga = obj._genetic_algorithm
                if ga._average_fitness_record:
                    mean_fit = ga._average_fitness_record[-1]
                    best_fit = ga._best_fitness_record[-1]
                    self.log.info(
                    f"Generation {self._generation}: "
                    f"Mean Fitness = {mean_fit:.3f}, "
                    f"Best Fitness = {best_fit:.3f}"
    )
    
    def log_begin_assessment(self) -> None:
        pass
    
    def log_end_assessment(self) -> None:
        pass
    
    def log_update(self) -> None:
        pass
