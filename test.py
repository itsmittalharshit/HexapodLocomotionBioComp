from demos.evo_mouse import EvoMouseSimulation

sim = EvoMouseSimulation()
sim.runs = 8  # Multiple independent runs

# Run in parallel - that's it!
sim.run_simulation(render=False, parallel=True)
