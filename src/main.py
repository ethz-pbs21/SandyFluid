from simulator import Simulator
from gui import SimulationGUI
import taichi as ti
import sys

if __name__ == "__main__":
    
    ti.init(arch=ti.cuda)

    num_step = int(sys.argv[1]) if len(sys.argv) >= 2 else None
    sim = Simulator()
    if num_step is None:
        gui = SimulationGUI(sim)
        gui.run()
    else:
        for i in range(num_step):
            print('step {0} of {1}'.format(i, num_step-1))
            sim.reconstruct_mesh()
            sim.step()