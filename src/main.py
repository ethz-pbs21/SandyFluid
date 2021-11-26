from simulator import Simulator
from gui import SimulationGUI
import taichi as ti

if __name__ == "__main__":
    
    ti.init(arch=ti.cuda)

    sim = Simulator()

    gui = SimulationGUI(sim)
    
    gui.run()