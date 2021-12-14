from simulator import Simulator
from gui import SimulationGUI
import taichi as ti
import sys

OUTPUT = False

if __name__ == "__main__":

    ti.init(arch=ti.cuda)

    n_frame = int(sys.argv[1]) if len(sys.argv) >= 2 else None
    reconstruct_step = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    sim = Simulator()
    if n_frame is None:
        gui = SimulationGUI(sim, output_img=OUTPUT)
        gui.run()
    else:
        for i in range(n_frame):
            print('frame {0} of {1}'.format(i, n_frame - 1))
            sim.reconstruct_mesh('{0}_{1:04d}.obj'.format(sim.mode, i))
            for j in range(reconstruct_step):
                sim.step()
