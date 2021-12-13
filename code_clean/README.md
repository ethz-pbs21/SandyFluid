## Animate Sand as a Fluid

*Physically-Based Simulation in Computer Graphics HS21*

*Group 19: Yifan Yu, Bo Li, Yitian Ma*

#### Usage
To setup the environment, please use
```bash
pip install -r requirements.txt
```

You can change the settings in /src/simulator.py
```bash
global_params = {
    'mode' : 'pic',                             # pic, apic, flip
    'flip_weight' : 0.95,                       # FLIP * flip_weight + PIC * (1 - flip_weight)
    'dt' : 0.01,                                # Time step
    'g' : (0.0, 0.0, -9.8),                     # Body force
    'rho': 1000.0,                              # Density of the fluid
    'grid_size' : (64, 64, 64),                 # Grid size (integer)
    'cell_extent': 0.1,                         # Extent of a single cell. grid_extent equals to the product of grid_size and cell_extent

    'reconstruct_resolution': (100, 100, 100),  # Mesh surface reconstruction grid resolution
    'reconstruct_threshold' : 0.75,             # Threshold of the metaball scalar fields
    'reconstruct_radius' : 0.1,                 # Radius of the metaball

    'num_jacobi_iter' : 100,                    # Number of iterations for pressure solving using jacobi solver
    'damped_jacobi_weight' : 1.0,               # Damping weighte in damped jacobi

    'simulate_sand': False,                     # Simulate sand or water
    'sand_dt': 0.001,                           # Time step for simulating sand

    'scene_init': 0,                            # Choose from 0, 1, 2 to init the particle positions differently
}
```

Run with GGUI:
```bash
python main.py
```

During running with GUI, you can press `Space` to reconstruct the mesh of current frame and export that into a obj model file. The output will reside in ../results.

You can also run without GUI and reconstruct each frame:

```bash
python main.py 300
```

where the simulation contains 300 frames.

If the simulation dt is too small, add a second cmdline argument to reconstruct surface every few number of simulation steps:

```bash
python main.py 300 10
```

##### Dependencies

- taichi==0.7.32
- numpy
- scipy
- PyMCubes
- connected-components-3d

