# To run:
# pymol -u pymol/example_2_cpu.pml

python
from pymol import cmd

def load_decoy(pdbfile, objname):
    found_start = False
    lines = []
    with open(pdbfile) as f:
        for line in f:
            if not found_start and not line.startswith("ATOM"):
                continue
            found_start = True
            lines.append(line)
            if line.startswith("TER"):
                break
    cmd.read_pdbstr("".join(lines), objname)

# Load original structure
load_decoy(
    "./results/example_2/cpu/original/example-2-gpu-0_d906fd986bc14c029cff3e39159e1850.pdb",
    "fold1"
)

# Load reproduced structure
load_decoy(
    "./results/example_2/cpu/reproduce/example-2-gpu-0_03bc22afcb3b4d2e9d0ad73857b8520c.pdb",
    "fold2"
)
python end

# Set global appearance
bg_color white
set ray_trace_mode, 3
set ray_trace_color, black
set ray_trace_gain, 0.15
set ray_opaque_background, 1
set ray_shadows, 0
set depth_cue, 0
set fog, 0

# Set representation
hide everything
show car
show sticks, all
set stick_radius, 0.15
set valence, 1
color tv_blue, fold1
color orange, fold2
color white, elem h

# Set transparency
set stick_transparency, 0
set cartoon_transparency, 0.1
set transparency_mode, 2

# Translate for rendering purposes only
translate [0.00001, 0.0, 0.0], fold2

# Setup view
set_view (\
   -0.535227895,    0.820223927,   -0.201893523,\
    0.842686176,    0.501972318,   -0.194663525,\
   -0.058325056,   -0.274325192,   -0.959865868,\
    0.000006223,    0.000001334,  -84.284797668,\
    0.537765861,   -0.372628808,   -0.583239734,\
   62.770500183,   90.563682556,  -20.000000000 )

# Ray trace
ray 5800,4400

# Save
png ./figures/example_2_cpu.png, dpi=600
