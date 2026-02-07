# To run:
# pymol -u pymol/example_2_gpu.pml

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
    "./results/example_2/gpu/original/example-2-gpu-1_45d23491d678412ca4e40dd92ba0ef7e.pdb",
    "fold1"
)

# Load reproduced structure
load_decoy(
    "./results/example_2/gpu/reproduce/example-2-gpu-1_8e4bed62169c4909a2166a25621741ec.pdb",
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

# Setup view
set_view (\
    0.884622753,   -0.145899564,    0.442892879,\
    0.355090201,    0.826394200,   -0.437017500,\
   -0.302242845,    0.543860912,    0.782854736,\
   -0.000000399,   -0.000004744,  -84.284797668,\
   -0.662451982,    1.229527593,    1.397104025,\
   65.428100586,  103.141494751,  -20.000000000 )

# # Ray trace
ray 4400,4400

# Save
png ./figures/example_2_gpu.png, dpi=600


