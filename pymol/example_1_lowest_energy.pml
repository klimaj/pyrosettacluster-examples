# To run:
# pymol -u pymol/example_1_lowest_energy.pml

reinit

# Load
load "./results/example_1/original/example-1_ee4e3f706805477b9fd81c8ff3516949.pdb.bz2", fold1
load "./results/example_1/reproduce/example-1-reproduce_2ccbfc4774024e0bb1e3aebe6bd9fd63.pdb.bz2", fold2

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
   -0.965818942,    0.150674224,    0.210924506,\
   -0.208492935,   -0.935046077,   -0.286739677,\
    0.154019773,   -0.320915401,    0.934498727,\
    0.000000317,   -0.000001555,  -60.323719025,\
    0.420873374,    0.804148018,    0.244348377,\
   47.957035065,   72.690376282,  -20.000000000 )

# Ray trace
ray 4800,4400

# Save
png ./figures/example_1_lowest_energy.png, dpi=600

