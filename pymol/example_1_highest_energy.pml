# To run:
# pymol -u pymol/example_1_highest_energy.pml

reinit

# Load
load "./results/example_1/original/example-1_1a5392a54bad4a86a6ee9721393ef357.pdb.bz2", fold1

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
color yelloworange, fold1
color white, elem h

# Set transparency
set stick_transparency, 0
set cartoon_transparency, 0.1
set transparency_mode, 2

# Setup view
set_view (\
   -0.965818942,    0.150674224,    0.210924506,\
   -0.208492935,   -0.935046077,   -0.286739677,\
    0.154019773,   -0.320915401,    0.934498727,\
    0.000000317,   -0.000001555,  -60.323719025,\
    0.420873374,    0.804148018,    0.244348377,\
   47.957035065,   72.690376282,  -20.000000000 )

# Ray trace
ray 6200,4400

# Save
png ./figures/example_1_highest_energy.png, dpi=600
