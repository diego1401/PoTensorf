#!/bin/bash

# Define an array with the iter values
scene=hotdog
ns-train pytensorf --timestamp ''\
                       --vis viewer\
                       --experiment-name debug_no_pol_super\
                       --data ../pynerf/data/nerf_synthetic/"$scene"\
                       --viewer.quit-on-train-completion True\
                       --output-dir outputs/debug\
                       --steps-per-eval-all-images 10000 \
                       --pipeline.model.pol-supervision False \
                       --pipeline.model.loss-coefficients.pol-supervision 0.001\
                       --pipeline.model.number-of-levels-color 6\
                       --pipeline.model.number-of-levels-density 6\
                       --pipeline.model.field-approach volumetric\
                       --max_num_iterations 5000\
                       --steps_per_save 1000\
                       blender-free-nerf
                       

