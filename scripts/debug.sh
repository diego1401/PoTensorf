#!/bin/bash

# Define an array with the iter values
scene=hotdog
ns-train pytensorf --timestamp ''\
                       --vis viewer\
                       --experiment-name debug\
                       --pipeline.model.number-of-levels 6 \
                       --data ../pynerf/data/nerf_synthetic/"$scene"\
                       --viewer.quit-on-train-completion True\
                       --output-dir outputs/debug\
                       --steps-per-eval-all-images 10000 \
                       blender-free-nerf

