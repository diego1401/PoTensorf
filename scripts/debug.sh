#!/bin/bash

# Define an array with the iter values
scene=hotdog
ns-train pytensorf --timestamp ''\
                    --vis viewer\
                    --experiment-name $scene\
                    --data ../pynerf/data/nerf_synthetic/$scene\
                    --viewer.quit-on-train-completion True\
                    --project-name debug\
                    --output-dir outputs/debug\
                    --pipeline.pol-supervision True \
                    blender-free-nerf

