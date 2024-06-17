#!/bin/bash

# Define an array with the iter values
scene_names=('hotdog' 'chair' 'ficus' 'drums' 'materials' 'mic' 'ship' 'lego')

# Loop through the iter values and run train.py with each
project_name=tensorf-few-shot-from-small-grid
for scene_name in "${scene_names[@]}"; do
    expname="$scene_name"
    ns-train tensorf --timestamp ''\
                       --vis wandb\
                       --experiment-name $expname\
                       --data ../pynerf/data/nerf_synthetic/"$scene_name"\
                       --viewer.quit-on-train-completion True\
                       --pipeline.model.init-resolution 10 \
                       --project-name $project_name\
                       --output-dir outputs/$project_name\
                       blender-free-nerf
    ns-eval --load-config outputs/$project_name/$scene_name/tensorf/config.yml \
            --output-path outputs/$project_name/$scene_name/tensorf/eval_free_nerf.json

done
