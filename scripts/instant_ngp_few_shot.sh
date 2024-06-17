#!/bin/bash

# Define an array with the iter values
scene_names=('hotdog' 'chair' 'ficus' 'drums' 'materials' 'mic' 'ship' 'lego')

# Loop through the iter values and run train.py with each
project_name=instant-ngp-bounded-few-shot
for scene_name in "${scene_names[@]}"; do
    expname="$scene_name"
    ns-train instant-ngp-bounded --timestamp ''\
                       --vis wandb\
                       --experiment-name $expname\
                       --data ../pynerf/data/nerf_synthetic/"$scene_name"\
                       --viewer.quit-on-train-completion True\
                       --project-name $project_name\
                       --output-dir outputs/$project_name\
                       blender-free-nerf
    ns-eval --load-config outputs/$project_name/$scene_name/instant-ngp-bounded/config.yml \
            --output-path outputs/$project_name/$scene_name/instant-ngp-bounded/eval_free_nerf.json

done
