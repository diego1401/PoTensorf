#!/bin/bash

# Define an array with the iter values
scene_names=('hotdog' 'chair' 'ficus' 'drums' 'materials' 'mic' 'ship' 'lego')

# Loop through the iter values and run train.py with each
project_name=pytensorf-few-shot-pol-supervision
for scene_name in "${scene_names[@]}"; do
    expname="$scene_name"
    ns-train pytensorf --timestamp ''\
                       --vis wandb\
                       --experiment-name $expname\
                       --pipeline.model.number-of-levels 6 \
                       --data ../pynerf/data/nerf_synthetic/"$scene_name"\
                       --viewer.quit-on-train-completion True\
                       --project-name $project_name\
                       --output-dir outputs/$project_name\
                       --pipeline.pol-supervision True \
                       --steps-per-eval-all-images 10000 \
                       blender-free-nerf
    ns-eval --load-config outputs/$project_name/$scene_name/pytensorf/config.yml \
            --output-path outputs/$project_name/$scene_name/pytensorf/eval_free_nerf.json

done
