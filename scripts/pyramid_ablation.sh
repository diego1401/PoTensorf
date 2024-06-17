#!/bin/bash

# Define an array with the iter values
number_of_levels_list=(1 2 3 4 5 6 7 8)

# Loop through the iter values and run train.py with each
project_name=pyramid_levels_ablation
for n_levels in "${number_of_levels_list[@]}"; do
    expname=pyramid_w_"$n_levels"_levels
    ns-train pytensorf --timestamp ''\
                       --vis viewer+wandb\
                       --experiment-name $expname\
                       --pipeline.model.number-of-levels $n_levels\
                       --pipeline.datamanager.train-num-images-to-sample-from 100\
                       --viewer.max-num-display-images 100\
                       --data ../pynerf/data/nerf_synthetic/lego\
                       --viewer.quit-on-train-completion True\
                       --project-name $project_name\
                       --output-dir outputs/$project_name\
                       blender-data

done
