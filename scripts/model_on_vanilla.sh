#!/bin/bash
# Define an array with the iter values
scene_names=('hotdog' 'chair' 'ficus' 'drums' 'materials' 'mic' 'ship' 'lego')
datasets=('blender-data' 'blender-free-nerf')

for dataset_name in "${datasets[@]}"; do
    # Loop through the iter values and run train.py with each
    project_name=pytensorf-volumetric-$dataset_name
    for scene_name in "${scene_names[@]}"; do
        expname="$scene_name"
        ns-train pytensorf --timestamp ''\
                        --vis wandb\
                        --experiment-name $expname\
                        --data ../pynerf/data/nerf_synthetic/"$scene_name"\
                        --viewer.quit-on-train-completion True\
                        --project-name $project_name\
                        --output-dir outputs/$project_name\
                        --steps-per-eval-all-images 10000 \
                        --pipeline.model.number-of-levels-color 6\
                        --pipeline.model.number-of-levels-density 6\
                        --pipeline.model.field-approach volumetric\
                        $dataset_name
        ns-eval --load-config outputs/$project_name/$scene_name/pytensorf/config.yml \
                --output-path outputs/$project_name/$scene_name/pytensorf/eval_free_nerf.json

    done

    # Loop through the iter values and run train.py with each
    project_name=pytensorf-image-$dataset_name
    for scene_name in "${scene_names[@]}"; do
        expname="$scene_name"
        ns-train pytensorf --timestamp ''\
                        --vis wandb\
                        --experiment-name $expname\
                        --data ../pynerf/data/nerf_synthetic/"$scene_name"\
                        --viewer.quit-on-train-completion True\
                        --project-name $project_name\
                        --output-dir outputs/$project_name\
                        --steps-per-eval-all-images 10000 \
                        --pipeline.model.pol-supervision False \
                        --pipeline.model.loss-coefficients.pol-supervision 0\
                        --pipeline.model.field-approach image\
                        $dataset_name
        ns-eval --load-config outputs/$project_name/$scene_name/pytensorf/config.yml \
                --output-path outputs/$project_name/$scene_name/pytensorf/eval_free_nerf.json

    done

    # Loop through the iter values and run train.py with each
    project_name=pytensorf-image-pol-supervision-$dataset_name
    for scene_name in "${scene_names[@]}"; do
        expname="$scene_name"
        ns-train pytensorf --timestamp ''\
                        --vis wandb\
                        --experiment-name $expname\
                        --data ../pynerf/data/nerf_synthetic/"$scene_name"\
                        --viewer.quit-on-train-completion True\
                        --project-name $project_name\
                        --output-dir outputs/$project_name\
                        --steps-per-eval-all-images 10000 \
                        --pipeline.model.pol-supervision True \
                        --pipeline.model.loss-coefficients.pol-supervision 0.1\
                        --pipeline.model.field-approach image\
                        $dataset_name
        ns-eval --load-config outputs/$project_name/$scene_name/pytensorf/config.yml \
                --output-path outputs/$project_name/$scene_name/pytensorf/eval_free_nerf.json

    done
done