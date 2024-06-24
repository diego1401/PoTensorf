#!/bin/bash

# Define an array with the iter values
scene=hotdog
project_name=different_models_test

# ns-train tensorf --timestamp ''\
#                        --vis wandb\
#                        --data ../pynerf/data/nerf_synthetic/"$scene"\
#                        --viewer.quit-on-train-completion True\
#                        --output-dir outputs/$project_name\
#                        --project-name $project_name\
#                        --steps-per-eval-all-images 4999\
#                        --max_num_iterations 5000\
#                        --steps_per_save 1000\
#                        blender-free-nerf


# ns-train pytensorf --timestamp ''\
#                        --vis wandb\
#                        --data ../pynerf/data/nerf_synthetic/"$scene"\
#                        --viewer.quit-on-train-completion True\
#                        --pipeline.model.pol-supervision False \
#                        --pipeline.model.number-of-levels-color 6 \
#                        --pipeline.model.number-of-levels-density 6 \
#                        --pipeline.model.field-approach volumetric\
#                        --output-dir outputs/$project_name\
#                        --project-name $project_name\
#                        --steps-per-eval-all-images 4999\
#                        --max_num_iterations 5000\
#                        --steps_per_save 1000\
#                        blender-free-nerf


ns-train pytensorf --timestamp ''\
                       --vis wandb\
                       --experiment-name pytensorf-image-no-pol\
                       --data ../pynerf/data/nerf_synthetic/"$scene"\
                       --viewer.quit-on-train-completion True\
                       --steps-per-eval-all-images 4999 \
                       --pipeline.model.pol-supervision False \
                       --pipeline.model.loss-coefficients.pol-supervision 0.01\
                       --pipeline.model.number-of-levels-color 6 \
                       --pipeline.model.number-of-levels-density 6 \
                       --pipeline.model.field-approach image\
                       --output-dir outputs/$project_name\
                       --project-name $project_name\
                       --steps-per-eval-all-images 4999\
                       --max_num_iterations 5000\
                       --steps_per_save 1000\
                       blender-free-nerf
                       

