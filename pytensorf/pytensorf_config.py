"""
Installing Pytensorf as a package.
We register the method with Nerfstudio CLI.
"""

from __future__ import annotations
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig

# PoL dataset
from pytensorf.datasets.pol_dataset import PoLDataset

# Pytensorf datamanager
from pytensorf.template_datamanager import (
    TemplateDataManagerConfig,
    TemplateDataManager
)

#Pytensorf pipeline
from pytensorf.template_pipeline import (
    TemplatePipelineConfig,
)

# some utils
from pytensorf.utils import create_all_possible_optimizers

# Including method 
from nerfstudio.plugins.types import MethodSpecification
from pytensorf.pytensorf import PyTensoRFModelConfig


pytensorf = MethodSpecification(
    config=TrainerConfig(
        method_name="pytensorf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=TemplatePipelineConfig(
            datamanager=TemplateDataManagerConfig(
                _target=TemplateDataManager,
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=PyTensoRFModelConfig(
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers=create_all_possible_optimizers(),
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio method template.",
)
