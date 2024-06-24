"""
Nerfstudio Template Pipeline
"""

import typing, random
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from pytensorf.template_datamanager import TemplateDataManagerConfig
from pytensorf.pytensorf import PyTensoRFModel, PyTensoRFModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.utils import profiler


@dataclass
class TemplatePipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: TemplatePipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = TemplateDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = PyTensoRFModelConfig()
    """specifies the model config"""
    # pol_supervision: bool = False
    # """Apply supervision on the different levels of the Laplacian Pyramid"""
    # pol_supervision_weight: float = 0.01


class TemplatePipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: TemplatePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                PyTensoRFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    # @profiler.time_function
    # def get_train_loss_dict(self, step: int):
    #     """This function gets your training loss dict. This will be responsible for
    #     getting the next batch of data from the DataManager and interfacing with the
    #     Model class, feeding the data to the model's forward function.

    #     Args:
    #         step: current iteration step to update sampler if using DDP (distributed)
    #     """
    #     ray_bundle, batch = self.datamanager.next_train(step)
    #     model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
    #     loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
    #     return model_outputs, loss_dict, metrics_dict
