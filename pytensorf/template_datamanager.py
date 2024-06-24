"""
Template DataManager
"""

import torch

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Generic
from typing_extensions import TypeVar

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from pytensorf.datasets.pol_dataset import PoLDataset

@dataclass
class TemplateDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TemplateDataManager)

class TemplateDataManager(VanillaDataManager[PoLDataset]):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: TemplateDataManagerConfig

    def __init__(
        self,
        config: TemplateDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
