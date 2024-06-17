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

# TDataset = TypeVar("TDataset", bound=PoLDataset, default=PoLDataset)

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

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch_tmp = next(self.iter_train_image_dataloader)
        image_batch = {'image_idx':image_batch_tmp['image_idx'],\
                        'image':image_batch_tmp['image']}
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
    
    def next_train_pol_supervision(self, level: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        image_batch = {'image_idx':image_batch['image_idx'],\
                       'image':image_batch['pyramid_of_laplacians'][level]}
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
