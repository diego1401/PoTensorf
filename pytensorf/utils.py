
import torch
from dataclasses import dataclass
from typing import Type

from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)

class ZeroLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features,out_features,bias,device,dtype)

    def reset_parameters(self) -> None:
        self.weight.data.fill_(0)
        if self.bias is not None:
            self.bias.data.fill_(0)


@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.AdamW
    weight_decay: float = 0
    """The weight decay to use."""

def create_all_possible_optimizers():
    optimizers={
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=0.001),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "camera_opt": {
            "optimizer": AdamWOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=5000),
        },
    }
    possible_number_of_levels = [1,2,3,4,5,6,7,8]
    scale = 0.5 #TODO:could be tested, probably the most effect on a pyramid with 6 levels
    wd_for_bottom_levels = 0.05/100
    init_lr = 0.02 # tensorf = 0.02
    
    for n_levels in possible_number_of_levels:
        for lvl in range(n_levels):
            reversed_levels_index = n_levels - 1 - lvl
            learning_rate = init_lr#/pow(2,scale*reversed_levels_index) 
            # wd = wd_for_bottom_levels if lvl < (n_levels-1) else 0
            wd = 0
            # print("learning rate",learning_rate,"weight",wd)
            optimizers[f"color_encodings_{lvl}_pysize_{n_levels}"] = {
                "optimizer": AdamWOptimizerConfig(lr=learning_rate,weight_decay=wd),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=learning_rate/10, max_steps=30000),
            }
            optimizers[f"density_encodings_{lvl}_pysize_{n_levels}"] = {
                "optimizer": AdamWOptimizerConfig(lr=learning_rate,weight_decay=wd), #density is not sparse in any of the levels
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=learning_rate/10, max_steps=30000),
            }
    return optimizers