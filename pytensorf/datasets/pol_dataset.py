
"""
Pyramid of Laplacian dataset.
"""

import json, os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.rich_utils import CONSOLE

from pytensorf.datasets.create_pols import process_images


class PoLDataset(InputDataset):
    """Dataset that returns images and their Pyramid of Laplacian decomposition.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        # Compute the pyramid of laplacian decomposition for the given dataset
        
        super().__init__(dataparser_outputs, scale_factor)
        CONSOLE.print("Computing Pyramid of Laplacian decomposition")
        
        input_dir = dataparser_outputs.image_filenames[0].parent
        output_dir = "/".join(str(input_dir).split('/')[2:])
        self.levels_dir = output_dir
        self.number_of_levels = 6
        if not os.path.exists(output_dir):
            process_images(input_dir,output_dir,levels=self.number_of_levels)
         

    def get_metadata(self, data: Dict) -> Dict:

        filepath = lambda idx: f"{self.levels_dir}/r_{data['image_idx']}/laplacian_level_{idx}.png"
        list_of_levels = []
        for lvl in range(self.number_of_levels):
            image_filename = filepath(lvl)
            pil_image = Image.open(image_filename)
            if self.scale_factor != 1.0:
                width, height = pil_image.size
                newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
                pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
            image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
            if len(image.shape) == 2:
                image = image[:, :, None].repeat(3, axis=2)
            assert len(image.shape) == 3
            assert image.dtype == np.uint8
            assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
            tensor_image = torch.from_numpy(image.astype("float32") / 255.0)
            if self._dataparser_outputs.alpha_color is not None and tensor_image.shape[-1] == 4:
                assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                    self._dataparser_outputs.alpha_color <= 1
                ).all(), "alpha color given is out of range between [0, 1]."
                tensor_image = tensor_image[:, :, :3] * tensor_image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - tensor_image[:, :, -1:])
            list_of_levels.append(tensor_image)
        return {"pyramid_of_laplacians": list_of_levels}