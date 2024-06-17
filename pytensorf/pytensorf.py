"""
Pytensorf Model file
"""
from __future__ import annotations

import torch
from torch.nn import Parameter
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, cast, Union

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding, TensorCPEncoding, TensorVMEncoding, TriplaneEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.cameras import Cameras

from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared, tv_loss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc

from pytensorf.pytensorf_field import PytensoRFField


@dataclass
class PyTensoRFModelConfig(ModelConfig):
    """PyTensoRF model config"""

    _target: Type = field(default_factory=lambda: PyTensoRFModel)
    """target class to instantiate"""
    resolution: int = 300
    """Resolution of the model"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb_loss": 1.0,
            "tv_reg_density": 1e-3,
            "tv_reg_color": 1e-4,
            "l1_reg": 5e-4,
        }
    )
    """Loss specific weights."""
    num_samples: int = 50
    """Number of samples in field evaluation"""
    num_uniform_samples: int = 200
    """Number of samples in density evaluation"""
    num_den_components: int = 16
    """Number of components in density encoding"""
    num_color_components: int = 48
    """Number of components in color encoding"""
    appearance_dim: int = 27
    """Number of channels for color encoding"""
    tensorf_encoding: Literal["triplane", "vm", "cp"] = "vm"
    regularization: Literal["none", "l1", "tv"] = "tv"
    """Regularization method used in tensorf paper"""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""

    # New parameters
    number_of_levels: int = 6

class PyTensoRFModel(Model):
    """PyTensoRF Model

    Args:
        config: PyTensoRF configuration to instantiate model
    """

    config: PyTensoRFModelConfig

    def __init__(
        self,
        config: PyTensoRFModelConfig,
        **kwargs,
    ) -> None:
        self.number_of_levels = config.number_of_levels
        self.resolution = config.resolution
        assert self.resolution > 2**config.number_of_levels
        self.num_den_components = config.num_den_components
        self.num_color_components = config.num_color_components
        self.appearance_dim = config.appearance_dim
        
        super().__init__(config=config, **kwargs)

    def return_encodings(self,resolution,init_scale_color,init_scale_density):
        if self.config.tensorf_encoding == "vm":
            density_encoding = TensorVMEncoding(
                resolution=resolution,
                num_components=self.num_den_components,
                init_scale=init_scale_density,
            )
            color_encoding = TensorVMEncoding(
                resolution=resolution,
                num_components=self.num_color_components,
                init_scale=init_scale_color,
            )
        elif self.config.tensorf_encoding == "cp":
            density_encoding = TensorCPEncoding(
                    resolution= resolution,
                    num_components=self.num_den_components,
                    init_scale=init_scale_density,
                )
            color_encoding = TensorCPEncoding(
                resolution= resolution,
                num_components=self.num_color_components,
                init_scale=init_scale_color,
            )
        elif self.config.tensorf_encoding == "triplane":
            density_encoding = TriplaneEncoding(
                resolution=resolution,
                num_components=self.num_den_components,
                init_scale=init_scale_density,
            )
            color_encoding = TriplaneEncoding(
                resolution=resolution,
                num_components=self.num_color_components,
                init_scale=init_scale_color,
            )
        else:
            raise ValueError(f"Encoding {self.config.tensorf_encoding} not supported")
        
        return density_encoding,color_encoding
    
    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        density_encoding_levels = []
        color_encoding_levels = []
        current_resolution = self.resolution
        for lvl in range(self.number_of_levels):

            init_scale = 0.1 if lvl == (self.number_of_levels-1) else 1e-7
            # all color levels are init at almost 0 but the last (coarsest) one
            density_encoding,color_encoding = self.return_encodings(current_resolution,init_scale_color=init_scale,init_scale_density=0.1)

            density_encoding_levels.append(density_encoding)
            color_encoding_levels.append(color_encoding)
            current_resolution //= 2

        density_encoding_levels = torch.nn.ModuleList(density_encoding_levels)
        color_encoding_levels = torch.nn.ModuleList(color_encoding_levels)

        feature_encoding = NeRFEncoding(in_dim=self.appearance_dim, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)

        self.field = PytensoRFField(
            self.scene_box.aabb,
            feature_encoding=feature_encoding,
            direction_encoding=direction_encoding,
            density_encoding_levels=density_encoding_levels,
            color_encoding_levels=color_encoding_levels,
            appearance_dim=self.appearance_dim,
            head_mlp_num_layers=2,
            head_mlp_layer_width=128,
            use_sh=False,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples, single_jitter=True, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

        # regularizations
        if self.config.tensorf_encoding == "cp" and self.config.regularization == "tv":
            raise RuntimeError("TV reg not supported for CP decomposition")

        # (optional) camera optimizer
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["fields"] = (
            list(self.field.mlp_head.parameters())
            + list(self.field.B.parameters())
            + list(self.field.field_output_rgb.parameters())
        )
        
        for lvl in range(self.number_of_levels):
            param_groups[f"color_encodings_{lvl}_pysize_{self.number_of_levels}"] = list(self.field.color_encoding_levels[lvl].parameters())
        for lvl in range(self.number_of_levels):
            param_groups[f"density_encodings_{lvl}_pysize_{self.number_of_levels}"] = list(self.field.density_encoding_levels[lvl].parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field:
        field_outputs_fine = self.field.forward(
            ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)
        pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        rgb_loss = self.rgb_loss(image, pred_image)

        loss_dict = {"rgb_loss": rgb_loss}

        if self.config.regularization == "l1":
            l1_parameters = []
            for lvl in range(self.number_of_levels):
                for parameter in self.field.density_encoding_levels[lvl].parameters():
                    l1_parameters.append(parameter.view(-1))
            loss_dict["l1_reg"] = torch.abs(torch.cat(l1_parameters)).mean()
        elif self.config.regularization == "tv":
            tv_density_loss = 0
            tv_reg_loss = 0
            for lvl in range(self.number_of_levels):
                density_plane_coef = self.field.density_encoding_levels[lvl].plane_coef
                color_plane_coef = self.field.color_encoding_levels[lvl].plane_coef
                assert isinstance(color_plane_coef, torch.Tensor) and \
                       isinstance(density_plane_coef, torch.Tensor), \
                       "TV reg only supported for TensoRF encoding types with plane_coef attribute"
                tv_density_loss += tv_loss(density_plane_coef)
                tv_reg_loss += tv_loss(color_plane_coef)
            loss_dict["tv_reg_density"] = tv_density_loss
            loss_dict["tv_reg_color"] = tv_reg_loss
        elif self.config.regularization == "none":
            pass
        else:
            raise ValueError(f"Regularization {self.config.regularization} not supported")

        self.camera_optimizer.get_loss_dict(loss_dict)

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = cast(torch.Tensor, self.ssim(image, rgb))
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        self.camera_optimizer.get_metrics_dict(metrics_dict)

        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict
    
    '''
    Functions to perform Laplacian Pyramid supervision
    '''

    def get_outputs_at_level(self, ray_bundle: RayBundle, level: int):
        # uniform sampling
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens = self.field.get_density_at_level(ray_samples_uniform,level)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field:
        field_outputs_fine = self.field.forward_at_level(
            ray_samples_pdf,level, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs
    
    def get_outputs_at_all_levels(self, ray_bundle: RayBundle, level: int):
        # uniform sampling
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens = self.field.get_density_at_all_levels(ray_samples_uniform) # [L, ...]
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field:
        field_outputs_fine = self.field.forward_at_all_levels(
            ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs

    def forward_at_level(self, ray_bundle: Union[RayBundle, Cameras], level: int) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs_at_level(ray_bundle,level)









