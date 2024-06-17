"""TensoRF Field"""

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from typing import Dict, Optional

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import Encoding, Identity, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field


class PytensoRFField(Field):
    """PytensoRF Field"""

    def __init__(
        self,
        aabb: Tensor,
        # the aabb bounding box of the dataset
        feature_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for appearance encoding outputs
        direction_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for ray direction
        density_encoding_levels: list = [Identity(in_dim=3)],
        # the tensor encoding method used for scene density
        color_encoding_levels: list = [Identity(in_dim=3)],
        # the tensor encoding method used for scene color
        appearance_dim: int = 27,
        # the number of dimensions for the appearance embedding
        head_mlp_num_layers: int = 2,
        # number of layers for the MLP
        head_mlp_layer_width: int = 128,
        # layer width for the MLP
        use_sh: bool = False,
        # whether to use spherical harmonics as the feature decoding function
        sh_levels: int = 2,
        # number of levels to use for spherical harmonics
    ) -> None:
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.aabb = Parameter(aabb, requires_grad=False)
        self.feature_encoding = feature_encoding
        self.direction_encoding = direction_encoding
        self.density_encoding_levels = density_encoding_levels
        self.color_encoding_levels = color_encoding_levels


        self.mlp_head = MLP(
            in_dim=appearance_dim + 3 + self.direction_encoding.get_out_dim() + self.feature_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )

        self.use_sh = use_sh

        if self.use_sh:
            self.sh = SHEncoding(sh_levels)
            self.B = nn.Linear(
                in_features=self.color_encoding_levels[0].get_out_dim(), out_features=3 * self.sh.get_out_dim(), bias=False
            )
        else:
            self.B = nn.Linear(in_features=self.color_encoding_levels[0].get_out_dim(), out_features=appearance_dim, bias=False)

        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(), activation=nn.Sigmoid())

    def get_density(self, ray_samples: RaySamples) -> Tensor:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        densities_list = []
        for lvl in range(len(self.density_encoding_levels)):
            density_encoding = self.density_encoding_levels[lvl]
            density_at_lvl = density_encoding(positions)
            densities_list.append(density_at_lvl)
        density = torch.stack(densities_list,dim=0).sum(0) # AGREGATION
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        relu = torch.nn.ReLU()
        density_enc = relu(density_enc)
        return density_enc
    
    def get_density_at_all_levels(self,ray_samples:RaySamples)-> Tensor:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        densities_list = []
        for lvl in range(len(self.density_encoding_levels)):
            density_encoding = self.density_encoding_levels[lvl]
            density_at_lvl = density_encoding(positions)
            densities_list.append(density_at_lvl)
        density = torch.stack(densities_list,dim=0)
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        relu = torch.nn.ReLU()
        density_enc = relu(density_enc)
        return density_enc
    
    def get_density_at_level(self,ray_samples:RaySamples, level:int)-> Tensor:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        density_encoding = self.density_encoding_levels[level]
        density_at_lvl = density_encoding(positions)
        density_enc = torch.sum(density_at_lvl, dim=-1)[:, :, None]
        relu = torch.nn.ReLU()
        density_enc = relu(density_enc)
        return density_enc

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        rgb_features_list = []
        for lvl in range(len(self.color_encoding_levels)):
            color_enconding = self.color_encoding_levels[lvl]
            features_at_lvl = color_enconding(positions)
            rgb_features_list.append(features_at_lvl)
        rgb_features = torch.stack(rgb_features_list,dim=0).sum(0) # AGREGATION       
        rgb_features = self.B(rgb_features)

        if self.use_sh:
            sh_mult = self.sh(d)[:, :, None]
            rgb_sh = rgb_features.view(sh_mult.shape[0], sh_mult.shape[1], 3, sh_mult.shape[-1])
            rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
        else:
            d_encoded = self.direction_encoding(d)
            rgb_features_encoded = self.feature_encoding(rgb_features)

            out = self.mlp_head(torch.cat([rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))  # type: ignore
            rgb = self.field_output_rgb(out)

        return rgb
    
    def get_outputs_at_level(self, ray_samples: RaySamples,level:int, density_embedding: Optional[Tensor] = None) -> Tensor:
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        color_enconding = self.color_encoding_levels[level]
        features_at_lvl = color_enconding(positions)  
        rgb_features = self.B(features_at_lvl)

        if self.use_sh:
            sh_mult = self.sh(d)[:, :, None]
            rgb_sh = rgb_features.view(sh_mult.shape[0], sh_mult.shape[1], 3, sh_mult.shape[-1])
            rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
        else:
            d_encoded = self.direction_encoding(d)
            rgb_features_encoded = self.feature_encoding(rgb_features)

            out = self.mlp_head(torch.cat([rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))  # type: ignore
            rgb = self.field_output_rgb(out)

        return rgb
    
    def get_outputs_at_all_levels(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        rgb_features_list = []
        for lvl in range(len(self.color_encoding_levels)):
            color_enconding = self.color_encoding_levels[lvl]
            features_at_lvl = color_enconding(positions)
            rgb_features_list.append(features_at_lvl)
        rgb_features = torch.stack(rgb_features_list,dim=0)#.sum(0) # AGREGATION       
        rgb_features = self.B(rgb_features)

        if self.use_sh:
            sh_mult = self.sh(d)[:, :, None]
            rgb_sh = rgb_features.view(sh_mult.shape[0], sh_mult.shape[1], 3, sh_mult.shape[-1])
            rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
        else:
            d_encoded = self.direction_encoding(d)
            rgb_features_encoded = self.feature_encoding(rgb_features)

            out = self.mlp_head(torch.cat([rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))  # type: ignore
            rgb = self.field_output_rgb(out)

        return rgb

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[Tensor] = None,
        bg_color: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals is True:
            raise ValueError("Surface normals are not currently supported with TensoRF")
        if mask is not None and bg_color is not None:
            base_density = torch.zeros(ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(ray_samples[:, :, None].shape)
            if mask.any():
                input_rays = ray_samples[mask, :]
                density = self.get_density(input_rays)
                rgb = self.get_outputs(input_rays, None)

                base_density[mask] = density
                base_rgb[mask] = rgb

                base_density.requires_grad_()
                base_rgb.requires_grad_()

            density = base_density
            rgb = base_rgb
        else:
            density = self.get_density(ray_samples)
            rgb = self.get_outputs(ray_samples, None)

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}
    
    def forward_at_level(
        self,
        ray_samples: RaySamples,
        level: int,
        compute_normals: bool = False,
        mask: Optional[Tensor] = None,
        bg_color: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals is True:
            raise ValueError("Surface normals are not currently supported with TensoRF")
        if mask is not None and bg_color is not None:
            base_density = torch.zeros(ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(ray_samples[:, :, None].shape)
            if mask.any():
                input_rays = ray_samples[mask, :]
                density = self.get_density_at_level(input_rays, level)
                rgb = self.get_outputs_at_level(input_rays, level, None)

                base_density[mask] = density
                base_rgb[mask] = rgb

                base_density.requires_grad_()
                base_rgb.requires_grad_()

            density = base_density
            rgb = base_rgb
        else:
            density = self.get_density_at_level(ray_samples,level)
            rgb = self.get_outputs(ray_samples, level, None)

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}
    
    def forward_at_all_levels(
        self,
        ray_samples: RaySamples,
        level: int,
        compute_normals: bool = False,
        mask: Optional[Tensor] = None,
        bg_color: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals is True:
            raise ValueError("Surface normals are not currently supported with TensoRF")
        if mask is not None and bg_color is not None:
            base_density = torch.zeros(ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(ray_samples[:, :, None].shape)
            if mask.any():
                input_rays = ray_samples[mask, :]
                density = self.get_density_at_all_levels(input_rays, level)
                rgb = self.get_outputs_at_all_levels(input_rays, level, None)

                base_density[mask] = density
                base_rgb[mask] = rgb

                base_density.requires_grad_()
                base_rgb.requires_grad_()

            density = base_density
            rgb = base_rgb
        else:
            density = self.get_density_at_all_levels(ray_samples,level)
            rgb = self.get_outputs_at_all_levels(ray_samples, level, None)

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}