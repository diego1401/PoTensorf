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

from pytensorf.utils import ZeroLinear


class PytensoRFVolumetricField(Field):
    """PytensoRF Field using volumetric approach"""

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
        self.number_of_levels_density = len(density_encoding_levels)
        self.number_of_levels_color = len(color_encoding_levels)

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
            out_features_B = 3 * self.sh.get_out_dim()
        else:
            out_features_B = appearance_dim
        self.B = nn.Linear(in_features=self.color_encoding_levels[0].get_out_dim(), \
                           out_features=out_features_B, bias=False)
        
        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(), \
                             activation=nn.Sigmoid())

    def get_density(self, ray_samples: RaySamples) -> Tensor:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        densities_list = []
        for lvl in range(self.number_of_levels_density):
            density_encoding = self.density_encoding_levels[lvl]
            density_at_lvl = density_encoding(positions)
            densities_list.append(density_at_lvl)
        density = torch.stack(densities_list,dim=0).sum(0) # Density should always be a shared value
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        relu = torch.nn.ReLU()
        density_enc = relu(density_enc)
        
        return density_enc #[number_of_levels, batch_size, feat_size, 1]

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        # Computing rgb features at all levels
        rgb_features_list = []
        number_of_levels = self.number_of_levels_color
        for lvl in range(number_of_levels):
            color_enconding = self.color_encoding_levels[lvl]
            features_at_lvl = color_enconding(positions)
            rgb_features_list.append(features_at_lvl)

        if self.use_sh:
            raise ValueError("Not implemented for Pyramid.")
        else:
            d_encoded = self.direction_encoding(d)
            rgb_features = self.B(torch.stack(rgb_features_list,dim=0).sum(0)) # Aggregation
            rgb_features_encoded = self.feature_encoding(rgb_features)
            out = self.mlp_head(torch.cat(\
                    [rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))
            rgb = self.field_output_rgb(out)
        return rgb #[batch_size, feat_size, 3]
    
    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[Tensor] = None,
        bg_color: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals is True:
            raise ValueError("Surface normals are not currently supported with TensoRF")
        if mask is not None and bg_color is not None:
            base_density = torch.zeros(*ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(*(ray_samples[:, :, None]).shape)
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


class PytensoRFImageField(Field):
    """PytensoRF Field using image approach"""

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
        self.number_of_levels_density = len(density_encoding_levels)
        self.number_of_levels_color = len(color_encoding_levels)

        self.mlp_heads = []
        for lvl in range(self.number_of_levels_color):
            self.mlp_heads.append(MLP(
            in_dim=appearance_dim + 3 + self.direction_encoding.get_out_dim() + self.feature_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        ))
        

        self.use_sh = use_sh

        if self.use_sh:
            self.sh = SHEncoding(sh_levels)
            out_features_B = 3 * self.sh.get_out_dim()
        else:
            out_features_B = appearance_dim
        self.Bs = []

        for lvl in range(self.number_of_levels_color):
            self.Bs.append(nn.Linear(in_features=self.color_encoding_levels[0].get_out_dim(), \
                                     out_features=out_features_B, bias=False) \
            )

        # Use only for coarse
        self.field_output_rgbs = []
        for lvl in range(self.number_of_levels_color):
            # Only coarsest level gets Sigmoid activation
            activation = nn.Tanh() if lvl < self.number_of_levels_color - 1 else nn.Sigmoid()
            layers = []
            if lvl < self.number_of_levels_color:
                layers.append(ZeroLinear(self.mlp_heads[0].get_out_dim(),3))
            layers.append(activation)

            rgb_head = torch.nn.Sequential(*layers)
            self.field_output_rgbs.append(rgb_head)
            # self.field_output_rgbs.append(\
            #     RGBFieldHead(in_dim=self.mlp_heads[0].get_out_dim(), \
            #                  activation=activation))
        
        # Turin into ModuleLists
        self.mlp_heads = torch.nn.ModuleList(self.mlp_heads)
        self.Bs = torch.nn.ModuleList(self.Bs)
        self.field_output_rgbs = torch.nn.ModuleList(self.field_output_rgbs)

    def get_density(self, ray_samples: RaySamples) -> Tensor:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        densities_list = []
        for lvl in range(self.number_of_levels_density):
            density_encoding = self.density_encoding_levels[lvl]
            density_at_lvl = density_encoding(positions)
            densities_list.append(density_at_lvl)
        density = torch.stack(densities_list,dim=0).sum(0) # Density should always be a shared value
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        relu = torch.nn.ReLU()
        density_enc = relu(density_enc)
        
        return density_enc #[number_of_levels, batch_size, feat_size, 1]

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1

        # Computing rgb features at all levels
        rgb_features_list = []
        number_of_levels = self.number_of_levels_color
        for lvl in range(number_of_levels):
            color_enconding = self.color_encoding_levels[lvl]
            features_at_lvl = color_enconding(positions)
            rgb_features_list.append(features_at_lvl)

        if self.use_sh:
            raise ValueError("Not implemented for Pyramid.")
        else:
            d_encoded = self.direction_encoding(d)
            outputs = []
            for lvl in range(number_of_levels):
                rgb_feat_at_lvl = self.Bs[lvl](rgb_features_list[lvl])
                rgb_features_encoded = self.feature_encoding(rgb_feat_at_lvl)
                out = self.mlp_heads[lvl](torch.cat(\
                    [rgb_feat_at_lvl, d, rgb_features_encoded, d_encoded], dim=-1))  # type: ignore
                out = self.field_output_rgbs[lvl](out)
                outputs.append(out)
            rgb = torch.stack(outputs,dim=0)
        return rgb #[number_of_levels, batch_size, feat_size, 3]
    
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
            base_density = torch.zeros(*ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(self.number_of_levels_color, *(ray_samples[:, :, None]).shape)
            if mask.any():
                input_rays = ray_samples[mask, :]
                density = self.get_density(input_rays)
                rgb = self.get_outputs(input_rays, None)

                base_density[mask] = density
                base_rgb[:,mask] = rgb

                base_density.requires_grad_()
                base_rgb.requires_grad_()

            density = base_density
            rgb = base_rgb
        else:
            density = self.get_density(ray_samples)
            rgb = self.get_outputs(ray_samples, None)
        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}