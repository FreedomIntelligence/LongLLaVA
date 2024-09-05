import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoProcessor, AutoModel, SiglipImageProcessor


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.resamplePooling = getattr(args, 'resamplePooling', None)
        

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)


    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        if "siglip" in self.vision_tower_name:
            self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name,)
            self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.vision_tower.requires_grad_(False)
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.vision_tower.requires_grad_(False)        

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                if "siglip" in self.vision_tower_name:
                    image_forward_out = self.vision_tower.vision_model(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                else:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                torch.cuda.empty_cache()
                image_features.append(image_feature)
        else:
            if "siglip" in self.vision_tower_name:
                image_forward_outs = self.vision_tower.vision_model(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
            
        if self.resamplePooling:
            if '2d' == self.resamplePooling:
                final_shape = (-1, 144)
                new_shape = (-1, 24, 24)
                image_features = torch.stack([
                    F.interpolate(feature.permute(1, 0).view(*new_shape).unsqueeze(0), size=(12, 12), mode='bilinear', align_corners=False).view(*final_shape).permute(1, 0)
                    for feature in image_features
                ])
            else:
                image_features = torch.stack([
                    F.interpolate(feature.unsqueeze(0).permute(0, 2, 1), size=(144,), mode='linear', align_corners=False).squeeze(0).permute(1, 0) 
                    for feature in image_features
                ])
                
            
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
