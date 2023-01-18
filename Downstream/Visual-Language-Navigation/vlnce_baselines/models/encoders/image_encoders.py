import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import spaces
from habitat import logger
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
import clip
import torchvision

class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            # self.visual_fc = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(
            #         np.prod(self.visual_encoder.output_shape), output_size
            #     ),
            #     nn.ReLU(True),
            # )
            None
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)


    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            # return self.visual_fc(x)
            return x


class TorchVisionResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self,
        observation_space,
        output_size,
        device,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.device = device
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            obs_size_0 = observation_space.spaces["rgb"].shape[0]
            obs_size_1 = observation_space.spaces["rgb"].shape[1]
            if obs_size_0 != 224 or obs_size_1 != 224:
                logger.warn(
                    "TorchVisionResNet50: observation size is not conformant to expected ResNet input size [3x224x224]"
                )
            linear_layer_input_size += self.resnet_layer_size
        else:
            self._n_input_rgb = 0

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        rgb_resnet = models.resnet50(pretrained=True)
        rgb_modules = list(rgb_resnet.children())[:-2]
        self.cnn = torch.nn.Sequential(*rgb_modules)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad_(False)
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            # self.fc = nn.Linear(linear_layer_input_size, output_size)
            # self.activation = nn.ReLU()
            None
        else:
            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x
            self.cnn.avgpool = SpatialAvgPool()
            self.cnn.fc = nn.Sequential()
            self.spatial_embeddings = nn.Embedding(4 * 4, 64)
            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

        # self.layer_extract = self.cnn._modules.get("avgpool")

        from torchvision import transforms
        self.rgb_transform = torch.nn.Sequential(
            # transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            )

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            # resnet_output = torch.zeros(
            #     1, dtype=torch.float32, device=observation.device
            # )
            # def hook(m, i, o):
            #     resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            # h = self.layer_extract.register_forward_hook(hook)
            resnet_output = self.cnn(observation)
            # h.remove()
            return resnet_output

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)

            rgb_observations = self.rgb_transform(rgb_observations)
            # rgb_observations = rgb_observations / 255.0  # normalize RGB

            resnet_output = resnet_forward(rgb_observations.contiguous())

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)#.to(self.device)
        else:
            # return self.activation(
            #     self.fc(torch.flatten(resnet_output, 1))
            # )  # [BATCH x OUTPUT_DIM]
            return resnet_output


class CLIPEncoder(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.
    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, device,
    ):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        from torchvision import transforms
        self.rgb_transform = torch.nn.Sequential(
            # transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            )

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """
        rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
        rgb_observations = self.rgb_transform(rgb_observations)
        output = self.model.encode_image(rgb_observations.contiguous())

        return output.float()