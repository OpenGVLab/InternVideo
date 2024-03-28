import copy
import numbers
from typing import Dict, List, Tuple, Union

import torch
from gym import spaces
from habitat.config import Config
from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import (
    center_crop,
    get_image_height_width,
    overwrite_gym_box_shape,
)
from torch import Tensor


@baseline_registry.register_obs_transformer()
class CenterCropperPerSensor(ObservationTransformer):
    """An observation transformer that center crops your input on a per-sensor basis."""

    sensor_crops: Dict[str, Union[int, Tuple[int, int]]]
    channels_last: bool

    def __init__(
        self,
        sensor_crops: List[Tuple[str, Union[int, Tuple[int, int]]]],
        channels_last: bool = True,
    ):
        """Args:
        size: A sequence (h, w) or int of the size you wish to resize/center_crop.
                If int, assumes square crop
        channels_list: indicates if channels is the last dimension
        trans_keys: The list of sensors it will try to centercrop.
        """
        super().__init__()

        self.sensor_crops = dict(sensor_crops)
        for k in self.sensor_crops:
            size = self.sensor_crops[k]
            if isinstance(size, numbers.Number):
                self.sensor_crops[k] = (int(size), int(size))
            assert len(size) == 2, "forced input size must be len of 2 (h, w)"

        self.channels_last = channels_last

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        observation_space = copy.deepcopy(observation_space)
        for key in observation_space.spaces:
            if (
                key in self.sensor_crops
                and observation_space.spaces[key].shape[-3:-1]
                != self.sensor_crops[key]
            ):
                h, w = get_image_height_width(
                    observation_space.spaces[key], channels_last=True
                )
                logger.info(
                    "Center cropping observation size of %s from %s to %s"
                    % (key, (h, w), self.sensor_crops[key])
                )

                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.sensor_crops[key]
                )
        return observation_space

    @torch.no_grad()
    def forward(self, observations: Dict[str, Tensor]) -> Dict[str, Tensor]:
        observations.update(
            {
                sensor: center_crop(
                    observations[sensor],
                    self.sensor_crops[sensor],
                    channels_last=self.channels_last,
                )
                for sensor in self.sensor_crops
                if sensor in observations
            }
        )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cc_config = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR
        return cls(cc_config.SENSOR_CROPS)

@baseline_registry.register_obs_transformer()
class ResizerPerSensor(ObservationTransformer):
    r"""An nn module the resizes images to any aspect ratio.
    This module assumes that all images in the batch are of the same size.
    """

    def __init__(
        self,
        sizes: int,
        channels_last: bool = True,
        trans_keys: Tuple[str] = ("rgb", "depth", "semantic"),
    ):
        super().__init__()
        """Args:
        size: The size you want to resize
        channels_last: indicates if channels is the last dimension
        """
        self.sensor_resizes = dict(sizes)
        for k in self.sensor_resizes:
            size = self.sensor_resizes[k]
            if isinstance(size, numbers.Number):
                self.sensor_resizes[k] = (int(size), int(size))
            assert len(size) == 2, "forced input size must be len of 2 (h, w)"

        self.channels_last = channels_last

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):

        for key in observation_space.spaces:
            if (
                key in self.sensor_resizes
                and observation_space.spaces[key].shape[-3:-1]
                != self.sensor_resizes[key]
            ):
                h, w = get_image_height_width(
                    observation_space.spaces[key], channels_last=True
                )
                logger.info(
                    "Resizing observation size of %s from %s to %s"
                    % (key, (h, w), self.sensor_resizes[key])
                )

                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.sensor_resizes[key]
                )

        return observation_space

    def _transform_obs(self, obs: torch.Tensor, size) -> torch.Tensor:
        img = torch.as_tensor(obs)
        no_batch_dim = len(img.shape) == 3
        if len(img.shape) < 3 or len(img.shape) > 5:
            raise NotImplementedError()
        if no_batch_dim:
            img = img.unsqueeze(0)  # Adds a batch dimension
        h, w = get_image_height_width(img, channels_last=self.channels_last)
        if self.channels_last:
            if len(img.shape) == 4:
                # NHWC -> NCHW
                img = img.permute(0, 3, 1, 2)
            else:
                # NDHWC -> NDCHW
                img = img.permute(0, 1, 4, 2, 3)

        h, w = size
        img = torch.nn.functional.interpolate(
            img.float(), size=(h, w), mode="area"
        ).to(dtype=img.dtype)
        if self.channels_last:
            if len(img.shape) == 4:
                # NCHW -> NHWC
                img = img.permute(0, 2, 3, 1)
            else:
                # NDCHW -> NDHWC
                img = img.permute(0, 1, 3, 4, 2)
        if no_batch_dim:
            img = img.squeeze(dim=0)  # Removes the batch dimension
        return img

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        observations.update(
            {
                sensor: self._transform_obs(
                    observations[sensor], self.sensor_resizes[sensor])
                for sensor in self.sensor_resizes
                if sensor in observations
            }
        )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        r_config = config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR
        return cls(r_config.SIZES)
