"""Video processor class for Molmo2"""
from typing import TYPE_CHECKING, Tuple, List, Optional, Union, Dict, Any
import numpy as np
import einops
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype

from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from transformers.video_utils import (
    VideoInput,
    valid_videos,
    make_batched_videos,
)
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.utils import logging
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging, to_numpy


if TYPE_CHECKING:
    from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)


def normalize_image(
    image: np.ndarray,
    image_mean: List[float],
    image_std: List[float],
) -> np.ndarray:
    image -= np.array(image_mean, dtype=np.float32)[None, None, :]
    image /= np.array(image_std, dtype=np.float32)[None, None, :]
    return image


def resize_image(
    image: np.ndarray,
    desired_output_size: List[int],
    resample: PILImageResampling,
) -> np.ndarray:
    if len(image.shape) == 3:
        is_video = False
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    else:
        is_video = True
        image = torch.permute(torch.from_numpy(image), [0, 3, 1, 2])
    dtype = image.dtype
    if torch.is_floating_point(image):
        in_min = 0.0
        in_max = 1.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, "SigLIP expects float images or uint8 images, but got {}".format(image.dtype)
        in_min = 0.0
        in_max = 255.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            resample,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)

    if is_video:
        resized = torch.permute(resized, [0, 2, 3, 1]).numpy()
    else:
        resized = torch.permute(resized, [1, 2, 0]).numpy()

    return resized


def build_resized_image(
    image: np.ndarray,
    base_image_input_size: List[int],
    resample: PILImageResampling,
    image_mean: List[float],
    image_std: List[float],
    image_patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    resized = resize_image(
        image, base_image_input_size, resample,
    )
    resized = normalize_image(resized, image_mean, image_std)
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, 0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
    return resized, resize_idx


def batch_pixels_to_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size*c])
        return array


def arange_for_pooling(
    idx_arr: np.ndarray,
    pool_h: int,
    pool_w: int,
) -> np.ndarray:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad//2, (h_pad+1)//2], [w_pad//2, (w_pad+1)//2]],
                     mode='constant',constant_values=-1)
    return einops.rearrange(
        idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


def image_to_patches_and_grids(
    image: ImageInput,
    base_image_input_size: List[int],
    resample: PILImageResampling,
    image_mean: List[float],
    image_std: List[float],
    image_patch_size: int,
    image_pooling_w: int,
    image_pooling_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :return image_grids, the shape of each image after pooling
    :return crops, the image crops to processes with the ViT
    :return pooled_patch_idx, for each patch_id tokens in `image_tokens`, the indices of the
                                patches in `crops` to pool for that token, masked with -1
    """
    if isinstance(base_image_input_size, int):
        base_image_input_size = (base_image_input_size, base_image_input_size)
    
    pooling_w = image_pooling_w
    pooling_h = image_pooling_h

    resized, resize_idx = build_resized_image(
        image,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )
    pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])
    image_grid = [h, w]
    return (
        image_grid,
        batch_pixels_to_patches(resized, image_patch_size),
        pooling_idx,
    )


class Molmo2VideoProcessorKwargs(VideosKwargs, total=False):
    patch_size: Optional[int]
    pooling_size: Optional[List[int]]


class Molmo2VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"height": 378, "width": 378}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    pooling_size = [3, 3]
    valid_kwargs = Molmo2VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_token_pooling", "video_grids"]

    def __init__(self, **kwargs: Unpack[Molmo2VideoProcessorKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("height", None) is None or self.size.get("width", None) is None
        ):
            raise ValueError("size must contain 'height' and 'width' keys.")

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if size is not None and ("height" not in size or "width" not in size):
            raise ValueError("size must contain 'height' and 'width' keys.")

        return super()._further_process_kwargs(size=size, **kwargs)
    
    def preprocess(
        self,
        videos: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        patch_size: Optional[int] = None,
        pooling_size: Optional[List[int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess a video for the model.
        Args:
            videos (`VideoInput`):
                Video to preprocess.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use when resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            pooling_size (`list[int]`, *optional*, defaults to `self.pooling_size`):
                The pooling size of the vision adapter.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.

        Returns:
            A `BatchFeature` containing the following keys:
                - `pixel_values_videos`: The preprocessed videos.
                - `video_token_pooling`: The indices of the patches in `crops` to pool for each token in `video_tokens`.
                - `video_grids`: The video grids.
        """
        videos = make_batched_videos(videos)

        if size is not None:
            if "height" not in size or "width" not in size:
                raise ValueError("size must contain 'height' and 'width' keys.")
        else:
            size = {**self.size}
        
        base_image_input_size = [size["height"], size["width"]]

        resample = resample or self.resample
        image_mean = image_mean or self.image_mean
        image_std = image_std or self.image_std
        do_convert_rgb = do_convert_rgb or self.do_convert_rgb

        patch_size = patch_size or self.patch_size
        pooling_size = pooling_size or self.pooling_size

        image_pooling_h, image_pooling_w = pooling_size

        # All transformations expect numpy arrays.
        videos = [to_numpy(video) for video in videos]

        batch_grids = []
        batch_crops = []
        batch_pooled_patches_idx = []

        for video in videos:
            all_crops = []
            pooled_patches_idx = []

            for frame in video:
                image_grid, crops, pooled_idx = image_to_patches_and_grids(
                    frame,
                    base_image_input_size,
                    resample,
                    image_mean,
                    image_std,
                    patch_size,
                    image_pooling_w,
                    image_pooling_h,
                )
                offset = sum(np.prod(x.shape[:2]) for x in all_crops)
                pooled_idx_with_offset = np.where(pooled_idx >= 0, pooled_idx + offset, pooled_idx)
                pooled_patches_idx.append(pooled_idx_with_offset)
                all_crops.append(crops)

            video_grid = np.array([len(video), image_grid[0], image_grid[1]])
            all_crops = np.concatenate(all_crops, 0)
            pooled_patches_idx = np.concatenate(pooled_patches_idx, 0)

            batch_grids.append(video_grid)
            batch_crops.append(all_crops)
            batch_pooled_patches_idx.append(pooled_patches_idx)
        
        video_grids = np.stack(batch_grids, 0)
        pixel_values_videos = np.concatenate(batch_crops, 0)
        video_token_pooling = np.concatenate(batch_pooled_patches_idx, 0)
        
        data =dict(
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )

        return BatchFeature(data, tensor_type=return_tensors)


Molmo2VideoProcessor.register_for_auto_class()