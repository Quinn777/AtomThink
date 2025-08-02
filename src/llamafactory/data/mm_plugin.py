import math
import re
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union
import numpy as np
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from typing_extensions import override
from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import is_pillow_available, is_pyav_available, is_transformers_version_greater_than

if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject
if is_pyav_available():
    import av
if is_transformers_version_greater_than("4.45.0"):
    from transformers.models.mllama.processing_mllama import (
        convert_sparse_cross_attention_mask_to_dense,
        get_cross_attention_token_mask,
    )
if TYPE_CHECKING:
    from av.stream import Stream
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor


    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]


    ImageInput = Union[str, bytes, EncodedImage, ImageObject]
    VideoInput = str


def _get_paligemma_token_type_ids(
        imglens: Sequence[int], seqlens: Sequence[int], processor: "ProcessorMixin"
) -> List[List[int]]:
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * getattr(processor, "image_seqlen")
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


class BasePlugin:
    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token
        self.expand_mm_tokens = True

    def _validate_input(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
    ) -> None:
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )
        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if (image.width * image.height) > image_resolution:
            resize_factor = math.sqrt(image_resolution / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream: "Stream", **kwargs) -> int:
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])
            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")
            results.append(self._preprocess_image(image, **kwargs))
        return results

    def _regularize_videos(self, videos: Sequence["VideoInput"], **kwargs) -> List[List["ImageObject"]]:
        r"""
        Regularizes videos to avoid error. Including reading, resizing and converting.
        """
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            total_frames = video_stream.frames
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)
        return results

    def _get_mm_inputs(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 768 * 768),
            )
            input_dict["images"] = images
        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 256 * 256),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            input_dict["videos"] = videos
        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(
            self,
            messages: Sequence[Dict[str, str]],
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        return messages

    def process_token_ids(
            self,
            input_ids: List[int],
            labels: Optional[List[int]],
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            tokenizer: "PreTrainedTokenizer",
            processor: Optional["ProcessorMixin"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(images, videos)
        return input_ids, labels

    def get_mm_inputs(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            imglens: Sequence[int],
            vidlens: Sequence[int],
            batch_ids: Sequence[List[int]],
            processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return {}


class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
            self,
            messages: Sequence[Dict[str, str]],
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        image_seqlen = getattr(processor, "image_seqlen") if self.expand_mm_tokens else 1
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", self.image_token)
        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")
        return messages

    @override
    def get_mm_inputs(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            imglens: Sequence[int],
            vidlens: Sequence[int],
            batch_ids: Sequence[List[int]],
            processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
            self,
            messages: Sequence[Dict[str, str]],
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)
        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages

    @override
    def _get_mm_inputs(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            processor: "ProcessorMixin",
            **kwargs,
    ) -> Dict[str, "torch.Tensor"]:
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        imglens: List[int] = kwargs["imglens"]
        images = self._regularize_images(images, image_resolution=getattr(processor, "image_resolution", 768 * 768))
        batch_images = []
        for image_length in imglens:
            batch_images.append(images[:image_length])
            images = images[image_length:]
        return image_processor(batch_images, return_tensors="pt")

    def get_mm_inputs(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            imglens: Sequence[int],
            vidlens: Sequence[int],
            batch_ids: Sequence[List[int]],
            processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        mm_inputs = self._get_mm_inputs(images, videos, processor, imglens=imglens)
        num_tiles = mm_inputs.pop("num_tiles")
        image_token_id = getattr(processor, "image_token_id")
        max_image_tiles = getattr(processor.image_processor, "max_image_tiles")
        cross_attention_token_mask = [
            get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
        ]
        mm_inputs["cross_attention_mask"] = torch.from_numpy(
            convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=max_image_tiles,
                length=max(len(input_ids) for input_ids in batch_ids),
            )
        )  # shape: (batch_size, length, max_num_images, max_num_tiles)
        return mm_inputs


PLUGINS = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
    "mllama": MllamaPlugin,

}


def get_mm_plugin(
        name: str,
        image_token: Optional[str] = None,
        video_token: Optional[str] = None,
) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return plugin_class(image_token, video_token)