import os

import imageio
import torch
import torchvision
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators = [] if operators is None else operators

    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class ImageCropAndResize(DataProcessingOperator):
    def __init__(
        self,
        height=None,
        width=None,
        max_pixels=None,
        height_division_factor=1,
        width_division_factor=1,
    ):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def get_height_width(self, image):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        width, height = image.size
        if width * height > self.max_pixels:
            scale = (width * height / self.max_pixels) ** 0.5
            height, width = int(height / scale), int(width / scale)
        height = height // self.height_division_factor * self.height_division_factor
        width = width // self.width_division_factor * self.width_division_factor
        return height, width

    def __call__(self, image):
        target_height, target_width = self.get_height_width(image)
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return torchvision.transforms.functional.center_crop(
            image, (target_height, target_width)
        )


class LoadVideo(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda frame: frame,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor

    def __call__(self, path):
        reader = imageio.get_reader(path)
        num_frames = min(self.num_frames, int(reader.count_frames()))
        while (
            num_frames > 1
            and num_frames % self.time_division_factor != self.time_division_remainder
        ):
            num_frames -= 1
        frames = [
            self.frame_processor(Image.fromarray(reader.get_data(frame_id)))
            for frame_id in range(num_frames)
        ]
        reader.close()
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, path):
        extension = path.rsplit(".", 1)[-1].lower()
        for extensions, operator in self.operator_map:
            if extension in extensions:
                return operator(path)
        raise ValueError(f"Unsupported file: {path}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data):
        for data_type, operator in self.operator_map:
            if isinstance(data, data_type):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location

    def __call__(self, path):
        return torch.load(path, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, path):
        return os.path.join(self.base_path, path)
