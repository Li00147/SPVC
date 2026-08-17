import json
import os

import torch

from .operators import (
    ImageCropAndResize,
    LoadTorchPickle,
    LoadVideo,
    RouteByExtensionName,
    RouteByType,
    ToAbsolutePath,
)


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        metadata_path=None,
        repeat=1,
        data_file_keys=(),
        main_data_operator=lambda data: data,
    ):
        self.base_path = base_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
    ):
        frame_processor = ImageCropAndResize(
            height,
            width,
            max_pixels,
            height_division_factor,
            width_division_factor,
        )
        return RouteByType(
            [
                (
                    str,
                    ToAbsolutePath(base_path)
                    >> RouteByExtensionName(
                        [
                            (("pt", "pth", "ckpt"), LoadTorchPickle()),
                            (
                                (
                                    "mp4",
                                    "avi",
                                    "mov",
                                    "wmv",
                                    "mkv",
                                    "flv",
                                    "webm",
                                ),
                                LoadVideo(
                                    num_frames,
                                    time_division_factor,
                                    time_division_remainder,
                                    frame_processor,
                                ),
                            ),
                        ]
                    ),
                )
            ]
        )

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            self.search_for_cached_data_files(self.base_path)
            return
        if not metadata_path.endswith(".json"):
            raise ValueError("dataset_metadata_path must be a JSON file")
        with open(metadata_path, encoding="utf-8") as file:
            self.data = json.load(file)

    def __getitem__(self, data_id):
        if self.load_from_cache:
            return self.cached_data_operator(
                self.cached_data[data_id % len(self.cached_data)]
            )
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        data = self.cached_data if self.load_from_cache else self.data
        return len(data) * self.repeat
