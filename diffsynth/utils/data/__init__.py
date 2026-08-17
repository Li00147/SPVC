import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


class LowMemoryVideo:

    def __init__(self, file_name):
        self.reader = imageio.get_reader(file_name)

    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return Image.fromarray(np.asarray(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        self.reader.close()


def crop_and_resize(image, height, width):
    image = np.asarray(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        cropped_width = int(image_height / height * width)
        left = (image_width - cropped_width) // 2
        image = image[:, left : left + cropped_width]
    else:
        cropped_height = int(image_width / width * height)
        top = (image_height - cropped_height) // 2
        image = image[top : top + cropped_height]
    return Image.fromarray(image).resize((width, height))


class VideoData:

    def __init__(self, video_file, height=None, width=None):
        self.data = LowMemoryVideo(video_file)
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        frame = self.data[item]
        if self.height is not None and self.width is not None:
            if frame.size != (self.width, self.height):
                frame = crop_and_resize(frame, self.height, self.width)
        return frame


def save_video(frames, save_path, fps, quality=9):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame in tqdm(frames, desc="Saving video"):
        writer.append_data(np.asarray(frame))
    writer.close()
