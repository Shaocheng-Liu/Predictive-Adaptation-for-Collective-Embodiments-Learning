# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Utility to record the environment frames into a video."""
import os

import imageio
import torchvision.transforms as transforms
import numpy as np
to_pil_image = transforms.ToPILImage()


class VideoRecorder(object):
    def __init__(self, dir_name, height=720, width=1280, camera_id=0, fps=30, video_format="mp4"):
        """Class to record the environment frames into a video.

        Args:
            dir_name ([type]): directory to save the recording.
            height (int, optional): height of the frame. Defaults to 720.
            width (int, optional): width of the frame. Defaults to 1280.
            camera_id (int, optional): id of the camera for recording. Defaults to 0.
            fps (int, optional): frames-per-second for the recording. Defaults to 30.
            video_format (str, optional): video format ('mp4' or 'gif'). Defaults to 'mp4'.
        """
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []
        self.video_format = video_format.lower()

    def reset(self):
        """Initialize the recorder.

        Args:
            enabled (bool, optional): should enable the recorder or not. Defaults to True.
        """
        self.frames = []

    def init(self, enabled=True):
        """Initialize the recorder.

        Args:
            enabled (bool, optional): should enable the recorder or not. Defaults to True.
        """
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, frame, env=None):
        """Record the frames.

        Args:
            env ([type]): environment to record the frames.
        """
        #if self.enabled:
        #    if frame is None:
        #        assert env is not None
        #        frame = env.render(
        #            mode="rgb_array",
        #            height=self.height,
        #            width=self.width,
        #            camera_id=self.camera_id,
        #        )
        #    self.frames.append(frame)
        self.frames.append(frame)

    def save(self, file_name):
        """Save the frames as video to `self.dir_name` in a file named `file_name`.

        Args:
            file_name ([type]): name of the file to store the video frames.
        """
        #if self.enabled:
        #    path = os.path.join(self.dir_name, file_name)
        #    imageio.mimsave(path, self.frames, fps=self.fps)
        imgs = [np.array(to_pil_image(img)) for img in self.frames]
        
        if self.video_format == "mp4":
            # Save as high-quality MP4 video
            output_path = os.path.join(self.dir_name, f'{file_name}.mp4')
            # Use imageio-ffmpeg for high-quality MP4 encoding
            imageio.mimsave(output_path, imgs, fps=self.fps, codec='libx264', pixelformat='yuv420p')
        else:
            # Save as GIF (lower quality, for backwards compatibility)
            output_path = os.path.join(self.dir_name, f'{file_name}.gif')
            imageio.mimsave(output_path, imgs, fps=self.fps)
