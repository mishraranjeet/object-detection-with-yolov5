import os
import ssl
import time
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np
from pytube import YouTube
from ultralytics import YOLO

# This restores the same behavior as before.
context = ssl._create_unverified_context()
ssl._create_default_https_context = ssl._create_stdlib_context

INPUT_FILES = os.getcwd()
FILE_NAME = "video.mp4"


class YOLOv5:
    def __init__(self, model_name: str = "yolov5su.pt"):
        # self.model = yolov5.load(model_name)
        self.model = YOLO(model_name)
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image
        self.result = None
        self.object_url = None
        self.object = None
        self.object_type = None
        self.save_dir = os.getcwd()

    def set_object(self, object_path: str, object_type: str, save_dir: str):
        self.object_url = object_path
        self.object_type = object_type
        self.save_dir = save_dir

    def load_object(self):
        object_url = urlparse(self.object_url)
        if not all([object_url.scheme, object_url.netloc]):
            raise ValueError("Object URL is not Valid")

        if self.object_type == "video":
            try:
                yt = YouTube(self.object_url)
                print("Downloading video file !!!")
                yt.streams.first().download(output_path=INPUT_FILES, filename=FILE_NAME)
                print("Saving video file !!!")
                while True:
                    if os.path.exists(f"{INPUT_FILES}/{FILE_NAME}"):
                        break
                    print("Still saving !!!!", f"{INPUT_FILES}/{FILE_NAME}")
                    time.sleep(1)  # Wait for one second
            except Exception as e:
                raise FileNotFoundError("Failed to import file", e)
            if os.path.isfile(f"{INPUT_FILES}/{FILE_NAME}"):
                self.object = f"{INPUT_FILES}/{FILE_NAME}"
            else:
                raise FileNotFoundError("Failed to import file")
        else:
            object_file = urlopen(self.object_url, context=context)
            # Convert the image to a numpy array
            image_array = np.frombuffer(bytearray(object_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            self.object = image

    def detect_and_save(self):
        if self.object is None:
            print("Loading objects !!!")
            self.load_object()
        try:
            print(f"Detecting objects in file {self.object}")
            result = self.model.predict(source=self.object, save=True, project=os.getcwd(),
                                        name=self.save_dir)
        except Exception as e:
            print("Object detection failed", e)
            raise ValueError("Object detection failed")

        finally:
            self.remove_downloaded_file()
        return result

    def remove_downloaded_file(self):
        if os.path.isfile(self.object):
            print("Removing file after detection")
            os.remove(self.object)
