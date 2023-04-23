import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from model import YOLOv5


class TestYOLOv5(unittest.TestCase):

    def setUp(self):
        self.yolo = YOLOv5()
        self.image_url = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"
        self.video_url = "https://www.youtube.com/watch?v=Gr0HpDM8Ki8"
        self.output_location = "output"

    def tearDown(self):
        self.yolo = None

    def test_init_method(self):
        assert self.yolo.model is not None
        assert self.yolo.result is None
        assert self.yolo.object_url is None
        assert self.yolo.object is None
        assert self.yolo.object_type is None
        assert self.yolo.save_dir == os.getcwd()

    def test_set_object_method(self):
        object_path = self.image_url
        object_type = "image"
        self.yolo.set_object(object_path, object_type, self.output_location)
        assert self.yolo.object_url == object_path
        assert self.yolo.object_type == object_type
        assert self.yolo.save_dir == self.output_location

    def test_load_image_object_method(self):
        object_path = self.image_url
        object_type = "image"
        self.yolo.set_object(object_path, object_type, self.output_location)
        self.yolo.load_object()
        assert isinstance(self.yolo.object, np.ndarray)

    def test_load_object_with_invalid_url(self):
        yolo = YOLOv5()
        yolo.set_object("invalid_url", "image", self.output_location)
        with self.assertRaises(ValueError):
            yolo.load_object()

    def test_detect_and_save_with_image(self):
        np_array = np.random.randint(0, 255, size=(10, 10, 3)).astype(np.uint8)
        self.yolo.set_object(self.image_url, "image", self.output_location)
        self.yolo.object = np_array
        with patch.object(self.yolo.model, 'predict') as mock_predict:
            mock_predict.return_value = "test_result"
            result = self.yolo.detect_and_save()
            self.assertEqual(result, "test_result")

    def test_detect_video_objects(self):
        object_path = self.video_url
        object_type = "video"
        self.yolo.set_object(object_path, object_type, self.output_location)
        self.yolo.load_object()
        result = self.yolo.detect_and_save()
        assert os.path.isfile(f"{os.getcwd()}/{self.output_location}/video.mp4")
        assert result is not None

