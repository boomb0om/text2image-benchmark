from T2IBenchmark.utils.resizers import Resizer
import numpy as np
from unittest import TestCase


class TestResizer(TestCase):

    def test_init(self):
        resizer = Resizer(
            lib='PIL',
            filter_name='bicubic',
            quantize_after=True,
            output_size=(200, 200),
            center_crop=True
        )
        self.assertEqual(resizer.lib, 'PIL')
        self.assertEqual(resizer.filter_name, 'bicubic')
        self.assertEqual(resizer.quantize_after, True)
        self.assertEqual(resizer.output_size, (200, 200))
        self.assertEqual(resizer.center_crop, True)

    def test_call_with_PIL_and_quantize_after(self):
        resizer = Resizer(
            lib='PIL',
            filter_name='bicubic',
            quantize_after=True,
            output_size=(100, 100),
            center_crop=False
        )
        img = np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8)
        resized_img = resizer(img)
        self.assertIsInstance(resized_img, np.ndarray)
        self.assertEqual(resized_img.shape, (100, 100, 3))
        self.assertEqual(resized_img.dtype, np.uint8)

    def test_call_with_PIL_and_without_quantize_after(self):
        resizer = Resizer(
            lib='PIL',
            filter_name='bicubic',
            quantize_after=False,
            output_size=(100, 100),
            center_crop=False
        )
        img = np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8)
        resized_img = resizer(img)
        self.assertIsInstance(resized_img, np.ndarray)
        self.assertEqual(resized_img.shape, (100, 100, 3))
        self.assertEqual(resized_img.dtype, np.float32)

    def test_call_with_OpenCV_and_quantize_after(self):
        resizer = Resizer(
            lib='OpenCV',
            filter_name='bicubic',
            quantize_after=True,
            output_size=(100, 100),
            center_crop=False
        )
        img = np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8)
        resized_img = resizer(img)
        self.assertIsInstance(resized_img, np.ndarray)
        self.assertEqual(resized_img.shape, (100, 100, 3))
        self.assertEqual(resized_img.dtype, np.uint8)

    def test_call_with_OpenCV_and_without_quantize_after(self):
        resizer = Resizer(
            lib='OpenCV',
            filter_name='bicubic',
            quantize_after=False,
            output_size=(100, 100),
            center_crop=False
        )
        img = np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8)
        resized_img = resizer(img)
        self.assertIsInstance(resized_img, np.ndarray)
        self.assertEqual(resized_img.shape, (100, 100, 3))
        self.assertEqual(resized_img.dtype, np.uint8)

    def test_call_with_invalid_lib_and_invalid_filter(self):
        with self.assertRaises(AssertionError):
            Resizer(
                lib='InvalidLib',
                filter_name='InvalidFilter',
                quantize_after=True,
                output_size=(100, 100),
                center_crop=False
            )
