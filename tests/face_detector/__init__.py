from unittest import TestCase

from src.face_detector import MTCNNFaceDetector


class TestDetection(TestCase):
    detector = MTCNNFaceDetector()
