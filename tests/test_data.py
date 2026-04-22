import pytest
import numpy as np
from src.data.face_detector import FaceDetector

class TestFaceDetector:
    @pytest.fixture
    def detector(self):
        return FaceDetector(target_size=224)

    def test_empty_image_handling(self, detector):
        # Arrange: Create a blank black image (no face)
        empty_image = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Act
        faces = detector.detect_faces(empty_image)
        cropped_face = detector.process_image(empty_image)
        
        # Assert
        assert len(faces) == 0
        assert cropped_face is None

    def test_invalid_image_type(self, detector):
        # Act & Assert
        with pytest.raises(Exception):
            detector.detect_faces("not_an_image")
