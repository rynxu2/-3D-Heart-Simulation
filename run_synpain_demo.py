"""Demo script to run inference using the trained SynPAIN model."""

import sys
import cv2
import torch
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SynPainConfig
from src.models.predictor import HeartPredictor
from src.data.synpain_loader import SYNPAIN_BINARY_NAMES

def run_demo(image_path: str, model_path: str):
    # 1. Load SynPAIN config
    config = SynPainConfig.from_yaml("synpain_config.yaml")
    config.num_classes = 2 # Đảm bảo là 2 class cho SynPAIN
    
    # 2. Initialize Predictor
    # Lưu ý: Chúng ta sẽ monkey-patch LABEL_NAMES để mapping đúng nhãn Pain/NoPain
    import src.models.predictor as predictor_module
    predictor_module.LABEL_NAMES = SYNPAIN_BINARY_NAMES
    
    logger.info(f"Initializing predictor with model: {model_path}")
    predictor = HeartPredictor(
        model_path=model_path,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3. Run Inference
    logger.info(f"Processing image: {image_path}")
    result = predictor.predict_file(image_path)

    if result["face_detected"]:
        print("\n" + "="*30)
        print(f"KẾT QUẢ DỰ ĐOÁN:")
        print(f"Nhãn: {result['label'].upper()}")
        print(f"Độ tin cậy: {result['confidence']:.2%}")
        print(f"Nhịp tim ước tính (BPM): {result['bpm']['min']}-{result['bpm']['max']}")
        print(f"Kiểu nhịp: {result['bpm']['pattern']}")
        print("="*30)
    else:
        print("❌ Không tìm thấy khuôn mặt trong ảnh.")

if __name__ == "__main__":
    # Thay đổi đường dẫn ảnh test của bạn ở đây
    test_img = "data/test_face.jpg" 
    model_pth = "models/synpain_best_model.pth"
    
    if not Path(test_img).exists():
        logger.error(f"Vui lòng để một ảnh khuôn mặt tại {test_img} để test.")
    else:
        run_demo(test_img, model_pth)
