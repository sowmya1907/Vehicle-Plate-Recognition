# License Plate Recognition

## Overview
License Plate Recognition (LPR) is a project designed to detect and recognize vehicle license plates from images. It involves two main tasks:
1. **License Plate Detection**: Identifying and localizing license plates in vehicle images.
2. **Character Recognition**: Extracting and recognizing the alphanumeric text from detected license plates.

## Features
- **Dataset Handling**: Uses a dataset of 900 vehicle images with annotated license plates and 900 license plate images with text annotations.
- **Preprocessing**: Includes image normalization, grayscale conversion, thresholding, and data augmentation.
- **License Plate Detection**: Utilizes YOLOv8 or Faster R-CNN for detecting license plates.
- **Character Recognition**: Implements OCR models such as Tesseract OCR, CRNN, or Transformer-based OCR.
- **Evaluation Metrics**: Measures performance using Intersection over Union (IoU) and Levenshtein Distance for text accuracy.
- **Deployment**: Can be deployed as a web application using Flask or Streamlit.

## Prerequisites
- Python 3.11.9
- Required Libraries: OpenCV, NumPy, Pandas, Matplotlib
- Deep Learning Frameworks: TensorFlow/PyTorch
- YOLOv8 (Ultralytics) for object detection
- Tesseract OCR / EasyOCR for character recognition
- Flask/Streamlit for web deployment

## Installation
1. Install required dependencies:
   ```bash
   pip install numpy opencv-python matplotlib pandas ultralytics pytesseract flask streamlit
   ```
2. Install YOLOv8:
   ```bash
   pip install ultralytics
   ```
3. Install and configure Tesseract OCR:
   ```bash
   sudo apt install tesseract-ocr
   pip install pytesseract
   ```
4. Clone the dataset and organize it properly in your directory.
5. Train the models (YOLO for detection, OCR for recognition).

## Usage
### Train the Models
```bash
python train_yolo.py  # Train the YOLO model
python train_ocr.py    # Train the OCR model
```

### Run License Plate Detection & Recognition
```bash
python detect_license_plate.py --image test_image.jpg
```

### Deploy Web App
```bash
python app.py
```
Users can upload an image via the web interface or API to receive the recognized license plate number.

## Evaluation Metrics
- **License Plate Detection**: IoU (Intersection over Union) and Precision-Recall.
- **Character Recognition**: Levenshtein Distance (Edit Distance) for text accuracy.

## Example Outputs
- Sample detected license plates before OCR.
- OCR-extracted text results.
- Model performance metrics (confusion matrix, precision-recall curves).

## Contributors
- Myself (Sowmya)

## License
This project is licensed under the **MIT License**.

---
Would you like to add more details or modify anything in the README? 🚀

