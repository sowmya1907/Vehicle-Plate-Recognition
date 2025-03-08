# 🚗 License Plate Recognition  

## 📌 Overview  
License Plate Recognition (LPR) is a project designed to detect and recognize vehicle license plates from images. It involves two main tasks:  
1. **License Plate Detection**: Identifying and localizing license plates in vehicle images using **YOLOv8**.  
2. **Character Recognition**: Extracting and recognizing the alphanumeric text from detected license plates using **OCR models**.  

## 🏆 Features  
- **License Plate Detection**: Utilizes **YOLOv8** for real-time object detection.  
- **Character Recognition**: Implements OCR models such as **Tesseract OCR, CRNN, or Transformer-based OCR**.  
- **Dataset Handling**: Uses a dataset of 900+ annotated vehicle and license plate images.  
- **Preprocessing**: Includes image normalization, grayscale conversion, and thresholding.  
- **Deployment**: Can be deployed as a **web application using Flask or Streamlit**.  

## 🔧 Prerequisites  
- **Python 3.11.9**  
- **Deep Learning Frameworks**: PyTorch  
- **Object Detection**: YOLOv8 (Ultralytics)  
- **OCR Tools**: Tesseract OCR / EasyOCR  
- **Libraries**: OpenCV, NumPy, Pandas, Matplotlib  
- **Deployment**: Flask/Streamlit  

## ⚙️ Installation  
1️⃣ Install required dependencies:  
   ```bash
   pip install numpy opencv-python matplotlib pandas ultralytics pytesseract flask streamlit
   ```
2️⃣ Install YOLOv8 (Ultralytics):  
   ```bash
   pip install ultralytics
   ```
3️⃣ Install and configure Tesseract OCR:  
   ```bash
   sudo apt install tesseract-ocr
   pip install pytesseract
   ```
4️⃣ Clone the dataset and organize it properly in your directory.  
5️⃣ Train the models (YOLO for detection, OCR for recognition).  

## 🚀 Usage  
### 🔹 License Plate Detection Using YOLOv8  
```python
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO(r"C:\Users\sowmy\Downloads\yolov8n (1).pt")

# Perform inference on an image
results = model(r"sample_image.jpg", save=True, conf=0.5)

# Display results
for result in results:
    print(result)
```

### 🔹 Run OCR for Character Recognition  
```python
import cv2
import pytesseract

# Load detected plate image
image = cv2.imread("detected_plate.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply OCR
text = pytesseract.image_to_string(gray, config='--psm 7')

print("Recognized License Plate:", text)
```

### 🔹 Deploy Web App  
```bash
python app.py
```
Users can upload an image via the web interface or API to receive the recognized license plate number.  

## 📄 YAML Configuration File  
The project includes a **YAML configuration file** to define **model parameters, dataset paths, and training settings**.  

Example `config.yaml` file for YOLO:  
```yaml
path: data/  # Dataset root directory
train: train/images/  # Training images
val: valid/images/  # Validation images
test: test/images/  # Test images

nc: 1  # Number of classes (License Plate)
names: ["License_Plate"]  # Class names

training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  optimizer: "Adam"
```
🔹 Modify `config.yaml` to **adjust settings without modifying the main codebase**.  

## 📊 Evaluation Metrics  
- **License Plate Detection**: IoU (Intersection over Union), Precision, Recall  
- **Character Recognition**: Levenshtein Distance (Edit Distance) for text accuracy  

## 📸 Example Outputs  
- Sample detected license plates using YOLOv8  
- OCR-extracted text results  
- Model performance metrics (confusion matrix, precision-recall curves)  

## 👩‍💻 Contributors  
- **Sowmya Guduguntla**  

## 📜 License  
None  
