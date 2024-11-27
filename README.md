# **Real-Time Detection and Segmentation**

This project demonstrates a Flask-based web application for real-time **Object Detection** and **Semantic Segmentation** using live video streams. It supports two video sources:

- **Default camera**
- **Mobile camera stream** via an IPv4 address (e.g., using IP Webcam for Android).  

The app dynamically switches between these video sources and modes, making it an interactive and efficient solution for real-time AI-powered vision tasks.

---

## **Features**

### **Real-Time Video Processing**
- Detect objects with the **YOLOv8** object detection model.  
- Perform semantic segmentation using **DeepLabV3**.  

### **Two Video Sources**
- **Default camera** (`cv2.VideoCapture(0)`).  
- **Mobile camera** via an IP stream (e.g., `http://192.168.x.x:8080/video`).  

### **Dynamic Mode and Source Switching**
- Switch between **Object Detection** and **Semantic Segmentation** modes with the click of a button.  
- Seamlessly change the camera source from the default webcam to a mobile camera.  

---

## **Tech Stack**

### **Frameworks and Libraries**
- **Backend**: Flask (Python)  
- **AI Models**:  
  - YOLOv8 for Object Detection  
  - DeepLabV3 (TorchVision) for Semantic Segmentation  
- **Frontend**: HTML, JavaScript, and CSS  
- **Computer Vision**: OpenCV  

### **Dependencies**
- Python 3.7+  
- Flask  
- OpenCV  
- PyTorch and TorchVision  
- ultralytics (for YOLOv8)  

---

## **How to Use**

### **Default Camera**
- When the app starts, it uses the **default camera** (e.g., built-in webcam or USB camera).  

### **Switch to Mobile Camera**
- Click the **"Switch to Mobile Camera"** button.  
- Enter the IPv4 stream URL of your mobile camera (e.g., `http://192.168.x.x:8080/video`).  

### **Change Modes**
- Use the **"Object Detection"** or **"Semantic Segmentation"** buttons to toggle between these modes.  

### **Real-Time Output**
- View the live video stream with object detection or segmentation applied in real time.  


`python -m venv venv`
`source venv/bin/activate  # On Windows: venv\Scripts\activate`
`pip install -r requirements.txt`

`python run.py`
