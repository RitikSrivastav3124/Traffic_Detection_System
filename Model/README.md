<div align="center">

# 🚦 AI Traffic Management System (YOLO-based)

### Real-Time Traffic Congestion Detection & Smart Signal Control using 4 Cameras

</div>

---

## 📌 Overview

This project is a **real-time AI-based Traffic Management System** that uses **YOLO (You Only Look Once)** for vehicle detection and dynamically controls traffic signals based on congestion levels.

The system takes input from **multiple cameras (up to 4)** and performs:

- 🚗 Vehicle Detection
- 📊 Traffic Density Calculation
- 🚦 Smart Signal Switching
- 🚑 Emergency Vehicle Priority Handling

---

## 🚀 Features

- 🔍 Real-time vehicle detection using YOLO
- 📹 Multi-camera support (4 lanes)
- 🚦 Dynamic traffic signal control
- ⏱️ Wait-time based optimization
- 🚑 Emergency vehicle detection (ambulance, police, fire truck)
- ⚡ Fast inference using PyTorch

---

## 🧠 How It Works

1. Input is taken from **4 camera streams** (DroidCam / IP Webcam / CCTV)
2. Each frame is processed using YOLO
3. Vehicles are detected and counted per lane
4. Priority score is calculated:
5. Lane with highest priority gets GREEN signal
6. Emergency vehicles override normal logic

---

## 🛠️ Tech Stack

- Python
- OpenCV
- PyTorch
- YOLOv5 (Ultralytics)
- NumPy

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Go to project folder
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

### 🔹 Single Camera (Testing)

```bash
python your_script.py --weights yolov5s.pt --source 0 --view-img
```
### 🔹 Multiple Camera (Testing)

```bash
python your_script.py --weights yolov5s.pt \
--source 0 1 2 3 \
--view-img
```
---

## 📱 Using Mobile Cameras (DroidCam Setup)

1. Install **DroidCam** app on your phone  
2. Install DroidCam Client on your PC  
3. Connect via WiFi or USB  
4. Start the camera on phone  
5. Use camera index (0,1,2...) in `--source`  

---

## 📂 Project Structure
---
Traffic_Detection/
1. Model/ # YOLO model files
2. utils/ # Helper functions
3. output/ # Output videos
4. your_script.py # Main detection script
5. requirements.txt
6. README.md


---

## ⚙️ Traffic Signal Logic

### Priority Formula:
```bash
Priority = (Vehicle Count × 2) + (Waiting Time × 1)
```


### Signal Rules:

- Minimum Green Time: 2 sec  
- Maximum Green Time: 3 sec  
- Lane with highest priority → GREEN  
- Other lanes → RED  
- Emergency vehicles → Instant GREEN 🚑  

---

## 🎯 Output

- Real-time video windows for each camera  
- Vehicle count per lane  
- Speed estimation (km/h)  
- Signal status (GREEN / RED)  
- Wait time display  

---

## 📊 Future Improvements

- 🔥 Hardware integration (Arduino / Raspberry Pi)
- 📡 Cloud-based monitoring dashboard
- 🧠 Advanced tracking (DeepSORT / ByteTrack)
- 📍 GPS-based emergency detection
- 🏙️ Smart city integration

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
Not intended for direct deployment in real-world traffic systems without testing and validation.

---

## 🙏 Acknowledgements

This project uses:

- Ultralytics YOLOv5  
  https://github.com/ultralytics/yolov5  

Special thanks to the authors for their open-source contribution.

---

## 📜 License

This project follows the **AGPL-3.0 License** (same as YOLOv5).  
Make sure to comply with it if used commercially.

---

## 👨‍💻 Author

**Ritik Srivastva**

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
