# People_Entry_Exit-Counter_using_YOLO_and_OpenCV

This project is a real-time **people entry and exit counting system** built with **Python, YOLOv5, and OpenCV**.  
It detects people in a video feed, tracks their movement, and automatically counts how many people **enter** and **exit** a defined area by crossing virtual lines.

✅ Ideal for:
- 🏪 Retail store footfall analysis  
- 🏢 Building entrance monitoring  
- 🏙️ Smart city crowd analytics  
- 🎓 Computer vision learning projects

---

## ✨ Features

- 🚶‍♂️ Detects people in video streams using YOLOv5  
- 📊 Counts how many people **enter** and **exit**  
- 🔢 Displays real-time `IN` and `OUT` counts  
- 🧠 Assigns unique IDs for accurate tracking  
- 💾 Saves processed video with bounding boxes and counts  
- 🛠️ Fully customizable detection zones

---

## 🛠️ Tech Stack

- Python 3.x  
- [YOLOv5](https://github.com/ultralytics/yolov5) (Ultralytics)  
- OpenCV  
- NumPy, SciPy

📊 How It Works (Updated with Line Explanation)

The system works by detecting people and tracking their movement relative to two virtual lines drawn on the video frame:

🔴 Red Line (Entry Line):
When a person crosses this upper red line from top to bottom, the system counts them as IN (entered).

🔵 Blue Line (Exit Line):
When a person crosses this lower blue line from bottom to top, the system counts them as OUT (exited).

These two lines act like digital gates — once a tracked person’s center point crosses them, the counter updates in real time.
