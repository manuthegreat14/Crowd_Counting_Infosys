People Counting System

Real-time people counting system using computer vision and deep learning for public spaces.
Features

    ✅ Real-time people detection and tracking using YOLOv8
    ✅ Multi-zone management (draw, save, edit zones)
    ✅ Live video feed from webcam/IP camera
    ✅ Interactive web dashboard with statistics
    ✅ Person tracking with unique IDs
    ✅ Zone-wise people counting
    ✅ Export CSV/PDF of hourly/daily counts
    
System Architecture
Camera Feed → YOLO Detection → Object Tracking → Zone Processing → Dashboard

Modules

1. Video Input & Zone Management
Capture video from webcam or IP camera
Draw and manage multiple zones (ROI)
Save and reload zone configurations

2. People Detection & Tracking
Detect humans using YOLOv8
Track individuals using unique IDs
Maintain consistent tracking across frames

3. Zone-Based Counting
Assign people to zones using bounding box logic
Count individuals per zone in real-time

4. Dashboard & Visualization
Display total count and zone-wise data
Real-time charts (bar & line graphs)
Live statistics update
Export CSV/PDF of hourly/daily counts

Technologies Used
Backend: Python, Flask
Computer Vision: OpenCV
Deep Learning: YOLOv8 (Ultralytics)
Frontend: HTML, CSS, JavaScript
```bash
Project Structure
├── app.py # Main Flask application (routes + API) 
├── tracking.py # YOLOv8 detection & tracking logic 
├── zone_editor.py # Zone drawing and management 
├── templates/ 
│ └── index.html # Dashboard UI (video + stats + charts)
  └── login.html # Login UI(Default username:admin ,password:1234)
└── zones.json # Saved zones (auto-generated) 
```
Installation & Setup
1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   (Linux/Mac)
venv\Scripts\activate      (Windows)

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
python app.py

6. Open in Browser
http://localhost:5000
