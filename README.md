WasteGuard — Waste & Baggage Detection System

🧠 AI-powered system to detect waste disposal behavior from images using YOLOv8

📌 Overview

WasteGuard is a Flask-based web application that analyzes uploaded images to detect:

🚶 Persons
🗑️ Waste items
🚗 Vehicle plates

It returns an annotated image with bounding boxes and a clear incident status report.

✨ Features
📤 Upload JPG/PNG images via a web interface
🤖 YOLOv8-powered object detection
🎯 Detects:
       Person
         Waste
           Vehicle Plate
🎨 Color-coded bounding boxes with confidence scores
⚠️ Smart incident detection:
            Incident Detected
            No Incident

🛠️ Tech Stack
Backend: Python + Flask
AI Model: YOLOv8 (Ultralytics)
Image Processing: OpenCV
Frontend: HTML, CSS

📁 Project Structure
WasteGuard/
│
├── app.py               # Flask routes and app entry point
├── detector.py          # YOLOv8 inference logic
├── annotator.py         # Bounding box drawing using OpenCV
├── file_handler.py      # File upload handling
│
├── templates/
│   └── index.html       # Frontend UI
│
├── static/
│   ├── uploads/         # Uploaded images
│   └── outputs/         # Annotated images
│
├── tests/               # Unit & property-based tests
│
└── yolov8n.pt           # YOLOv8 model weights

⚙️ Setup & Run
1️⃣ Clone the repository
git clone https://github.com/your-username/wasteguard.git
cd wasteguard
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the application
python app.py
4️⃣ Open in browser
http://localhost:5000

🧪 Example Workflow
Upload an image
Model detects objects
Annotated image is generated
Incident status is displayed
