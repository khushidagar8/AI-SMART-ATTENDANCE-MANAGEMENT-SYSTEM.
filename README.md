# AI Smart Attendance Management System

## About the Project
The AI Smart Attendance Management System is a Python-based desktop application that uses face recognition to record and manage student attendance. It includes an advanced dashboard for student management, attendance history, and analytics.

## Features
- Student registration using a webcam
- Face detection and recognition using OpenCV
- Advanced desktop dashboard
- Student search and management
- Daily attendance recording
- Attendance history with date filtering
- Export attendance records to CSV
- Monthly attendance analytics
- Simple graphical user interface using Tkinter

## Technologies Used
- Python
- OpenCV
- NumPy
- Tkinter
- Matplotlib
- CSV and JSON

## Project Files
- `smart_attendance.py` — Basic attendance system
- `smart_attendance_advanced.py` — Advanced desktop application
- `requirements.txt` — Required Python libraries

## Installation

### 1. Install Python
Install Python on your computer if it is not already installed.

### 2. Install Required Libraries
Open Command Prompt and run:

```bash
py -m pip install -r requirements.txt
```

### 3. Run the Basic Version

```bash
py smart_attendance.py
```

### 4. Run the Advanced Version

```bash
py smart_attendance_advanced.py
```

## How to Use
1. Open the application.
2. Register a student with a student ID and name.
3. Capture face samples using the webcam.
4. Train the face recognition model.
5. Start attendance recognition.
6. View attendance records in the dashboard.
7. Export attendance data when required.

## Learning Outcomes
- Python programming and GUI development
- Face recognition using OpenCV
- Data handling using CSV and JSON
- Debugging and problem-solving
- Building a practical desktop application

## Privacy and Limitations
This project is an educational prototype. Face recognition may make mistakes, and the system does not include reliable liveness detection. Attendance should be verified manually. Use face data only with informed permission and keep personal information private.

## Future Improvements
- Improve recognition accuracy
- Add secure login and role-based access
- Generate attendance percentage reports
- Add PDF report generation
- Improve data security

## Author
B.Tech Artificial Intelligence Student

## Project Purpose
Developed as a learning project to practise Python, computer vision, and desktop application development.
