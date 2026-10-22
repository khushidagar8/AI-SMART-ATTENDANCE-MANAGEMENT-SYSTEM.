
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import json
import csv
from datetime import datetime

# ---------- PROJECT FILES ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "student_faces")
STUDENTS_FILE = os.path.join(BASE_DIR, "students.json")
MODEL_FILE = os.path.join(BASE_DIR, "face_model.yml")
LABELS_FILE = os.path.join(BASE_DIR, "face_labels.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------- FACE DETECTOR ----------
CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)
detector = cv2.CascadeClassifier(CASCADE_PATH)


def load_students():
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_students(students):
    with open(STUDENTS_FILE, "w", encoding="utf-8") as file:
        json.dump(students, file, indent=4)


# ---------- REGISTER STUDENT ----------
def register_student():
    student_id = simpledialog.askstring(
        "Registration", "Enter Student ID:"
    )
    if not student_id:
        return

    student_id = student_id.strip()

    if not student_id.isalnum():
        messagebox.showerror(
            "Invalid ID", "Use letters and numbers only."
        )
        return

    students = load_students()

    if student_id in students:
        messagebox.showerror(
            "Already Registered", "This Student ID already exists."
        )
        return

    name = simpledialog.askstring(
        "Registration", "Enter Student Name:"
    )
    if not name or not name.strip():
        return

    folder = os.path.join(DATA_DIR, student_id)
    os.makedirs(folder, exist_ok=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        messagebox.showerror(
            "Camera Error", "Could not open webcam."
        )
        return

    count = 0
    messagebox.showinfo(
        "Face Capture",
        "Look at the camera in good lighting.\n"
        "Move your head slightly to capture different angles.\n"
        "Press Q in the camera window to finish."
    )

    while count < 50:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            cv2.rectangle(
                frame, (x, y), (x+w, y+h), (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Samples: {count}/50",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            if count < 50:
                cv2.imwrite(
                    os.path.join(folder, f"{count}.jpg"), face
                )
                count += 1
                break

        cv2.imshow("Register Student - Press Q to quit", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    if count < 10:
        messagebox.showwarning(
            "Not Enough Samples",
            f"Captured {count} images. Please register again "
            "and capture at least 10 face images."
        )
        return

    students[student_id] = name.strip()
    save_students(students)

    messagebox.showinfo(
        "Registration Complete",
        f"Student: {name.strip()}\n"
        f"ID: {student_id}\n"
        f"Images captured: {count}\n\n"
        "Now click Train Face Model."
    )


# ---------- TRAIN FACE MODEL ----------
def train_model():
    students = load_students()

    if not students:
        messagebox.showwarning(
            "No Students", "Register a student first."
        )
        return

    faces_list = []
    labels_list = []
    label_map = {}
    label_number = 0

    for student_id in students:
        folder = os.path.join(DATA_DIR, student_id)
        if not os.path.isdir(folder):
            continue

        student_faces = []

        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                image = cv2.resize(image, (200, 200))
                student_faces.append(image)

        if len(student_faces) >= 10:
            label_map[str(label_number)] = student_id

            for face in student_faces:
                faces_list.append(face)
                labels_list.append(label_number)

            label_number += 1

    if not faces_list:
        messagebox.showerror(
            "Training Error",
            "No student has at least 10 saved face images."
        )
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(
        faces_list, np.array(labels_list, dtype=np.int32)
    )
    recognizer.save(MODEL_FILE)

    with open(LABELS_FILE, "w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=4)

    messagebox.showinfo(
        "Training Complete",
        f"Model trained with {len(label_map)} student(s)!\n"
        "You can now start attendance."
    )


# ---------- MARK ATTENDANCE ----------
def mark_attendance(student_id):
    students = load_students()
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(
            ATTENDANCE_FILE, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Student ID", "Name", "Date", "Time"])

    with open(
        ATTENDANCE_FILE, "r", newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    for record in records:
        if (record["Student ID"] == student_id
                and record["Date"] == today):
            return "Already Marked"

    with open(
        ATTENDANCE_FILE, "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            student_id, students[student_id], today, now
        ])

    return "Attendance Marked"


def start_attendance():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        messagebox.showwarning(
            "Model Missing",
            "Please register students and train the model first."
        )
        return

    with open(LABELS_FILE, "r", encoding="utf-8") as file:
        label_map = json.load(file)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        messagebox.showerror(
            "Camera Error", "Could not open webcam."
        )
        return

    students = load_students()
    marked_this_session = set()

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(face)

            student_id = label_map.get(str(label))
            name = "Unknown"
            color = (0, 0, 255)

            # Lower LBPH confidence means a closer match.
            if student_id in students and confidence < 65:
                name = students[student_id]
                color = (0, 255, 0)

                if student_id not in marked_this_session:
                    result = mark_attendance(student_id)
                    marked_this_session.add(student_id)

                    print(
                        f"{name} ({student_id}): {result}"
                    )

            cv2.rectangle(
                frame, (x, y), (x+w, y+h), color, 2
            )
            cv2.putText(
                frame, f"{name} [{confidence:.1f}]",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )

        cv2.putText(
            frame, "Press Q to stop attendance",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        cv2.imshow("AI Smart Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


# ---------- VIEW ATTENDANCE ----------
def view_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        messagebox.showinfo(
            "Attendance", "No attendance records found yet."
        )
        return

    with open(ATTENDANCE_FILE, "r", encoding="utf-8") as file:
        content = file.read()

    window = tk.Toplevel(root)
    window.title("Attendance Records")
    window.geometry("650x350")

    text_box = tk.Text(window, wrap="none", font=("Consolas", 10))
    text_box.pack(fill="both", expand=True)
    text_box.insert("1.0", content)
    text_box.config(state="disabled")


# ---------- MAIN WINDOW ----------
root = tk.Tk()
root.title("AI Smart Attendance System")
root.geometry("520x470")
root.configure(bg="#eaf4ff")

title = tk.Label(
    root,
    text="AI SMART ATTENDANCE SYSTEM",
    font=("Arial", 17, "bold"),
    bg="#eaf4ff",
    fg="#123b63"
)
title.pack(pady=25)

subtitle = tk.Label(
    root,
    text="Face Recognition | Student Records",
    font=("Arial", 10),
    bg="#eaf4ff",
    fg="#444444"
)
subtitle.pack(pady=3)


def make_button(text, command):
    return tk.Button(
        root,
        text=text,
        command=command,
        font=("Arial", 12, "bold"),
        width=27,
        bg="#1769aa",
        fg="white",
        activebackground="#0d4778",
        activeforeground="white",
        relief="flat",
        pady=9,
        cursor="hand2"
    )


make_button("1. Register Student", register_student).pack(pady=7)
make_button("2. Train Face Model", train_model).pack(pady=7)
make_button("3. Start Attendance", start_attendance).pack(pady=7)
make_button("4. View Attendance", view_attendance).pack(pady=7)

tk.Label(
    root,
    text="Press Q to close the webcam window",
    font=("Arial", 9),
    bg="#eaf4ff",
    fg="#444444"
).pack(pady=16)

root.mainloop()
