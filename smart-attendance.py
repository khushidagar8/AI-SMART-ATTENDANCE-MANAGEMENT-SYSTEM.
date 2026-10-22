
import os
import csv
import json
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from datetime import datetime, date
from collections import Counter

import cv2
import numpy as np

# Optional charts: install matplotlib to enable the chart tab.
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

# Keep this file in the SAME folder as the existing project.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "student_faces")
STUDENTS_FILE = os.path.join(BASE_DIR, "students.json")
MODEL_FILE = os.path.join(BASE_DIR, "face_model.yml")
LABELS_FILE = os.path.join(BASE_DIR, "face_labels.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

os.makedirs(DATA_DIR, exist_ok=True)

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

BG = "#eef3f8"
NAVY = "#17365d"
BLUE = "#2463a6"
WHITE = "#ffffff"
GREEN = "#238636"
RED = "#c0392b"


def load_students():
    if not os.path.exists(STUDENTS_FILE):
        return {}
    try:
        with open(STUDENTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_students(data):
    with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    try:
        with open(ATTENDANCE_FILE, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["Student ID", "Name", "Date", "Time"]
            )


def mark_attendance(student_id):
    students = load_students()
    if student_id not in students:
        return "Unknown student"

    today = date.today().isoformat()
    records = load_attendance()

    for row in records:
        if (row.get("Student ID") == student_id
                and row.get("Date") == today):
            return "Already Marked"

    ensure_attendance_file()
    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            student_id,
            students[student_id],
            today,
            datetime.now().strftime("%H:%M:%S")
        ])
    return "Attendance Marked"


def capture_student(student_id, name):
    folder = os.path.join(DATA_DIR, student_id)
    os.makedirs(folder, exist_ok=True)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        messagebox.showerror("Camera Error", "Webcam open nahi hua.")
        return False

    count = 0
    messagebox.showinfo(
        "Face Registration",
        "Achhi lighting mein camera ki taraf dekho.\n"
        "50 samples capture honge. Jaldi band karne ke liye Q dabao."
    )

    try:
        while count < 50:
            ok, frame = camera.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = CASCADE.detectMultiScale(
                gray, 1.2, 5, minSize=(100, 100)
            )

            for x, y, w, h in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Samples: {count}/50", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.imwrite(
                    os.path.join(folder, f"{count}.jpg"), face
                )
                count += 1
                break

            cv2.imshow("Register Face - Q to stop", frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if count < 10:
        messagebox.showwarning(
            "Registration Incomplete",
            f"Sirf {count} samples mile. Kam se kam 10 chahiye."
        )
        return False

    students = load_students()
    students[student_id] = name
    save_students(students)

    messagebox.showinfo(
        "Registration Complete",
        f"{name} registered. Samples: {count}\n"
        "Ab Train Face Model par click karo."
    )
    return True


def train_face_model():
    students = load_students()
    faces_list, labels_list = [], []
    label_map = {}
    label_num = 0

    for student_id in students:
        folder = os.path.join(DATA_DIR, student_id)
        if not os.path.isdir(folder):
            continue

        samples = []
        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(folder, filename), 0)
            if img is not None:
                samples.append(cv2.resize(img, (200, 200)))

        if len(samples) >= 10:
            label_map[str(label_num)] = student_id
            faces_list.extend(samples)
            labels_list.extend([label_num] * len(samples))
            label_num += 1

    if not faces_list:
        messagebox.showerror(
            "Training Error",
            "Kisi student ke 10 valid face images nahi mile."
        )
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(
        faces_list, np.array(labels_list, dtype=np.int32)
    )
    recognizer.save(MODEL_FILE)

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)

    messagebox.showinfo(
        "Training Complete",
        f"Model trained with {len(label_map)} student(s)."
    )
    return True


def run_face_attendance():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        messagebox.showwarning(
            "Model Missing", "Pehle Train Face Model run karo."
        )
        return

    with open(LABELS_FILE, encoding="utf-8") as f:
        label_map = json.load(f)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        messagebox.showerror("Camera Error", "Webcam open nahi hua.")
        return

    students = load_students()
    session_marked = set()

    messagebox.showinfo(
        "Attendance",
        "Camera start hoga. Recognized face verify karo.\n"
        "Band karne ke liye camera window mein Q dabao."
    )

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = CASCADE.detectMultiScale(
                gray, 1.2, 5, minSize=(100, 100)
            )

            for x, y, w, h in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                label, confidence = recognizer.predict(face)
                student_id = label_map.get(str(label))
                color = (0, 0, 255)
                display = "Unknown"

                # Tune this threshold only after testing with
                # different lighting and face angles.
                if student_id in students and confidence < 65:
                    display = students[student_id]
                    color = (0, 200, 0)

                    if student_id not in session_marked:
                        result = mark_attendance(student_id)
                        session_marked.add(student_id)
                        print(display, result)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    frame, f"{display} ({confidence:.1f})",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2
                )

            cv2.putText(
                frame, "Q = Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.imshow("AI Smart Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Smart Attendance | Advanced")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)

        self.search_var = tk.StringVar()
        self.date_var = tk.StringVar(value=date.today().isoformat())

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "Treeview", rowheight=28, font=("Arial", 10),
            background=WHITE, fieldbackground=WHITE
        )
        self.style.configure(
            "Treeview.Heading", font=("Arial", 10, "bold")
        )
        self.style.configure(
            "TNotebook.Tab", padding=(15, 8), font=("Arial", 10, "bold")
        )

        header = tk.Frame(root, bg=NAVY, height=80)
        header.pack(fill="x")
        tk.Label(
            header, text="AI SMART ATTENDANCE",
            font=("Arial", 21, "bold"), fg=WHITE, bg=NAVY
        ).pack(side="left", padx=25, pady=20)
        tk.Label(
            header, text="ADVANCED DESKTOP",
            font=("Arial", 10, "bold"), fg="#c9def5", bg=NAVY
        ).pack(side="right", padx=25)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=16)

        self.dashboard_tab = ttk.Frame(self.notebook)
        self.students_tab = ttk.Frame(self.notebook)
        self.attendance_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.students_tab, text="Students")
        self.notebook.add(self.attendance_tab, text="Attendance")
        self.notebook.add(self.analytics_tab, text="Analytics")

        self.build_dashboard()
        self.build_students()
        self.build_attendance()
        self.build_analytics()
        self.refresh_all()

    def make_tree(self, parent, columns, headings):
        tree = ttk.Treeview(
            parent, columns=columns, show="headings", height=15
        )
        for col, heading in zip(columns, headings):
            tree.heading(col, text=heading)
            tree.column(col, width=150, anchor="center")
        tree.pack(fill="both", expand=True, padx=12, pady=10)
        return tree

    def build_dashboard(self):
        tab = self.dashboard_tab
        tk.Label(
            tab, text="Overview", font=("Arial", 19, "bold")
        ).pack(anchor="w", padx=18, pady=(18, 6))

        self.stats_frame = tk.Frame(tab, bg=BG)
        self.stats_frame.pack(fill="x", padx=10, pady=10)

        self.stat_labels = {}
        for key, title in [
            ("students", "TOTAL STUDENTS"),
            ("present", "PRESENT TODAY"),
            ("absent", "NOT MARKED TODAY"),
            ("rate", "ATTENDANCE RATE")
        ]:
            card = tk.Frame(
                self.stats_frame, bg=WHITE,
                highlightbackground="#d8e1ec", highlightthickness=1
            )
            card.pack(side="left", fill="both", expand=True, padx=6, pady=5)
            tk.Label(
                card, text=title, bg=WHITE, fg="#526579",
                font=("Arial", 9, "bold")
            ).pack(pady=(16, 4))
            value = tk.Label(
                card, text="0", bg=WHITE, fg=BLUE,
                font=("Arial", 23, "bold")
            )
            value.pack(pady=(0, 16))
            self.stat_labels[key] = value

        actions = tk.LabelFrame(
            tab, text="Quick Actions", font=("Arial", 11, "bold")
        )
        actions.pack(fill="x", padx=18, pady=15)

        for text, command in [
            ("Register Student", self.add_student),
            ("Train Face Model", self.train),
            ("Start Attendance", self.start_camera),
            ("Refresh Dashboard", self.refresh_all)
        ]:
            tk.Button(
                actions, text=text, command=command,
                bg=BLUE, fg=WHITE, activebackground=NAVY,
                activeforeground=WHITE, relief="flat",
                padx=12, pady=10, cursor="hand2"
            ).pack(side="left", padx=8, pady=12)

        tk.Label(
            tab,
            text="Reminder: verify attendance records; face recognition can make mistakes.",
            fg=RED, font=("Arial", 10)
        ).pack(anchor="w", padx=20, pady=5)

    def build_students(self):
        tab = self.students_tab
        top = tk.Frame(tab)
        top.pack(fill="x", padx=12, pady=12)

        tk.Label(
            top, text="Search ID or name:",
            font=("Arial", 10, "bold")
        ).pack(side="left")
        entry = ttk.Entry(top, textvariable=self.search_var, width=30)
        entry.pack(side="left", padx=8)
        entry.bind("<KeyRelease>", lambda event: self.refresh_students())

        tk.Button(
            top, text="Add Student", command=self.add_student,
            bg=GREEN, fg=WHITE
        ).pack(side="right", padx=5)

        self.student_tree = self.make_tree(
            tab, ("id", "name", "samples"),
            ("Student ID", "Student Name", "Saved Face Images")
        )

        buttons = tk.Frame(tab)
        buttons.pack(fill="x", padx=12, pady=5)
        tk.Button(
            buttons, text="Update Name", command=self.update_name
        ).pack(side="left", padx=5)
        tk.Button(
            buttons, text="Delete Student", command=self.delete_student,
            fg=RED
        ).pack(side="left", padx=5)
        tk.Label(
            buttons, text="Deletion also removes this student's saved face images.",
            fg=RED
        ).pack(side="left", padx=8)

    def build_attendance(self):
        tab = self.attendance_tab
        top = tk.Frame(tab)
        top.pack(fill="x", padx=12, pady=12)

        tk.Label(top, text="Date (YYYY-MM-DD):").pack(side="left")
        ttk.Entry(
            top, textvariable=self.date_var, width=14
        ).pack(side="left", padx=6)

        tk.Button(
            top, text="Filter", command=self.refresh_attendance
        ).pack(side="left", padx=4)
        tk.Button(
            top, text="Show All", command=self.show_all_attendance
        ).pack(side="left", padx=4)
        tk.Button(
            top, text="Export CSV", command=self.export_csv,
            bg=GREEN, fg=WHITE
        ).pack(side="right", padx=4)

        self.attendance_tree = self.make_tree(
            tab, ("id", "name", "date", "time"),
            ("Student ID", "Student Name", "Date", "Time")
        )

    def build_analytics(self):
        tab = self.analytics_tab
        tk.Label(
            tab, text="Attendance Analytics",
            font=("Arial", 18, "bold")
        ).pack(anchor="w", padx=15, pady=15)

        if not CHARTS_AVAILABLE:
            tk.Label(
                tab,
                text="Install matplotlib to view charts:\n"
                     "py -m pip install matplotlib",
                font=("Arial", 12)
            ).pack(pady=30)
            return

        self.chart_area = tk.Frame(tab)
        self.chart_area.pack(fill="both", expand=True)

        tk.Button(
            tab, text="Refresh Chart", command=self.refresh_chart
        ).pack(pady=8)

    def refresh_dashboard(self):
        students = load_students()
        records = load_attendance()
        today = date.today().isoformat()
        present_ids = {
            row.get("Student ID") for row in records
            if row.get("Date") == today
        }
        total = len(students)
        present = len(present_ids.intersection(students.keys()))
        absent = max(0, total - present)
        rate = f"{present / total * 100:.1f}%" if total else "0%"

        for key, value in [
            ("students", total), ("present", present),
            ("absent", absent), ("rate", rate)
        ]:
            self.stat_labels[key].config(text=str(value))

    def refresh_students(self):
        for item in self.student_tree.get_children():
            self.student_tree.delete(item)

        query = self.search_var.get().strip().lower()
        for student_id, name in sorted(load_students().items()):
            if query and query not in student_id.lower() and query not in name.lower():
                continue
            folder = os.path.join(DATA_DIR, student_id)
            samples = 0
            if os.path.isdir(folder):
                samples = sum(
                    f.lower().endswith((".jpg", ".jpeg", ".png"))
                    for f in os.listdir(folder)
                )
            self.student_tree.insert(
                "", "end", values=(student_id, name, samples)
            )

    def refresh_attendance(self):
        try:
            selected_date = date.fromisoformat(self.date_var.get().strip()).isoformat()
        except ValueError:
            messagebox.showerror(
                "Invalid Date", "Date format YYYY-MM-DD use karo."
            )
            return
        self.populate_attendance(selected_date)

    def populate_attendance(self, selected_date=None):
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)

        records = load_attendance()
        if selected_date:
            records = [
                r for r in records if r.get("Date") == selected_date
            ]

        for row in records:
            self.attendance_tree.insert(
                "", "end",
                values=(
                    row.get("Student ID", ""),
                    row.get("Name", ""),
                    row.get("Date", ""),
                    row.get("Time", "")
                )
            )

    def show_all_attendance(self):
        self.populate_attendance()

    def refresh_chart(self):
        if not CHARTS_AVAILABLE:
            return

        for widget in self.chart_area.winfo_children():
            widget.destroy()

        records = load_attendance()
        month_counts = Counter()
        for row in records:
            try:
                month = date.fromisoformat(row.get("Date", "")).strftime("%Y-%m")
                month_counts[month] += 1
            except ValueError:
                continue

        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        months = sorted(month_counts.keys())[-12:]

        if months:
            ax.bar(months, [month_counts[m] for m in months])
            ax.set_title("Attendance Entries by Month")
            ax.set_ylabel("Attendance entries")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
        else:
            ax.text(
                0.5, 0.5, "No attendance data yet",
                ha="center", va="center"
            )
            ax.set_axis_off()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def add_student(self):
        student_id = simpledialog.askstring(
            "Register Student", "Student ID (letters and numbers only):",
            parent=self.root
        )
        if not student_id:
            return
        student_id = student_id.strip()

        if not student_id.isalnum():
            messagebox.showerror(
                "Invalid ID", "Student ID mein sirf letters aur numbers use karo."
            )
            return

        students = load_students()
        if student_id in students:
            messagebox.showerror(
                "Duplicate ID", "Ye ID already registered hai."
            )
            return

        name = simpledialog.askstring(
            "Register Student", "Student Name:", parent=self.root
        )
        if not name or not name.strip():
            return

        if capture_student(student_id, name.strip()):
            self.refresh_all()

    def update_name(self):
        selected = self.student_tree.selection()
        if not selected:
            messagebox.showinfo("Select Student", "Pehle student select karo.")
            return

        student_id = self.student_tree.item(selected[0], "values")[0]
        students = load_students()
        new_name = simpledialog.askstring(
            "Update Name", "New name:", initialvalue=students[student_id],
            parent=self.root
        )
        if new_name and new_name.strip():
            students[student_id] = new_name.strip()
            save_students(students)
            self.refresh_all()

    def delete_student(self):
        selected = self.student_tree.selection()
        if not selected:
            messagebox.showinfo("Select Student", "Pehle student select karo.")
            return

        student_id = self.student_tree.item(selected[0], "values")[0]
        if not messagebox.askyesno(
            "Confirm Delete",
            f"Student {student_id} aur uski face images delete karni hain?"
        ):
            return

        students = load_students()
        students.pop(student_id, None)
        save_students(students)

        folder = os.path.join(DATA_DIR, student_id)
        if os.path.isdir(folder):
            import shutil
            shutil.rmtree(folder)

        # Retrain after removing a student so the model does not
        # continue recognizing the deleted student.
        if os.path.exists(MODEL_FILE):
            try:
                os.remove(MODEL_FILE)
            except OSError:
                pass
        if os.path.exists(LABELS_FILE):
            try:
                os.remove(LABELS_FILE)
            except OSError:
                pass

        messagebox.showinfo(
            "Student Deleted",
            "Student removed. Attendance history is retained for reporting. "
            "Train the face model again before starting attendance."
        )
        self.refresh_all()

    def train(self):
        if train_face_model():
            self.refresh_all()

    def start_camera(self):
        run_face_attendance()
        self.refresh_all()

    def export_csv(self):
        records = load_attendance()
        if not records:
            messagebox.showinfo("Export", "Export karne ke liye records nahi hain.")
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Attendance",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["Student ID", "Name", "Date", "Time"]
                )
                writer.writeheader()
                writer.writerows(records)
            messagebox.showinfo("Export Complete", f"Report saved:\n{path}")
        except OSError as exc:
            messagebox.showerror("Export Error", str(exc))

    def refresh_all(self):
        self.refresh_dashboard()
        self.refresh_students()
        self.refresh_attendance()
        self.refresh_chart()


if __name__ == "__main__":
    ensure_attendance_file()
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
