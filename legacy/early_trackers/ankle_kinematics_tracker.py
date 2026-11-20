
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import csv
import time
import math
import os
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

# Marker colours in HSV.
# Default arrangement:
# Knee  = RED
# Ankle = GREEN
# Foot  = BLUE

HSV_RANGES = {
    "knee": [
        ((0, 100, 80), (10, 255, 255)),
        ((170, 100, 80), (180, 255, 255))
    ],

    "ankle": [
        ((35, 70, 60), (90, 255, 255))
    ],

    "foot": [
        ((95, 80, 60), (135, 255, 255))
    ]
}

# Marker detection settings
MIN_MARKER_AREA = 12
MAX_MARKER_AREA = 5000
MIN_CIRCULARITY = 0.30

# Maximum distance from MediaPipe landmark
# within which a coloured marker is accepted.
MAX_MARKER_DISTANCE = 0.15

# Camera image settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# ANKLE ANGLE CALCULATION
# ============================================================

def calculate_ankle_angle(knee, ankle, foot):

    """
    Calculates signed angle between:
        Shank vector: Ankle -> Knee
        Foot vector:  Ankle -> Foot

    Image y-coordinate increases downward.
    Convert to Cartesian coordinates with y upward.

    Returns angle in degrees.
    """

    v1 = np.array([
        knee[0] - ankle[0],
        -(knee[1] - ankle[1])
    ], dtype=float)

    v2 = np.array([
        foot[0] - ankle[0],
        -(foot[1] - ankle[1])
    ], dtype=float)

    cross = v1[0] * v2[1] - v1[1] * v2[0]

    dot = np.dot(v1, v2)

    angle = math.degrees(math.atan2(cross, dot))

    return angle


def wrap_angle(angle):
    """Wrap angle into [-180, 180] degrees."""
    return (angle + 180) % 360 - 180


# ============================================================
# COLOURED MARKER DETECTION
# ============================================================

def detect_marker(hsv, marker_name, expected_point):

    """
    Detect a coloured marker near the corresponding
    MediaPipe anatomical landmark.

    Returns:
        (x, y) marker centre, or None.
    """

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in HSV_RANGES[marker_name]:

        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        current_mask = cv2.inRange(
            hsv, lower, upper
        )

        mask = cv2.bitwise_or(mask, current_mask)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if expected_point is None:
        return None

    best_marker = None
    best_distance = float("inf")

    max_distance = MAX_MARKER_DISTANCE * math.sqrt(
        hsv.shape[0]**2 + hsv.shape[1]**2
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if not MIN_MARKER_AREA <= area <= MAX_MARKER_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * math.pi * area / (perimeter * perimeter)
        )

        if circularity < MIN_CIRCULARITY:
            continue

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        distance = math.dist(
            (cx, cy), expected_point
        )

        if distance < best_distance and distance < max_distance:

            best_distance = distance
            best_marker = (cx, cy)

    return best_marker


# ============================================================
# MAIN APPLICATION
# ============================================================

class AnkleTracker:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Lower Limb Ankle Kinematics Tracker"
        )

        self.root.geometry("1250x850")

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Unable to open webcam."
            )
            self.root.destroy()
            return

        self.recording = False

        self.csv_file = None
        self.csv_writer = None

        self.start_time = None
        self.frame_number = 0

        self.neutral_angle = None
        self.current_raw_angle = None
        self.current_relative_angle = None

        self.last_markers = None

        self.selected_side = tk.StringVar(value="Left")

        self.setup_gui()

        self.update_frame()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

    # --------------------------------------------------------
    # GUI
    # --------------------------------------------------------

    def setup_gui(self):

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(main_frame)
        self.video_label.pack(pady=5)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10)

        ttk.Label(
            control_frame,
            text="Tracking Side:"
        ).grid(row=0, column=0, padx=5)

        side_menu = ttk.Combobox(
            control_frame,
            textvariable=self.selected_side,
            values=["Left", "Right"],
            state="readonly",
            width=10
        )

        side_menu.grid(row=0, column=1, padx=5)

        self.start_button = ttk.Button(
            control_frame,
            text="START RECORDING",
            command=self.start_recording
        )

        self.start_button.grid(
            row=0, column=2, padx=10
        )

        self.stop_button = ttk.Button(
            control_frame,
            text="STOP RECORDING",
            command=self.stop_recording,
            state=tk.DISABLED
        )

        self.stop_button.grid(
            row=0, column=3, padx=10
        )

        self.neutral_button = ttk.Button(
            control_frame,
            text="SET NEUTRAL",
            command=self.set_neutral
        )

        self.neutral_button.grid(
            row=0, column=4, padx=10
        )

        self.angle_label = ttk.Label(
            main_frame,
            text="Ankle Angle: --",
            font=("Arial", 22, "bold")
        )

        self.angle_label.pack(pady=10)

        self.status_label = ttk.Label(
            main_frame,
            text="Status: Waiting for subject",
            font=("Arial", 12)
        )

        self.status_label.pack(pady=5)

        self.neutral_label = ttk.Label(
            main_frame,
            text="Neutral angle: Not calibrated",
            font=("Arial", 11)
        )

        self.neutral_label.pack(pady=5)

        instructions = (
            "Knee: RED | Ankle: GREEN | Foot: BLUE\n"
            "Place the camera perpendicular to the sagittal plane."
        )

        ttk.Label(
            main_frame,
            text=instructions,
            font=("Arial", 10)
        ).pack(pady=10)

    # --------------------------------------------------------
    # MEDIAPIPE LANDMARK EXTRACTION
    # --------------------------------------------------------

    def get_pose_points(self, frame):

        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        if self.selected_side.get() == "Left":

            indices = {
                "knee": mp_pose.PoseLandmark.LEFT_KNEE.value,
                "ankle": mp_pose.PoseLandmark.LEFT_ANKLE.value,
                "foot": mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
            }

        else:

            indices = {
                "knee": mp_pose.PoseLandmark.RIGHT_KNEE.value,
                "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                "foot": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
            }

        h, w = frame.shape[:2]

        points = {}

        for name, idx in indices.items():

            lm = landmarks[idx]

            if lm.visibility < 0.5:
                return None

            points[name] = (
                int(lm.x * w),
                int(lm.y * h)
            )

        return points

    # --------------------------------------------------------
    # START RECORDING
    # --------------------------------------------------------

    def start_recording(self):

        if self.recording:
            return

        if self.last_markers is None:
            messagebox.showwarning(
                "Markers Not Detected",
                "Ensure all three markers are visible before recording."
            )
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=(
                "ankle_kinematics_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".csv"
            )
        )

        if not filename:
            return

        try:
            self.csv_file = open(
                filename,
                "w",
                newline=""
            )

            self.csv_writer = csv.writer(self.csv_file)

            self.csv_writer.writerow([
                "Timestamp",
                "Time_s",
                "Frame",
                "Side",

                "Knee_X_px",
                "Knee_Y_px",

                "Ankle_X_px",
                "Ankle_Y_px",

                "Foot_X_px",
                "Foot_Y_px",

                "Raw_Ankle_Angle_deg",
                "Neutral_Angle_deg",
                "Ankle_Angle_deg",

                "Marker_Tracking_Valid"
            ])

        except OSError as e:
            messagebox.showerror(
                "File Error", str(e)
            )
            return

        self.recording = True
        self.frame_number = 0
        self.start_time = time.perf_counter()

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.status_label.config(
            text="Status: RECORDING"
        )

    # --------------------------------------------------------
    # STOP RECORDING
    # --------------------------------------------------------

    def stop_recording(self):

        if not self.recording:
            return

        self.recording = False

        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()

        self.csv_file = None
        self.csv_writer = None

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        self.status_label.config(
            text="Status: Recording stopped. CSV saved."
        )

    # --------------------------------------------------------
    # NEUTRAL CALIBRATION
    # --------------------------------------------------------

    def set_neutral(self):

        if self.current_raw_angle is None:
            messagebox.showwarning(
                "Calibration",
                "All three markers must be visible."
            )
            return

        self.neutral_angle = self.current_raw_angle

        self.neutral_label.config(
            text=(
                f"Neutral angle: "
                f"{self.neutral_angle:.2f} degrees"
            )
        )

    # --------------------------------------------------------
    # CSV DATA STORAGE
    # --------------------------------------------------------

    def save_data(self, markers, raw_angle, relative_angle):

        if not self.recording:
            return

        elapsed = time.perf_counter() - self.start_time

        timestamp = datetime.now().isoformat(
            timespec="milliseconds"
        )

        knee = markers["knee"]
        ankle = markers["ankle"]
        foot = markers["foot"]

        self.csv_writer.writerow([

            timestamp,
            f"{elapsed:.6f}",
            self.frame_number,
            self.selected_side.get(),

            knee[0], knee[1],
            ankle[0], ankle[1],
            foot[0], foot[1],

            f"{raw_angle:.6f}",
            (
                f"{self.neutral_angle:.6f}"
                if self.neutral_angle is not None
                else ""
            ),

            (
                f"{relative_angle:.6f}"
                if relative_angle is not None
                else ""
            ),

            1
        ])

        # Flush periodically to reduce data loss.
        if self.frame_number % 30 == 0:
            self.csv_file.flush()

    # --------------------------------------------------------
    # FRAME PROCESSING
    # --------------------------------------------------------

    def update_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            self.root.after(20, self.update_frame)
            return


        h, w = frame.shape[:2]

        pose_points = self.get_pose_points(frame)

        markers = None

        if pose_points is not None:

            hsv = cv2.cvtColor(
                frame, cv2.COLOR_BGR2HSV
            )

            markers = {}

            for name in ["knee", "ankle", "foot"]:

                marker = detect_marker(
                    hsv,
                    name,
                    pose_points[name]
                )

                markers[name] = marker

            if any(v is None for v in markers.values()):
                markers = None

        if markers is not None:

            self.last_markers = markers

            raw_angle = calculate_ankle_angle(
                markers["knee"],
                markers["ankle"],
                markers["foot"]
            )

            self.current_raw_angle = raw_angle

            if self.neutral_angle is not None:

                relative_angle = wrap_angle(
                    raw_angle - self.neutral_angle
                )

            else:
                relative_angle = None

            self.current_relative_angle = relative_angle

            # Draw marker points.
            marker_colours = {
                "knee": (0, 0, 255),
                "ankle": (0, 255, 0),
                "foot": (255, 0, 0)
            }

            for name, point in markers.items():

                cv2.circle(
                    frame,
                    point,
                    9,
                    marker_colours[name],
                    -1
                )

                cv2.putText(
                    frame,
                    name.upper(),
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    marker_colours[name],
                    2
                )

            # Draw shank and foot segments.
            cv2.line(
                frame,
                markers["knee"],
                markers["ankle"],
                (0, 255, 255),
                3
            )

            cv2.line(
                frame,
                markers["ankle"],
                markers["foot"],
                (255, 255, 0),
                3
            )

            angle_text = (
                f"Raw angle: {raw_angle:.2f} deg"
            )

            if relative_angle is not None:

                angle_text += (
                    f" | Relative: {relative_angle:.2f} deg"
                )

            cv2.putText(
                frame,
                angle_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            self.angle_label.config(
                text=(
                    f"Ankle Angle: "
                    f"{relative_angle:.2f} deg"
                    if relative_angle is not None
                    else f"Raw Ankle Angle: {raw_angle:.2f} deg"
                )
            )

            self.status_label.config(
                text=(
                    "Status: RECORDING"
                    if self.recording
                    else "Status: Tracking"
                )
            )

            if self.recording:

                self.save_data(
                    markers,
                    raw_angle,
                    relative_angle
                )

                self.frame_number += 1

        else:

            self.last_markers = None

            self.angle_label.config(
                text="Ankle Angle: Tracking Lost"
            )

            self.status_label.config(
                text="Status: One or more markers not detected"
            )

        # Show recording indicator.
        if self.recording:

            cv2.circle(
                frame, (w - 35, 35), 10, (0, 0, 255), -1
            )

            cv2.putText(
                frame,
                "REC",
                (w - 90, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Convert frame for Tkinter.
        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        # Resize for display while maintaining aspect ratio.
        image.thumbnail((1150, 650))

        imgtk = ImageTk.PhotoImage(image=image)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(20, self.update_frame)

    # --------------------------------------------------------
    # CLOSE APPLICATION
    # --------------------------------------------------------

    def close_application(self):

        if self.recording:
            self.stop_recording()

        if self.cap.isOpened():
            self.cap.release()

        pose.close()

        self.root.destroy()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = AnkleTracker(root)

    root.mainloop()
