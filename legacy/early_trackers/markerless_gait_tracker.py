"""
Markerless Lower-Limb Gait Tracker
----------------------------------
Windows webcam application using MediaPipe Pose + OpenCV.

Tracks one side of the body in a sagittal-plane camera view and estimates:
    * Hip angle: trunk-to-thigh orientation
    * Knee flexion angle
    * Ankle angle: shank-to-foot orientation

Features:
    * Live webcam preview with pose skeleton
    * Left/right side selection
    * Neutral-pose calibration
    * Subject ID and activity selection
    * Start / Stop recording
    * Unlimited sequential trials (limited by disk space)
    * Separate CSV for each trial:
          output_folder / Subject_ID / Activity / Trial_XX_timestamp.csv
    * Writes a row for every frame, including invalid tracking frames
    * Saves pixel coordinates, normalized landmarks, angles and tracking flags

Important:
    * This is 2D markerless pose estimation, not laboratory-grade 3D motion capture.
    * Place the camera perpendicular to the participant's sagittal plane.
    * Keep the selected side visible and avoid occlusion.
    * Angles are image-plane estimates. Validate against a reference system
      before using them as research-grade measurements.
"""

import csv
import math
import re
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

VISIBILITY_THRESHOLD = 0.50
SMOOTHING_ALPHA = 0.35       # Higher = follows movement faster
DISPLAY_WIDTH = 1050
DISPLAY_HEIGHT = 650
FRAME_INTERVAL_MS = 15

DEFAULT_OUTPUT_FOLDER = Path.home() / "Gait_Analysis_Recordings"

ACTIVITIES = [
    "Normal_Walking",
    "Fast_Walking",
    "Slow_Walking",
    "Running",
    "Standing",
    "Sit_to_Stand",
    "Stair_Ascent",
    "Stair_Descent",
    "Other",
]

# MediaPipe Pose landmark identifiers
POSE = mp.solutions.pose
LANDMARKS = {
    "shoulder": POSE.PoseLandmark.LEFT_SHOULDER.value,
    "hip": POSE.PoseLandmark.LEFT_HIP.value,
    "knee": POSE.PoseLandmark.LEFT_KNEE.value,
    "ankle": POSE.PoseLandmark.LEFT_ANKLE.value,
    "heel": POSE.PoseLandmark.LEFT_HEEL.value,
    "foot_index": POSE.PoseLandmark.LEFT_FOOT_INDEX.value,
}


# ---------------------------------------------------------------------
# GEOMETRY / ANGLE UTILITIES
# ---------------------------------------------------------------------

def wrap_degrees(angle):
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def vector_angle_deg(v1, v2):
    """Signed angle rotating v1 to v2, with image y already converted to up."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = float(np.dot(v1, v2))
    return math.degrees(math.atan2(cross, dot))


def to_cartesian_vector(point_a, point_b):
    """
    Vector from point_a to point_b in image coordinates, converted so +y is up.
    Points are (x_pixel, y_pixel).
    """
    return np.array(
        [point_b[0] - point_a[0], -(point_b[1] - point_a[1])],
        dtype=float
    )


def calculate_joint_angles(points):
    """
    Calculate image-plane joint angles.

    Hip:
        Signed angle from trunk vector (hip -> shoulder) to thigh vector
        (hip -> knee). Relative hip angle is calibrated to the neutral pose.

    Knee:
        Flexion angle = 180 - included angle between (knee -> hip) and
        (knee -> ankle). Approximately 0 degrees when the leg is straight.

    Ankle:
        Signed angle from shank vector (ankle -> knee) to foot vector
        (ankle -> foot_index). Relative ankle angle is calibrated to neutral.

    Returns raw angle values in degrees, or None if geometry is invalid.
    """
    shoulder = points["shoulder"]
    hip = points["hip"]
    knee = points["knee"]
    ankle = points["ankle"]
    foot = points["foot_index"]

    trunk = to_cartesian_vector(hip, shoulder)
    thigh_from_hip = to_cartesian_vector(hip, knee)

    hip_raw = vector_angle_deg(trunk, thigh_from_hip)

    thigh_from_knee = to_cartesian_vector(knee, hip)
    shank_from_knee = to_cartesian_vector(knee, ankle)
    included_knee = vector_angle_deg(thigh_from_knee, shank_from_knee)
    knee_flexion = None if included_knee is None else 180.0 - abs(included_knee)

    shank_from_ankle = to_cartesian_vector(ankle, knee)
    foot_from_ankle = to_cartesian_vector(ankle, foot)
    ankle_raw = vector_angle_deg(shank_from_ankle, foot_from_ankle)

    return {
        "hip_raw_deg": hip_raw,
        "knee_flexion_deg": knee_flexion,
        "ankle_raw_deg": ankle_raw,
    }


def safe_name(value, fallback="Unknown"):
    """Make a folder/file-safe name from user input."""
    value = str(value).strip()
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip(" ._")
    return value or fallback


# ---------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------

class MarkerlessGaitTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Markerless Lower-Limb Gait Tracker")
        self.root.geometry("1350x930")
        self.root.minsize(1050, 760)

        self.cap = None
        self.pose_model = None
        self.photo_image = None
        self.after_id = None
        self.closed = False

        self.recording = False
        self.csv_handle = None
        self.csv_writer = None
        self.recording_path = None
        self.record_start_perf = None
        self.record_frame_count = 0
        self.global_frame_count = 0

        self.neutral_angles = None
        self.current_points = None
        self.current_angles = None
        self.smoothed_points = None
        self.last_valid_points = None
        self.last_frame_time = time.perf_counter()
        self.current_fps = 0.0

        self.side_var = tk.StringVar(value="Left")
        self.subject_var = tk.StringVar(value="Subject_01")
        self.activity_var = tk.StringVar(value="Normal_Walking")
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_FOLDER))
        self.camera_var = tk.StringVar(value=str(CAMERA_INDEX))

        self._build_gui()
        self._initialize_pose()
        self._initialize_camera()

        self.root.protocol("WM_DELETE_WINDOW", self.close_application)
        if self.cap is not None and self.cap.isOpened():
            self.update_frame()
        else:
            self.status_var.set("Camera unavailable. Check camera index and permissions.")

    # -----------------------------------------------------------------
    # GUI
    # -----------------------------------------------------------------

    def _build_gui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        settings = ttk.LabelFrame(outer, text="Experiment setup", padding=8)
        settings.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(settings, text="Subject ID:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.subject_entry = ttk.Entry(settings, textvariable=self.subject_var, width=18)
        self.subject_entry.grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(settings, text="Activity:").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.activity_combo = ttk.Combobox(
            settings, textvariable=self.activity_var, values=ACTIVITIES, width=22
        )
        self.activity_combo.grid(row=0, column=3, padx=4, pady=4)

        ttk.Label(settings, text="Tracking side:").grid(row=0, column=4, sticky="w", padx=4, pady=4)
        self.side_combo = ttk.Combobox(
            settings, textvariable=self.side_var, values=["Left", "Right"],
            state="readonly", width=10
        )
        self.side_combo.grid(row=0, column=5, padx=4, pady=4)

        ttk.Label(settings, text="Camera index:").grid(row=0, column=6, sticky="w", padx=4, pady=4)
        self.camera_entry = ttk.Entry(settings, textvariable=self.camera_var, width=5)
        self.camera_entry.grid(row=0, column=7, padx=4, pady=4)

        ttk.Label(settings, text="Save folder:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.output_entry = ttk.Entry(settings, textvariable=self.output_var)
        self.output_entry.grid(row=1, column=1, columnspan=6, sticky="ew", padx=4, pady=4)
        self.browse_button = ttk.Button(settings, text="Browse...", command=self.choose_output_folder)
        self.browse_button.grid(row=1, column=7, padx=4, pady=4)
        settings.columnconfigure(6, weight=0)
        settings.columnconfigure(1, weight=1)

        main = ttk.Frame(outer)
        main.pack(fill=tk.BOTH, expand=True)

        video_frame = ttk.LabelFrame(main, text="Live camera / pose tracking", padding=5)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame, anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        side_panel = ttk.Frame(main, width=250)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side_panel.pack_propagate(False)

        angles_box = ttk.LabelFrame(side_panel, text="Live joint angles", padding=10)
        angles_box.pack(fill=tk.X, pady=(0, 8))

        self.hip_label = ttk.Label(angles_box, text="Hip: --", font=("Segoe UI", 15, "bold"))
        self.hip_label.pack(anchor="w", pady=6)
        self.knee_label = ttk.Label(angles_box, text="Knee: --", font=("Segoe UI", 15, "bold"))
        self.knee_label.pack(anchor="w", pady=6)
        self.ankle_label = ttk.Label(angles_box, text="Ankle: --", font=("Segoe UI", 15, "bold"))
        self.ankle_label.pack(anchor="w", pady=6)

        cal_box = ttk.LabelFrame(side_panel, text="Neutral calibration", padding=10)
        cal_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            cal_box,
            text="Stand in a comfortable upright pose with the selected side visible, then calibrate.",
            wraplength=210
        ).pack(anchor="w", pady=(0, 8))
        self.calibrate_button = ttk.Button(
            cal_box, text="SET NEUTRAL POSE", command=self.calibrate_neutral
        )
        self.calibrate_button.pack(fill=tk.X)
        self.neutral_status = ttk.Label(cal_box, text="Neutral: not set", wraplength=210)
        self.neutral_status.pack(anchor="w", pady=(8, 0))

        rec_box = ttk.LabelFrame(side_panel, text="Trial recording", padding=10)
        rec_box.pack(fill=tk.X, pady=(0, 8))
        self.start_button = ttk.Button(
            rec_box, text="START RECORDING", command=self.start_recording
        )
        self.start_button.pack(fill=tk.X, pady=4)
        self.stop_button = ttk.Button(
            rec_box, text="STOP RECORDING", command=self.stop_recording, state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X, pady=4)
        self.recording_label = ttk.Label(rec_box, text="Not recording", foreground="gray")
        self.recording_label.pack(anchor="w", pady=(6, 0))

        info_box = ttk.LabelFrame(side_panel, text="Tracking information", padding=10)
        info_box.pack(fill=tk.X, pady=(0, 8))
        self.fps_label = ttk.Label(info_box, text="FPS: --")
        self.fps_label.pack(anchor="w", pady=3)
        self.frame_label = ttk.Label(info_box, text="Frame: 0")
        self.frame_label.pack(anchor="w", pady=3)
        self.side_status_label = ttk.Label(info_box, text="Landmarks: waiting", wraplength=210)
        self.side_status_label.pack(anchor="w", pady=3)

        self.status_var = tk.StringVar(value="Initializing...")
        status = ttk.Label(outer, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status.pack(fill=tk.X, pady=(8, 0))

        help_text = (
            "Camera setup: side-on sagittal view, full body/leg visible, good lighting. "
            "Press SET NEUTRAL before recording for relative joint angles."
        )
        ttk.Label(outer, text=help_text, wraplength=1250).pack(anchor="w", pady=(6, 0))

    # -----------------------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------------------

    def _initialize_pose(self):
        try:
            self.pose_model = POSE.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        except Exception as exc:
            messagebox.showerror(
                "MediaPipe initialization error",
                f"Could not initialize MediaPipe Pose.\n\n{exc}\n\n"
                "Try: python -m pip install mediapipe==0.10.21"
            )
            self.pose_model = None

    def _initialize_camera(self):
        try:
            camera_index = int(self.camera_var.get())
        except ValueError:
            camera_index = 0
            self.camera_var.set("0")

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(camera_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        else:
            messagebox.showerror(
                "Camera error",
                f"Unable to open camera index {camera_index}.\n"
                "Close other applications using the webcam or try camera index 1."
            )

    # -----------------------------------------------------------------
    # SETTINGS
    # -----------------------------------------------------------------

    def choose_output_folder(self):
        selected = filedialog.askdirectory(
            title="Choose folder for gait recordings",
            initialdir=self.output_var.get() if Path(self.output_var.get()).exists() else str(Path.home())
        )
        if selected:
            self.output_var.set(selected)

    def _set_controls_recording_state(self, recording):
        state = tk.DISABLED if recording else tk.NORMAL
        for widget in (
            self.subject_entry, self.activity_combo, self.side_combo,
            self.camera_entry, self.output_entry, self.browse_button
        ):
            widget.configure(state=state)
        self.start_button.configure(state=tk.DISABLED if recording else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if recording else tk.DISABLED)
        self.calibrate_button.configure(state=tk.DISABLED if recording else tk.NORMAL)

    def _selected_landmark_ids(self):
        side_prefix = "LEFT_" if self.side_var.get() == "Left" else "RIGHT_"
        names = ["shoulder", "hip", "knee", "ankle", "heel", "foot_index"]
        enum_names = {
            "shoulder": side_prefix + "SHOULDER",
            "hip": side_prefix + "HIP",
            "knee": side_prefix + "KNEE",
            "ankle": side_prefix + "ANKLE",
            "heel": side_prefix + "HEEL",
            "foot_index": side_prefix + "FOOT_INDEX",
        }
        return {
            name: getattr(POSE.PoseLandmark, enum_names[name]).value
            for name in names
        }

    # -----------------------------------------------------------------
    # LANDMARK EXTRACTION / FILTERING
    # -----------------------------------------------------------------

    def _extract_points(self, frame, pose_result):
        if not pose_result.pose_landmarks:
            return None, None

        h, w = frame.shape[:2]
        ids = self._selected_landmark_ids()
        landmarks = pose_result.pose_landmarks.landmark

        points = {}
        normalized = {}
        required = ["shoulder", "hip", "knee", "ankle", "heel", "foot_index"]

        for name in required:
            lm = landmarks[ids[name]]
            if lm.visibility < VISIBILITY_THRESHOLD:
                return None, None
            x = float(lm.x * w)
            y = float(lm.y * h)
            points[name] = (x, y)
            normalized[name] = (float(lm.x), float(lm.y), float(lm.z), float(lm.visibility))

        return points, normalized

    def _smooth_points(self, points):
        if self.smoothed_points is None:
            self.smoothed_points = {
                name: np.array(point, dtype=float) for name, point in points.items()
            }
        else:
            a = SMOOTHING_ALPHA
            for name, point in points.items():
                current = np.array(point, dtype=float)
                self.smoothed_points[name] = (
                    a * current + (1.0 - a) * self.smoothed_points[name]
                )
        return {
            name: (float(value[0]), float(value[1]))
            for name, value in self.smoothed_points.items()
        }

    # -----------------------------------------------------------------
    # CALIBRATION
    # -----------------------------------------------------------------

    def calibrate_neutral(self):
        if self.recording:
            return
        if self.current_angles is None:
            messagebox.showwarning(
                "Calibration unavailable",
                "The selected-side landmarks are not currently visible. "
                "Move into view and try again."
            )
            return

        required = (
            self.current_angles["hip_raw_deg"],
            self.current_angles["knee_flexion_deg"],
            self.current_angles["ankle_raw_deg"],
        )
        if any(value is None or not np.isfinite(value) for value in required):
            messagebox.showwarning("Calibration unavailable", "Joint angle calculation is invalid.")
            return

        self.neutral_angles = {
            "hip": float(required[0]),
            "knee": float(required[1]),
            "ankle": float(required[2]),
        }
        self.neutral_status.configure(
            text=(
                f"Neutral set | Hip {required[0]:.1f}°, "
                f"Knee {required[1]:.1f}°, Ankle {required[2]:.1f}°"
            )
        )
        self.status_var.set("Neutral pose calibrated. Relative angles will be recorded.")

    # -----------------------------------------------------------------
    # RECORDING
    # -----------------------------------------------------------------

    def _next_trial_path(self):
        subject = safe_name(self.subject_var.get(), "Subject_Unknown")
        activity = safe_name(self.activity_var.get(), "Activity_Unknown")
        base = Path(self.output_var.get()).expanduser()
        folder = base / subject / activity
        folder.mkdir(parents=True, exist_ok=True)

        existing = list(folder.glob("Trial_*.csv"))
        numbers = []
        for path in existing:
            match = re.match(r"Trial_(\d+)_", path.name)
            if match:
                numbers.append(int(match.group(1)))
        trial_number = max(numbers, default=0) + 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Trial_{trial_number:02d}_{stamp}.csv"
        return folder / filename, subject, activity, trial_number

    def start_recording(self):
        if self.recording:
            return
        if self.current_points is None or self.current_angles is None:
            messagebox.showwarning(
                "Tracking unavailable",
                "The required body landmarks are not visible. Ensure the selected side "
                "is fully visible before starting a trial."
            )
            return

        if self.neutral_angles is None:
            answer = messagebox.askyesno(
                "Neutral calibration",
                "A neutral pose has not been calibrated.\n\n"
                "Do you want to record raw angles without neutral-relative angles?"
            )
            if not answer:
                return

        try:
            path, subject, activity, trial_number = self._next_trial_path()
            handle = open(path, "w", newline="", encoding="utf-8-sig")
        except Exception as exc:
            messagebox.showerror("File error", f"Could not create recording file.\n\n{exc}")
            return

        self.csv_handle = handle
        self.csv_writer = csv.writer(handle)
        self.recording_path = path
        self.record_start_perf = time.perf_counter()
        self.record_frame_count = 0

        header = [
            "Timestamp_ISO", "Elapsed_Time_s", "Frame_Number",
            "Subject_ID", "Activity", "Trial_Number", "Tracking_Side",
            "Tracking_Valid",
        ]
        for name in ("shoulder", "hip", "knee", "ankle", "heel", "foot_index"):
            header.extend([
                f"{name}_X_px", f"{name}_Y_px",
                f"{name}_X_norm", f"{name}_Y_norm",
                f"{name}_Z_norm", f"{name}_Visibility",
            ])
        header.extend([
            "Hip_Raw_deg", "Hip_Relative_deg",
            "Knee_Flexion_deg", "Knee_Relative_deg",
            "Ankle_Raw_deg", "Ankle_Relative_deg",
            "Neutral_Calibrated",
        ])
        self.csv_writer.writerow(header)

        self.recording = True
        self._active_subject = subject
        self._active_activity = activity
        self._active_trial = trial_number
        self._set_controls_recording_state(True)
        self.recording_label.configure(text=f"RECORDING: Trial {trial_number:02d}", foreground="red")
        self.status_var.set(f"Recording started: {path}")

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        try:
            if self.csv_handle:
                self.csv_handle.flush()
                self.csv_handle.close()
        except Exception:
            pass

        saved_path = self.recording_path
        count = self.record_frame_count
        self.csv_handle = None
        self.csv_writer = None
        self.recording_path = None
        self.record_start_perf = None

        self._set_controls_recording_state(False)
        self.recording_label.configure(text="Not recording", foreground="gray")
        self.status_var.set(f"Recording saved ({count} frames): {saved_path}")
        messagebox.showinfo(
            "Trial saved",
            f"Recording saved successfully.\n\n"
            f"Frames: {count}\n"
            f"File: {saved_path}"
        )

    def _write_csv_row(self, valid, points=None, normalized=None, angles=None):
        if not self.recording or self.csv_writer is None:
            return

        elapsed = time.perf_counter() - self.record_start_perf
        timestamp = datetime.now().isoformat(timespec="milliseconds")

        row = [
            timestamp, f"{elapsed:.6f}", self.record_frame_count,
            self._active_subject, self._active_activity, self._active_trial,
            self.side_var.get(), int(bool(valid)),
        ]

        names = ("shoulder", "hip", "knee", "ankle", "heel", "foot_index")
        for name in names:
            if valid and points is not None and name in points:
                x, y = points[name]
                norm = normalized.get(name) if normalized else None
                if norm is None:
                    row.extend([f"{x:.3f}", f"{y:.3f}", "", "", "", ""])
                else:
                    row.extend([
                        f"{x:.3f}", f"{y:.3f}",
                        f"{norm[0]:.7f}", f"{norm[1]:.7f}",
                        f"{norm[2]:.7f}", f"{norm[3]:.5f}",
                    ])
            else:
                row.extend(["", "", "", "", "", ""])

        if valid and angles is not None:
            hip_raw = angles["hip_raw_deg"]
            knee_raw = angles["knee_flexion_deg"]
            ankle_raw = angles["ankle_raw_deg"]

            hip_rel = (
                wrap_degrees(hip_raw - self.neutral_angles["hip"])
                if self.neutral_angles is not None and hip_raw is not None else None
            )
            knee_rel = (
                knee_raw - self.neutral_angles["knee"]
                if self.neutral_angles is not None and knee_raw is not None else None
            )
            ankle_rel = (
                wrap_degrees(ankle_raw - self.neutral_angles["ankle"])
                if self.neutral_angles is not None and ankle_raw is not None else None
            )

            row.extend([
                self._format_number(hip_raw), self._format_number(hip_rel),
                self._format_number(knee_raw), self._format_number(knee_rel),
                self._format_number(ankle_raw), self._format_number(ankle_rel),
                int(self.neutral_angles is not None),
            ])
        else:
            row.extend(["", "", "", "", "", "", int(self.neutral_angles is not None)])

        self.csv_writer.writerow(row)
        self.record_frame_count += 1

        # Flush regularly to reduce data loss if the application closes unexpectedly.
        if self.record_frame_count % 30 == 0:
            self.csv_handle.flush()

    @staticmethod
    def _format_number(value):
        if value is None or not np.isfinite(value):
            return ""
        return f"{float(value):.6f}"

    # -----------------------------------------------------------------
    # DRAWING / DISPLAY
    # -----------------------------------------------------------------

    def _draw_overlay(self, frame, points, angles, valid):
        if valid and points is not None:
            p = {name: (int(round(x)), int(round(y))) for name, (x, y) in points.items()}

            connections = [
                ("shoulder", "hip"),
                ("hip", "knee"),
                ("knee", "ankle"),
                ("ankle", "heel"),
                ("heel", "foot_index"),
                ("ankle", "foot_index"),
            ]
            for a, b in connections:
                cv2.line(frame, p[a], p[b], (0, 255, 255), 3, cv2.LINE_AA)

            colors = {
                "shoulder": (255, 255, 0),
                "hip": (255, 0, 255),
                "knee": (0, 0, 255),
                "ankle": (0, 255, 0),
                "heel": (255, 128, 0),
                "foot_index": (255, 0, 0),
            }
            for name, point in p.items():
                cv2.circle(frame, point, 7, colors[name], -1, cv2.LINE_AA)
                cv2.putText(
                    frame, name.upper(), (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colors[name], 2, cv2.LINE_AA
                )

            if angles:
                text_lines = [
                    f"Hip raw: {self._format_number(angles['hip_raw_deg'])} deg",
                    f"Knee flexion: {self._format_number(angles['knee_flexion_deg'])} deg",
                    f"Ankle raw: {self._format_number(angles['ankle_raw_deg'])} deg",
                ]
                for i, line in enumerate(text_lines):
                    cv2.putText(
                        frame, line, (15, 30 + i * 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
                    )

        if self.recording:
            cv2.circle(frame, (frame.shape[1] - 35, 35), 11, (0, 0, 255), -1)
            cv2.putText(
                frame, "REC", (frame.shape[1] - 90, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
            )

        side_text = f"SIDE: {self.side_var.get().upper()}"
        cv2.putText(
            frame, side_text, (15, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        return frame

    def _update_angle_labels(self, angles):
        if angles is None:
            self.hip_label.configure(text="Hip: --")
            self.knee_label.configure(text="Knee: --")
            self.ankle_label.configure(text="Ankle: --")
            return

        def display(raw, key):
            if raw is None:
                return "--"
            if self.neutral_angles is None:
                return f"{raw:.1f}° raw"
            if key == "knee":
                relative = raw - self.neutral_angles[key]
            else:
                relative = wrap_degrees(raw - self.neutral_angles[key])
            return f"{relative:+.1f}° relative"

        self.hip_label.configure(text=f"Hip: {display(angles['hip_raw_deg'], 'hip')}")
        self.knee_label.configure(text=f"Knee: {display(angles['knee_flexion_deg'], 'knee')}")
        self.ankle_label.configure(text=f"Ankle: {display(angles['ankle_raw_deg'], 'ankle')}")

    # -----------------------------------------------------------------
    # MAIN FRAME LOOP
    # -----------------------------------------------------------------

    def update_frame(self):
        if self.closed or self.cap is None or not self.cap.isOpened():
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Could not read a frame from the webcam.")
            self.after_id = self.root.after(100, self.update_frame)
            return

        self.global_frame_count += 1
        pose_result = None
        points = None
        normalized = None
        angles = None
        valid = False

        if self.pose_model is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_result = self.pose_model.process(rgb)
            rgb.flags.writeable = True

            raw_points, normalized = self._extract_points(frame, pose_result)
            if raw_points is not None:
                points = self._smooth_points(raw_points)
                angles = calculate_joint_angles(points)
                valid = (
                    angles["hip_raw_deg"] is not None
                    and angles["knee_flexion_deg"] is not None
                    and angles["ankle_raw_deg"] is not None
                )
                if valid:
                    self.current_points = points
                    self.current_angles = angles
                    self.last_valid_points = points
                else:
                    points = None
                    normalized = None
                    angles = None

        self._update_angle_labels(angles if valid else None)

        if valid:
            self.current_points = points
            self.current_angles = angles
            self.side_status_label.configure(text="Landmarks: all required points visible")
            self.status_var.set("Tracking selected side.")
        else:
            self.current_points = None
            self.current_angles = None
            self.side_status_label.configure(text="Landmarks: missing / low visibility")
            if not self.recording:
                self.status_var.set("Tracking lost or incomplete. Keep the selected side visible.")

        if self.recording:
            self._write_csv_row(
                valid=valid,
                points=points if valid else None,
                normalized=normalized if valid else None,
                angles=angles if valid else None,
            )

        frame = self._draw_overlay(frame, points, angles, valid)

        now = time.perf_counter()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            instantaneous_fps = 1.0 / dt
            self.current_fps = (
                0.15 * instantaneous_fps + 0.85 * self.current_fps
                if self.current_fps > 0 else instantaneous_fps
            )
        self.fps_label.configure(text=f"FPS: {self.current_fps:.1f}")
        self.frame_label.configure(text=f"Frame: {self.global_frame_count}")

        display = cv2.resize(
            frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA
        )
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(display_rgb)
        self.photo_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo_image)

        self.after_id = self.root.after(FRAME_INTERVAL_MS, self.update_frame)

    # -----------------------------------------------------------------
    # SHUTDOWN
    # -----------------------------------------------------------------

    def close_application(self):
        if self.closed:
            return
        if self.recording:
            answer = messagebox.askyesno(
                "Recording in progress",
                "A trial is currently recording. Stop and save it before closing?"
            )
            if answer:
                self.stop_recording()
            else:
                return

        self.closed = True
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass

        if self.cap is not None:
            self.cap.release()
        if self.pose_model is not None:
            self.pose_model.close()
        self.root.destroy()


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = MarkerlessGaitTracker(root)
    if app.pose_model is None:
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
