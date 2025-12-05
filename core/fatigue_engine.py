import csv
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Landmarks para EAR y boca
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_L = 78
MOUTH_R = 308


class FatigueEngine:
    """
    Motor de detección de fatiga reutilizable tanto para desktop como para móvil.
    """

    def __init__(
        self,
        eye_model_path: str,
        yawn_model_path: str,
        calibration_time: float = 5.0,
        perclos_window: int = 20,
        yawn_horizon: float = 15.0,
        yawn_conf: float = 0.55,
        yawn_conf_strict: float = 0.70,
        mouth_open_min: float = 13.0,
        smooth_decay: float = 0.985,
        smooth_up_avg: bool = True,
        perclos_som: float = 12,
        perclos_fat: float = 23,
        yawn_som: float = 15.0,
        yawn_fat: float = 30.0,
        log_path: Optional[str] = None,
    ) -> None:
        # Almacenar configuración
        self.calibration_time = calibration_time
        self.perclos_window_size = perclos_window
        self.yawn_horizon = yawn_horizon
        self.yawn_conf = yawn_conf
        self.yawn_conf_strict = yawn_conf_strict
        self.mouth_open_min = mouth_open_min
        self.smooth_decay = smooth_decay
        self.smooth_up_avg = smooth_up_avg
        self.perclos_som = perclos_som
        self.perclos_fat = perclos_fat
        self.yawn_som = yawn_som
        self.yawn_fat = yawn_fat
        self.log_path = log_path

        # Modelos
        self.model_eye = YOLO(eye_model_path)
        self.model_yawn = YOLO(yawn_model_path)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Estado interno
        self.reset()

        if self.log_path:
            self._ensure_log_file()

    def reset(self) -> None:
        """Reinicia contadores y estado interno."""
        self.closed_frames = 0
        self.total_frames = 0
        self.perclos_window = deque(maxlen=self.perclos_window_size)
        self.yawn_active = False
        self.yawn_start = 0.0
        self.yawn_durations = []
        self.yawn_ratio = 0.0
        self.calibrated_ear: Optional[float] = None
        self.calibrating = True
        self.calib_start: Optional[float] = None
        self._ear_samples = []
        self.prev_time = time.time()
        self.fps = 0

    @staticmethod
    def _ear(pts):
        a = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        b = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        c = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (a + b) / (2.0 * c + 1e-6)

    def start_calibration(self) -> None:
        self.calibrating = True
        self.calib_start = time.time()
        self._ear_samples = []

    def _ensure_log_file(self) -> None:
        if self.log_path and not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "perclos", "yawn_ratio", "estado"])

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if not self.log_path:
            return
        self._ensure_log_file()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    round(metrics.get("avg_perclos", 0.0), 2),
                    round(metrics.get("yawn_ratio", 0.0), 2),
                    metrics.get("estado", "Desconocido"),
                ]
            )

    def _yolo_yawn(self, crop_bgr) -> Tuple[bool, float]:
        result = self.model_yawn.predict(crop_bgr, imgsz=320, conf=self.yawn_conf, verbose=False)
        if not result:
            return False, 0.0
        res = result[0]
        for cls, conf in zip(res.boxes.cls, res.boxes.conf):
            if self.model_yawn.names[int(cls)] == "yawn":
                return True, float(conf)
        return False, 0.0

    def _update_calibration(self, landmarks_xy, w, h):
        left = [(landmarks_xy[i][0] * w, landmarks_xy[i][1] * h) for i in LEFT_EYE_IDXS]
        right = [(landmarks_xy[i][0] * w, landmarks_xy[i][1] * h) for i in RIGHT_EYE_IDXS]
        ear = (self._ear(left) + self._ear(right)) / 2.0
        self._ear_samples.append(ear)
        if time.time() - (self.calib_start or 0.0) >= self.calibration_time and self._ear_samples:
            self.calibrated_ear = float(np.mean(self._ear_samples))
            self.calibrating = False

    def _compute_estado(self, avg_perclos: float, yawn_ratio: float) -> str:
        if avg_perclos > self.perclos_fat or yawn_ratio > self.yawn_fat:
            return "Fatigado"
        if avg_perclos > self.perclos_som or yawn_ratio > self.yawn_som:
            return "Somnoliento"
        return "Alerta"

    def step(self, frame_bgr) -> Tuple[np.ndarray, Dict[str, float]]:
        h, w, _ = frame_bgr.shape
        annotated = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        self.total_frames += 1
        eyes_closed = False
        face_present = bool(result.multi_face_landmarks)
        estado = "Sin rostro"

        if face_present:
            lm = result.multi_face_landmarks[0]
            pts = [(p.x, p.y) for p in lm.landmark]

            if self.calibrating:
                if self.calib_start is None:
                    self.start_calibration()
                self._update_calibration(pts, w, h)
                remain = max(0, int(self.calibration_time - (time.time() - (self.calib_start or time.time()))))
                cv2.putText(
                    annotated,
                    f"Calibrando EAR: {remain}s",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
            else:
                left = [(pts[i][0] * w, pts[i][1] * h) for i in LEFT_EYE_IDXS]
                right = [(pts[i][0] * w, pts[i][1] * h) for i in RIGHT_EYE_IDXS]
                ear_avg = (self._ear(left) + self._ear(right)) / 2.0
                eye_thresh = (self.calibrated_ear or ear_avg) * 0.75
                eyes_closed = ear_avg < eye_thresh

                for eye in [LEFT_EYE_IDXS, RIGHT_EYE_IDXS]:
                    x1 = int(min([pts[i][0] * w for i in eye]))
                    y1 = int(min([pts[i][1] * h for i in eye]))
                    x2 = int(max([pts[i][0] * w for i in eye]))
                    y2 = int(max([pts[i][1] * h for i in eye]))
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255) if eyes_closed else (0, 255, 0),
                        2,
                    )

                if eyes_closed:
                    self.closed_frames += 1

                mx1 = max(0, int(pts[MOUTH_L][0] * w))
                mx2 = min(w, int(pts[MOUTH_R][0] * w))
                my1 = max(0, int(pts[UPPER_LIP][1] * h))
                my2 = min(h, int(pts[LOWER_LIP][1] * h))

                yawn_detected = False
                conf_yawn = 0.0
                if my2 > my1 and mx2 > mx1:
                    crop = frame_bgr[my1:my2, mx1:mx2]
                    if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                        yawn_detected, conf_yawn = self._yolo_yawn(crop)

                    mouth_open = abs(pts[UPPER_LIP][1] - pts[LOWER_LIP][1]) * h
                    if not (yawn_detected and conf_yawn > self.yawn_conf_strict and mouth_open > self.mouth_open_min):
                        yawn_detected = False

                    cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (255, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"yawn {'YES' if yawn_detected else 'NO'}",
                        (mx1, max(25, my1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                else:
                    yawn_detected = False

                now = time.time()
                if yawn_detected and not self.yawn_active:
                    self.yawn_active = True
                    self.yawn_start = now
                elif not yawn_detected and self.yawn_active:
                    self.yawn_active = False
                    self.yawn_durations.append((now, now - self.yawn_start))

                self.yawn_durations = [(t, d) for (t, d) in self.yawn_durations if now - t < self.yawn_horizon]
                total_yawn_time = sum(d for _, d in self.yawn_durations)
                self.yawn_ratio = min(100.0, (total_yawn_time / self.yawn_horizon) * 100.0)

                estado = self._compute_estado(self._avg_perclos(), self.yawn_ratio)
        else:
            if self.calibrating and self.calib_start is None:
                self.start_calibration()

        perclos = (self.closed_frames / max(1, self.total_frames)) * 100.0
        self.perclos_window.append(perclos)
        if len(self.perclos_window) > 1:
            if eyes_closed and not self.calibrating:
                if self.smooth_up_avg:
                    self.perclos_window[-1] = (self.perclos_window[-1] + self.perclos_window[-2]) / 2.0
            else:
                self.perclos_window[-1] = self.perclos_window[-2] * self.smooth_decay

        avg_perclos = self._avg_perclos()

        now = time.time()
        self.fps = int(1.0 / (now - self.prev_time + 1e-9))
        self.prev_time = now
        cv2.putText(annotated, f"FPS: {self.fps}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2)

        if not self.calibrating:
            color = (0, 255, 0)
            if estado == "Somnoliento":
                color = (0, 165, 255)
            elif estado == "Fatigado":
                color = (0, 0, 255)

            cv2.putText(
                annotated,
                f"PERCLOS: {avg_perclos:.1f}%",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"Estado: {estado}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
            cv2.putText(
                annotated,
                f"Bostezos: {self.yawn_ratio:.0f}%/{int(self.yawn_horizon)}s",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        metrics = {
            "perclos": perclos,
            "avg_perclos": avg_perclos,
            "estado": "Calibrando" if self.calibrating else estado,
            "fps": self.fps,
            "yawn_ratio": self.yawn_ratio,
            "face_present": face_present,
        }
        return annotated, metrics

    def _avg_perclos(self) -> float:
        if not self.perclos_window:
            return 0.0
        return float(np.mean(self.perclos_window))
