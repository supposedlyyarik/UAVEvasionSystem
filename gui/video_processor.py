import cv2
import numpy as np
from threading import Thread
from queue import Queue
import time
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from detector import DroneDetector
from tracker import DroneTracker
from guidance import UAVGuidance
from logger import UAVLogger

class VideoProcessor:

    def __init__(self):
        self.detector = None
        self.tracker = None
        self.guidance = None
        self.logger = None
        self.is_running = False
        self.is_paused = False
        self.current_frame = None
        self.processed_frame = None

        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0

        self.settings = {
            'model': 'yolov8s.pt',
            'confidence': 0.15,
            'iou': 0.3,
            'show_trails': True,
            'show_velocity': True,
            'show_uav': True,
            'show_cone': True,
            'show_commands': True,
            'uav_speed': 15,
            'command_threshold': 50
        }

        self.thread = None
        self.frame_queue = Queue(maxsize=2)

        self.cap = None
        self.video_path = None

    def load_video(self, path):
        self.video_path = path

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.detector = DroneDetector()
        self.tracker = DroneTracker()
        self.guidance = UAVGuidance(width, height)

        self.frame_count = 0
        self.logger = UAVLogger(session_name=f"")

        self.logger.log_event("VIDEO", "Video loaded", {
            'path': path,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': self.total_frames
        })
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': self.total_frames
        }

    def update_settings(self, settings_dict):
        self.settings.update(settings_dict)

        if 'model' in settings_dict and self.detector:
            self.detector = DroneDetector()

    def start(self):
        """Запустити обробку"""
        if not self.cap:
            raise ValueError("No video loaded")

        self.is_running = True
        self.is_paused = False

        self.thread = Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def pause(self):
        self.is_paused = not self.is_paused

    def _process_loop(self):
        fps_start = time.time()
        fps_counter = 0

        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue

            # Читаємо кадр
            ret, frame = self.cap.read()

            if not ret:
                self.is_running = False
                break

            self.frame_count += 1
            self.current_frame = frame.copy()


            detections = self.detector.detect(frame)
            for det in detections:
                self.logger.log_detection(
                    self.frame_count,
                    det['bbox'],
                    det['conf'],
                    det['class']
                )
            tracks = self.tracker.update(detections, frame)

            # Фільтр
            confirmed_tracks = [t for t in tracks if t.get('hits', 0) > 3]
            for track in confirmed_tracks:
                self.logger.log_track(
                    self.frame_count,
                    track['id'],
                    track['center'],
                    track['velocity'],
                    track['hits']
                )
            self.logger.log_frame(
                self.frame_count,
                len(detections),
                len(confirmed_tracks)
            )
            # Найближча загроза
            closest = None
            if confirmed_tracks:
                min_dist = float('inf')
                for track in confirmed_tracks:
                    cx, cy = track['center']
                    dist = np.sqrt(
                        (cx - self.guidance.uav_x) ** 2 +
                        (cy - self.guidance.uav_y) ** 2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest = track

            # Guidance
            guidance_data = self.guidance.get_commands(closest) if closest else None
            if guidance_data and closest:
                vx, vy = closest['velocity']
                speed = np.sqrt(vx ** 2 + vy ** 2)

                self.logger.log_threat(
                    self.frame_count,
                    closest['id'],
                    guidance_data['distance'],
                    guidance_data['threat_level'],
                    guidance_data.get('pattern', 'unknown'),
                    speed,
                    guidance_data.get('uncertainty_radius', 0),
                    guidance_data['evasion_point'],
                    guidance_data['commands']
                )
            processed = self._draw_frame(
                frame.copy(),
                confirmed_tracks,
                closest,
                guidance_data
            )

            if not self.frame_queue.full():
                self.frame_queue.put(processed)

            # FPS
            fps_counter += 1
            if fps_counter >= 30:
                elapsed = time.time() - fps_start
                self.fps = fps_counter / elapsed
                loop_time = (time.time() - fps_start) * 1000
                self.logger.log_statistics(self.fps, loop_time)
                fps_start = time.time()
                fps_counter = 0

        if self.logger:
            self.logger.close()

        if self.cap:
            self.cap.release()

    def _draw_frame(self, frame, tracks, closest, guidance_data):
        h, w = frame.shape[:2]

        for track in tracks:
            tid = track['id']
            x1, y1, x2, y2 = track['bbox']
            cx, cy = track['center']
            vx, vy = track['velocity']

            is_target = (track == closest)

            if is_target:
                color = (0, 0, 255)
                thickness = 3
            else:
                color = self._get_color(tid)
                thickness = 2

            # Bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # ID
            label = f"ID:{tid}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Траєкторія
            if self.settings['show_trails'] and is_target:
                trajectory = self.tracker.get_trajectory(tid, 30)
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)

            # Швидкість
            if self.settings['show_velocity'] and is_target:
                speed = np.sqrt(vx ** 2 + vy ** 2)
                if speed > 1:
                    ex = int(cx + vx * 5)
                    ey = int(cy + vy * 5)
                    cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                                    (255, 255, 0), 2, tipLength=0.3)

        # === UAV GUIDANCE ===
        if self.settings['show_uav']:
            self._draw_guidance(frame, guidance_data, closest)

        # === INFO PANEL ===
        self._draw_info_panel(frame, tracks, guidance_data)

        return frame

    def _draw_guidance(self, frame, guidance_data, closest_track):
        h, w = frame.shape[:2]
        ux, uy = self.guidance.uav_x, self.guidance.uav_y

        # UAV маркер
        if self.settings['show_uav']:
            cv2.drawMarker(frame, (ux, uy), (0, 255, 0),
                           cv2.MARKER_CROSS, 25, 2)
            cv2.circle(frame, (ux, uy), 40, (0, 255, 0), 2)
            cv2.putText(frame, "UAV", (ux - 20, uy - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Якщо немає цілі
        if guidance_data is None or closest_track is None:
            if self.settings['show_commands']:
                cmd_x = 10
                cmd_y = 120

                overlay = frame.copy()
                cv2.rectangle(overlay, (cmd_x, cmd_y),
                              (cmd_x + 250, cmd_y + 80),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                cv2.rectangle(frame, (cmd_x, cmd_y),
                              (cmd_x + 250, cmd_y + 80),
                              (0, 255, 0), 2)

                cv2.putText(frame, "EVASION STATUS", (cmd_x + 10, cmd_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, "✓ NO THREAT", (cmd_x + 40, cmd_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return

        # target found
        ex, ey = guidance_data['evasion_point']
        dist = guidance_data['distance']
        threat_level = guidance_data['threat_level']
        commands = guidance_data['commands']
        pattern = guidance_data.get('pattern', 'unknown')
        uncertainty = guidance_data.get('uncertainty_radius', 0)

        if threat_level == "CRITICAL":
            threat_color = (0, 0, 255)
        elif threat_level == "HIGH":
            threat_color = (0, 165, 255)
        elif threat_level == "MEDIUM":
            threat_color = (0, 255, 255)
        else:
            threat_color = (0, 255, 0)

        # === КОНУС НЕВИЗНАЧЕНОСТІ ===
        if self.settings['show_cone'] and uncertainty > 0:
            tx, ty = closest_track['center']

            num_circles = 5
            for i in range(num_circles):
                t = (i + 1) / num_circles

                cone_x = int(tx + (ex - tx) * t)
                cone_y = int(ty + (ey - ty) * t)

                radius = int(uncertainty * t)
                alpha = 0.3 - (t * 0.2)

                overlay = frame.copy()
                cv2.circle(overlay, (cone_x, cone_y), radius, (255, 0, 0), 2)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # === ТОЧКА УХИЛЕННЯ ===
        cv2.circle(frame, (ex, ey), 15, threat_color, 2)
        cv2.circle(frame, (ex, ey), 10, threat_color, -1)
        cv2.putText(frame, "EVADE", (ex - 25, ey - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, threat_color, 2)

        # === ЛІНІЯ УХИЛЕННЯ (пунктир) ===
        line_length = np.sqrt((ex - ux) ** 2 + (ey - uy) ** 2)
        num_dashes = max(1, int(line_length / 20))

        for i in range(num_dashes):
            t1 = i / num_dashes
            t2 = (i + 0.5) / num_dashes

            x1 = int(ux + (ex - ux) * t1)
            y1 = int(uy + (ey - uy) * t1)
            x2 = int(ux + (ex - ux) * t2)
            y2 = int(uy + (ey - uy) * t2)

            cv2.line(frame, (x1, y1), (x2, y2), threat_color, 2, cv2.LINE_AA)

        # === ЛІНІЯ ДО ЗАГРОЗИ ===
        tx, ty = closest_track['center']
        cv2.line(frame, (ux, uy), (tx, ty), (0, 0, 255), 1, cv2.LINE_AA)

        # commands
        if self.settings['show_commands']:
            cmd_x = 10
            cmd_y = 120
            cmd_width = 270
            cmd_height = 40 + len(commands) * 35

            overlay = frame.copy()
            cv2.rectangle(overlay, (cmd_x, cmd_y),
                          (cmd_x + cmd_width, cmd_y + cmd_height),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv2.rectangle(frame, (cmd_x, cmd_y),
                          (cmd_x + cmd_width, cmd_y + cmd_height),
                          threat_color, 2)

            cv2.putText(frame, "EVASION COMMANDS", (cmd_x + 10, cmd_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, threat_color, 2)

            for i, cmd in enumerate(commands):
                y = cmd_y + 65 + i * 35

                if "RIGHT" in cmd:
                    icon = "→"
                elif "LEFT" in cmd:
                    icon = "←"
                elif "UP" in cmd:
                    icon = "↑"
                elif "DOWN" in cmd:
                    icon = "↓"
                elif "THROTTLE" in cmd:
                    icon = "⚡"
                elif "NOW" in cmd:
                    icon = "⚠️"
                else:
                    icon = "✓"

                cv2.putText(frame, f"{icon} {cmd}", (cmd_x + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # === THREAT INFO ===
        info_x = 10
        info_y = h - 200
        info_width = 300
        info_height = 190

        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y),
                      (info_x + info_width, info_y + info_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.rectangle(frame, (info_x, info_y),
                      (info_x + info_width, info_y + info_height),
                      threat_color, 2)

        cv2.putText(frame, f"THREAT: ID-{closest_track['id']}",
                    (info_x + 10, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, threat_color, 2)

        cv2.putText(frame, f"Level: {threat_level}",
                    (info_x + 10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, threat_color, 2)

        pattern_text = pattern.upper().replace('_', ' ')
        cv2.putText(frame, f"Pattern: {pattern_text}",
                    (info_x + 10, info_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"Distance: {dist:.0f} px",
                    (info_x + 10, info_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        vx, vy = closest_track['velocity']
        speed = np.sqrt(vx ** 2 + vy ** 2)
        cv2.putText(frame, f"Speed: {speed:.1f} px/f",
                    (info_x + 10, info_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, f"Uncertainty: ±{uncertainty:.0f} px",
                    (info_x + 10, info_y + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)

    def _draw_info_panel(self, frame, tracks, guidance_data):
        """Інфо панель зверху"""
        h, w = frame.shape[:2]

        # Фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)[:]

        # Текст
        progress = (self.frame_count / self.total_frames * 100) if self.total_frames > 0 else 0

        cv2.putText(frame, f"Frame: {self.frame_count}/{self.total_frames} ({progress:.1f}%)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Tracks: {len(tracks)}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _get_color(self, track_id):
        np.random.seed(int(track_id) if isinstance(track_id, int) else hash(track_id) % (2 ** 32))
        return tuple(map(int, np.random.randint(50, 255, 3)))

    def get_frame(self):
        if not self.frame_queue.empty():
            self.processed_frame = self.frame_queue.get()

        return self.processed_frame

    def get_stats(self):
        return {
            'frame_count': self.frame_count,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'is_running': self.is_running,
            'is_paused': self.is_paused
        }