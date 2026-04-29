from deep_sort_realtime.deepsort_tracker import DeepSort
from config import MAX_AGE, N_INIT, MAX_IOU_DISTANCE, MAX_COSINE_DISTANCE, CLASS_NAMES
class DroneTracker:

    def __init__(self):
        self.tracker = DeepSort(
            max_age=MAX_AGE,
            n_init=N_INIT,
            max_iou_distance=MAX_IOU_DISTANCE,
            max_cosine_distance=MAX_COSINE_DISTANCE,
            embedder="mobilenet",
            half=False,
            embedder_gpu=False
        )
        self.track_history = {}
        # Лічильник hits для фільтрації
        self.track_hits = {}

    def update(self, detections, frame):
        deep_sort_input = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1

            deep_sort_input.append((
                [x1, y1, w, h],
                det['conf'],
                CLASS_NAMES[det['class']]
            ))

        raw_tracks = self.tracker.update_tracks(deep_sort_input, frame=frame)

        tracks = []

        for track in raw_tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Центр
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Оновлення hits
            if track_id not in self.track_hits:
                self.track_hits[track_id] = 0
            self.track_hits[track_id] += 1

            # Оновлення історії
            if track_id not in self.track_history:
                self.track_history[track_id] = []

            self.track_history[track_id].append((cx, cy))

            if len(self.track_history[track_id]) > 100:
                self.track_history[track_id].pop(0)

            vx, vy = self._calculate_velocity(track_id)

            tracks.append({
                'id': track_id,
                'bbox': [x1, y1, x2, y2],
                'center': (cx, cy),
                'velocity': (vx, vy),
                'class': 4,
                'hits': self.track_hits[track_id]  # ← ДОДАЛИ!
            })

        return tracks

    def _calculate_velocity(self, track_id, frames=5):
        if track_id not in self.track_history:
            return 0, 0

        history = self.track_history[track_id]

        if len(history) < 2:
            return 0, 0

        recent = history[-min(frames, len(history)):]

        x_start, y_start = recent[0]
        x_end, y_end = recent[-1]

        dt = len(recent) - 1
        if dt == 0:
            return 0, 0

        vx = (x_end - x_start) / dt
        vy = (y_end - y_start) / dt

        return vx, vy

    def get_trajectory(self, track_id, max_length=30):
        if track_id not in self.track_history:
            return []

        return self.track_history[track_id][-max_length:]