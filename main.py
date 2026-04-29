import cv2
import numpy as np
import os

from detector import DroneDetector
from tracker import DroneTracker
from guidance import UAVGuidance
from config import SHOW_TRAILS, TRAIL_LENGTH, SHOW_VELOCITY


def get_random_color(track_id):
    if isinstance(track_id, str):
        seed = hash(track_id) % (2 ** 32)
    else:
        seed = int(track_id)

    np.random.seed(seed)
    return tuple(map(int, np.random.randint(50, 255, 3)))


def draw_guidance(frame, uav_pos, guidance_data, closest_track):
    h, w = frame.shape[:2]
    ux, uy = uav_pos

    # UAV маркер
    cv2.drawMarker(frame, (ux, uy), (0, 255, 0),
                   cv2.MARKER_CROSS, 25, 2)
    cv2.circle(frame, (ux, uy), 40, (0, 255, 0), 2)
    cv2.putText(frame, '', (ux - 20, uy - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if guidance_data is None or closest_track is None:
        cmd_x = 10
        cmd_y = 120

        overlay = frame.copy()
        cv2.rectangle(overlay, (cmd_x, cmd_y),
                      (cmd_x + 250, cmd_y + 80),
                      (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)[:]

        cv2.rectangle(frame, (cmd_x, cmd_y),
                      (cmd_x + 250, cmd_y + 80),
                      (0, 255, 0), 2)

        cv2.putText(frame, "EVASION STATUS", (cmd_x + 10, cmd_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "✓ NO THREAT", (cmd_x + 40, cmd_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return

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
    if closest_track and uncertainty > 0:
        tx, ty = closest_track['center']

        num_circles = 5
        for i in range(num_circles):
            t = (i + 1) / num_circles

            # Інтерполяція позиції
            cone_x = int(tx + (ex - tx) * t)
            cone_y = int(ty + (ey - ty) * t)
            radius = int(uncertainty * t)
            alpha = 0.3 - (t * 0.2)

            # Малюємо коло
            overlay = frame.copy()
            cv2.circle(overlay, (cone_x, cone_y), radius, (255, 0, 0), 2)
            frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)[:]

    # === ТОЧКА УХИЛЕННЯ ===
    cv2.circle(frame, (ex, ey), 15, threat_color, 2)
    cv2.circle(frame, (ex, ey), 10, threat_color, -1)
    cv2.putText(frame, "EVADE", (ex - 25, ey - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, threat_color, 2)

    # === ЛІНІЯ УХИЛЕННЯ ===
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

    # === КОМАНДИ ===
    cmd_x = 10
    cmd_y = 120
    cmd_width = 270
    cmd_height = 40 + len(commands) * 35

    overlay = frame.copy()
    cv2.rectangle(overlay, (cmd_x, cmd_y),
                  (cmd_x + cmd_width, cmd_y + cmd_height),
                  (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)[:]

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
    frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)[:]

    cv2.rectangle(frame, (info_x, info_y),
                  (info_x + info_width, info_y + info_height),
                  threat_color, 2)

    cv2.putText(frame, f"THREAT: ID-{closest_track['id']}",
                (info_x + 10, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, threat_color, 2)

    cv2.putText(frame, f"Level: {threat_level}",
                (info_x + 10, info_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, threat_color, 2)

    # === Pattern ===
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

    # === Uncertainty ===
    cv2.putText(frame, f"Uncertainty: ±{uncertainty:.0f} px",
                (info_x + 10, info_y + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
def main():
    print("=" * 60)
    print("UAV INTERCEPT SYSTEM")
    print("=" * 60)

    detector = DroneDetector()
    tracker = DroneTracker()

    video_path = input("\nШлях до відео: ").strip()

    if not os.path.exists(video_path):
        print(f"Файл не знайдено: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    guidance = UAVGuidance(width, height)

    print(f"\n Відео: {width}x{height} @ {fps}fps")
    print(f"Кадрів: {total_frames}")
    print("\n Запуск...")
    print("Q - вихід | S - скріншот | P - пауза\n")

    frame_count = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nВідео закінчилось")
                break
            frame_count += 1

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        # Фільтруємо тільки confirmed треки
        confirmed_tracks = [t for t in tracks if t.get('hits', 0) > 3]

        closest = None
        min_dist = float('inf')

        for track in confirmed_tracks:
            cx, cy = track['center']
            dist = np.sqrt((cx - guidance.uav_x) ** 2 + (cy - guidance.uav_y) ** 2)

            if dist < min_dist:
                min_dist = dist
                closest = track

        guidance_data = guidance.get_commands(closest) if closest else None

        # === ВІЗУАЛІЗАЦІЯ ТРЕКІВ ===
        for track in confirmed_tracks:
            tid = track['id']
            x1, y1, x2, y2 = track['bbox']
            cx, cy = track['center']
            vx, vy = track['velocity']

            is_target = (track == closest)

            if is_target:
                color = (0, 0, 255)
                thickness = 3
            else:
                color = get_random_color(tid)
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Траєкторія для цілі
            if SHOW_TRAILS and is_target:
                trajectory = tracker.get_trajectory(tid, TRAIL_LENGTH)
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)

            label = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.circle(frame, (cx, cy), 4, color, -1)

            if SHOW_VELOCITY and is_target:
                speed = np.sqrt(vx ** 2 + vy ** 2)
                if speed > 1:
                    scale = 5
                    end_x = int(cx + vx * scale)
                    end_y = int(cy + vy * scale)
                    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y),
                                    (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)

        draw_guidance(frame, (guidance.uav_x, guidance.uav_y),
                      guidance_data, closest)

        # === ВЕРХНЯ ПАНЕЛЬ ===
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 110), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)[:]

        progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Tracks: {len(confirmed_tracks)}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if len(confirmed_tracks) > 0 else (255, 255, 255), 2)

        status = "PAUSED" if paused else "TRACKING"
        status_color = (0, 165, 255) if paused else (0, 255, 0)
        cv2.putText(frame, status, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # === НИЖНЯ ПАНЕЛЬ ===
        bar_h = 8
        bar_y = height - bar_h - 2

        cv2.rectangle(frame, (0, bar_y), (width, height), (40, 40, 40), -1)
        bar_w = int((frame_count / total_frames) * width) if total_frames > 0 else 0
        cv2.rectangle(frame, (0, bar_y), (bar_w, height), (0, 255, 0), -1)

        hint_text = "Q-quit | S-screenshot | P-pause"
        (tw, th), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, hint_text, (width - tw - 10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('UAV Intercept System - DeepSORT', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n Зупинено користувачем")
            break
        elif key == ord('s'):
            fname = f"screenshot_{frame_count:06d}.jpg"
            cv2.imwrite(fname, frame)
            print(f" {fname}")
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    print(f"Кадрів оброблено: {frame_count}")
    print(f"Унікальних треків: {len(tracker.track_history)}")


if __name__ == "__main__":
    main()