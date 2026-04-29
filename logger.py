import json
from datetime import datetime
from pathlib import Path


class UAVLogger:

    def __init__(self, log_dir="logs", session_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        if session_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_name = f"session_{timestamp}"

        self.session_name = session_name

        self.log_file = self.log_dir / f"{session_name}.log"
        self.json_file = self.log_dir / f"{session_name}.json"
        self.csv_file = self.log_dir / f"{session_name}.csv"

        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'events': [],
            'statistics': {
                'total_frames': 0,
                'total_detections': 0,
                'total_tracks': 0,
                'total_threats': 0,
                'total_commands': 0,
                'threat_levels': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                'patterns': {'straight': 0, 'zigzag': 0, 'turning': 0, 'accelerating': 0, 'unknown': 0}
            }
        }

        self._init_log_file()
        self._init_csv_file()

        self.log_event("SYSTEM", "Logger initialized", {
            'session': session_name,
            'log_file': str(self.log_file),
            'json_file': str(self.json_file),
            'csv_file': str(self.csv_file)
        })

    def _init_log_file(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"UAV EVASION SYSTEM LOG\n")
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def _init_csv_file(self):
        with open(self.csv_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,frame,event_type,track_id,distance,speed,threat_level,pattern,commands\n")

    def log_event(self, event_type, message, data=None):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        event = {
            'timestamp': timestamp,
            'type': event_type,
            'message': message
        }

        if data:
            event['data'] = data

        self.session_data['events'].append(event)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] [{event_type:12s}] {message}\n")
            if data:
                for key, value in data.items():
                    f.write(f"    {key}: {value}\n")

    def log_frame(self, frame_num, detections_count, tracks_count):
        self.session_data['statistics']['total_frames'] += 1
        self.session_data['statistics']['total_detections'] += detections_count

        self.log_event("FRAME", f"Frame {frame_num} processed", {
            'frame': frame_num,
            'detections': detections_count,
            'tracks': tracks_count
        })

    def log_detection(self, frame_num, bbox, confidence, class_id):
        self.log_event("DETECTION", f"Drone detected", {
            'frame': frame_num,
            'bbox': bbox,
            'confidence': f"{confidence:.2f}",
            'class': class_id
        })

    def log_track(self, frame_num, track_id, center, velocity, hits):
        self.session_data['statistics']['total_tracks'] += 1

        vx, vy = velocity
        speed = (vx ** 2 + vy ** 2) ** 0.5

        self.log_event("TRACK", f"Track ID:{track_id}", {
            'frame': frame_num,
            'track_id': track_id,
            'position': f"({center[0]}, {center[1]})",
            'velocity': f"({vx:.1f}, {vy:.1f})",
            'speed': f"{speed:.1f} px/f",
            'hits': hits
        })

    def log_threat(self, frame_num, track_id, distance, threat_level, pattern,
                   speed, uncertainty, evasion_point, commands):

        self.session_data['statistics']['total_threats'] += 1
        self.session_data['statistics']['total_commands'] += len(commands)
        self.session_data['statistics']['threat_levels'][threat_level] += 1
        self.session_data['statistics']['patterns'][pattern] += 1

        self.log_event("THREAT", f"Track ID:{track_id} - {threat_level}", {
            'frame': frame_num,
            'track_id': track_id,
            'distance': f"{distance:.1f} px",
            'threat_level': threat_level,
            'pattern': pattern,
            'speed': f"{speed:.1f} px/f",
            'uncertainty': f"±{uncertainty:.1f} px",
            'evasion_point': f"({evasion_point[0]}, {evasion_point[1]})",
            'commands': ', '.join(commands)
        })

        with open(self.csv_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()},{frame_num},THREAT,{track_id},"
                    f"{distance:.1f},{speed:.1f},{threat_level},{pattern},"
                    f"\"{';'.join(commands)}\"\n")

    def log_command(self, frame_num, commands):
        if not commands:
            return

        self.log_event("COMMAND", f"Evasion commands issued", {
            'frame': frame_num,
            'commands': commands
        })

    def log_statistics(self, fps, processing_time):
        self.log_event("STATS", "Performance stats", {
            'fps': f"{fps:.1f}",
            'processing_time': f"{processing_time:.1f} ms"
        })

    def save_session(self):
        self.session_data['end_time'] = datetime.now().isoformat()

        events = self.session_data['events']
        if events:
            duration = (datetime.fromisoformat(self.session_data['end_time']) -
                        datetime.fromisoformat(self.session_data['start_time'])).total_seconds()

            self.session_data['statistics']['duration_seconds'] = duration
            self.session_data['statistics']['avg_fps'] = (
                self.session_data['statistics']['total_frames'] / duration
                if duration > 0 else 0
            )

        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("SESSION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Frames:     {self.session_data['statistics']['total_frames']}\n")
            f.write(f"Total Detections: {self.session_data['statistics']['total_detections']}\n")
            f.write(f"Total Threats:    {self.session_data['statistics']['total_threats']}\n")
            f.write(f"Total Commands:   {self.session_data['statistics']['total_commands']}\n")
            f.write(f"\nThreat Levels:\n")
            for level, count in self.session_data['statistics']['threat_levels'].items():
                f.write(f"  {level:10s}: {count}\n")
            f.write(f"\nPatterns:\n")
            for pattern, count in self.session_data['statistics']['patterns'].items():
                f.write(f"  {pattern:12s}: {count}\n")
            f.write("=" * 80 + "\n")

        self.log_event("SYSTEM", "Session saved", {
            'json_file': str(self.json_file),
            'total_events': len(events)
        })

    def close(self):
        self.save_session()
        self.log_event("SYSTEM", "Logger closed")