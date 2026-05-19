import numpy as np
from config import UAV_SPEED, COMMAND_THRESHOLD
import time

class UAVGuidance:

    def __init__(self, frame_width, frame_height):
        # UAV в центрі екрану
        self.uav_x = frame_width // 2
        self.uav_y = frame_height // 2
        self.last_command_time = 0
        self.command_cooldown = 1.0
        self.frame_width = frame_width
        self.frame_height = frame_height
        # Історія для аналізу патернів
        self.target_history = {}

    def get_commands(self, target):
        """
        Args:
            target: dict {'id', 'center': (x,y), 'velocity': (vx,vy), 'bbox': [...]}

        Returns:
            dict: {
                'evasion_point': (ex, ey),
                'distance': float,
                'threat_level': str,
                'commands': [...],
                'pattern': str,
                'uncertainty_radius': float
            }
        """
        if target is None:
            return None

        track_id = target['id']
        tx, ty = target['center']
        vx, vy = target['velocity']
        dx_bearing = tx - self.uav_x
        dy_bearing = ty - self.uav_y
        bearing = np.degrees(np.arctan2(dy_bearing, dx_bearing))
        if bearing < 0:
            bearing += 360
        self._update_history(track_id, tx, ty, vx, vy)

        pattern = self._analyze_pattern(track_id)

        predicted_pos, uncertainty = self._predict_with_uncertainty(
            track_id, (tx, ty), (vx, vy), pattern
        )
        real_distance = np.sqrt(
            (tx - self.uav_x) ** 2 + (ty - self.uav_y) ** 2 )
        # Розрахунок точки ухилення
        ex, ey, dist, threat = self._calculate_evasion(
            (self.uav_x, self.uav_y),
            predicted_pos,  # Використовуємо передбачену позицію
            (vx, vy),
            uncertainty
        )
        # Генерація команд
        commands = self._generate_evasion_commands(ex, ey, vx, vy, dist, pattern)

        return {
            'evasion_point': (int(ex), int(ey)),
            'distance': real_distance,
            'threat_level': threat,
            'commands': commands,
            'pattern': pattern,
            'uncertainty_radius': uncertainty,
            'bearing': bearing
        }

    def _update_history(self, track_id, x, y, vx, vy):
        if track_id not in self.target_history:
            self.target_history[track_id] = []

        self.target_history[track_id].append({
            'pos': (x, y),
            'vel': (vx, vy),
            'timestamp': len(self.target_history[track_id])
        })

        if len(self.target_history[track_id]) > 50:
            self.target_history[track_id].pop(0)

    def _analyze_pattern(self, track_id):
        """
        Returns:
            str: 'straight', 'zigzag', 'turning', 'accelerating', 'unknown'
        """
        if track_id not in self.target_history:
            return 'unknown'

        history = self.target_history[track_id]

        # Потрібно мінімум 10 кадрів для аналізу
        if len(history) < 10:
            return 'unknown'
        velocities = [h['vel'] for h in history[-20:]]  # Останні 20

        vx_list = [v[0] for v in velocities]
        vy_list = [v[1] for v in velocities]

        # === АНАЛІЗ 1: ПРЯМОЛІНІЙНИЙ РУХ ===
        # Перевіряємо стабільність швидкості
        vx_std = np.std(vx_list)  # Стандартне відхилення
        vy_std = np.std(vy_list)

        if vx_std < 0.5 and vy_std < 0.5:
            # Швидкість майже не змінюється → прямолінійний рух
            return 'straight'

        # === АНАЛІЗ 2: ЗИГЗАГ ===
        # Перевіряємо чи змінюється знак vx (ліво-право-ліво)
        sign_changes = 0
        for i in range(1, len(vx_list)):
            if vx_list[i] * vx_list[i - 1] < 0:  # Різні знаки
                sign_changes += 1

        if sign_changes >= 3:  # 3+ зміни напрямку
            return 'zigzag'

        # === АНАЛІЗ 3: ПОВОРОТ ===
        # Рахуємо кути руху
        angles = []
        for vx, vy in velocities:
            angle = np.degrees(np.arctan2(vy, vx))
            angles.append(angle)

        # Перевіряємо чи кут плавно змінюється
        angle_changes = []
        for i in range(1, len(angles)):
            delta = angles[i] - angles[i - 1]
            # Нормалізація -180 до 180
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360
            angle_changes.append(delta)

        avg_angle_change = np.mean(angle_changes)

        if abs(avg_angle_change) > 5:  # Постійна зміна кута > 5
            return 'turning'

        # === АНАЛІЗ 4: ПРИСКОРЕННЯ ===
        # Рахуємо швидкості (модулі)
        speeds = [np.sqrt(vx ** 2 + vy ** 2) for vx, vy in velocities]

        # Перевіряємо чи швидкість зростає
        speed_trend = speeds[-1] - speeds[0]  # Різниця між останнім і першим

        if speed_trend > 2:  # Прискорився на 2+ px/кадр
            return 'accelerating'

        # === ЗА ЗАМОВЧУВАННЯМ ===
        return 'unknown'

    def _predict_with_uncertainty(self, track_id, current_pos, current_vel, pattern):
        """
        Передбачення з конусом невизначеності

        Returns:
            tuple: (predicted_pos, uncertainty_radius)
        """
        tx, ty = current_pos
        vx, vy = current_vel

        # Базова невизначеність
        base_uncertainty = 10  # пікселів

        # === ЧАС ДО ПЕРЕХОПЛЕННЯ ===
        dx = tx - self.uav_x
        dy = ty - self.uav_y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        threat_speed = np.sqrt(vx ** 2 + vy ** 2)

        if threat_speed < 0.1:
            time_to_threat = 50  # Статична загроза
        else:
            time_to_threat = distance / threat_speed if threat_speed > 0 else 50

        # Обмежуємо час (не передбачаємо занадто далеко)
        time_to_threat = min(time_to_threat, 100)

        # === ПЕРЕДБАЧЕННЯ ЗАЛЕЖНО ВІД ПАТЕРНУ ===

        if pattern == 'straight':
            # Прямолінійний рух - стандартне передбачення
            predicted_x = tx + vx * time_to_threat
            predicted_y = ty + vy * time_to_threat

            # Невизначеність зростає лінійно
            uncertainty = base_uncertainty + time_to_threat * 0.3

        elif pattern == 'zigzag':
            # Передбачаємо середню позицію
            predicted_x = tx + vx * time_to_threat * 0.7  # Менший коефіцієнт
            predicted_y = ty + vy * time_to_threat

            # Велика невизначеність по X (бо зигзагує)
            uncertainty = base_uncertainty + time_to_threat * 0.8

        elif pattern == 'turning':
            # Поворот - екстраполюємо дугу
            # Спрощено: додаємо невизначеність
            predicted_x = tx + vx * time_to_threat
            predicted_y = ty + vy * time_to_threat

            uncertainty = base_uncertainty + time_to_threat * 0.5

        elif pattern == 'accelerating':
            # Прискорення - рахуємо з прискоренням
            history = self.target_history.get(track_id, [])

            if len(history) >= 5:
                # Рахуємо прискорення
                recent_vels = [h['vel'] for h in history[-5:]]
                ax = (recent_vels[-1][0] - recent_vels[0][0]) / 4
                ay = (recent_vels[-1][1] - recent_vels[0][1]) / 4

                # Формула з прискоренням: s = v*t + 0.5*a*t²
                predicted_x = tx + vx * time_to_threat + 0.5 * ax * time_to_threat ** 2
                predicted_y = ty + vy * time_to_threat + 0.5 * ay * time_to_threat ** 2
            else:
                predicted_x = tx + vx * time_to_threat
                predicted_y = ty + vy * time_to_threat

            # Середня невизначеність
            uncertainty = base_uncertainty + time_to_threat * 0.4

        else:
            predicted_x = tx + vx * time_to_threat # Якщо патерн ворожого дрону невідомий, діємо з максимальною обережністю
            predicted_y = ty + vy * time_to_threat
            uncertainty = base_uncertainty + time_to_threat * 1.0

        uncertainty = min(uncertainty, 200)  # Максимум 200 пікселів

        return (predicted_x, predicted_y), uncertainty

    def _calculate_evasion(self, uav_pos, threat_pos, threat_vel, uncertainty):
        """
        Розрахунок точки ухилення з врахуванням невизначеності
        """
        ux, uy = uav_pos
        tx, ty = threat_pos

        # Вектор ВІД загрози
        dx = ux - tx
        dy = uy - ty

        # Відстань до ПЕРЕДБАЧЕНОЇ позиції
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Віднімаємо невизначеність
        effective_distance = distance - uncertainty

        if effective_distance < 100:
            threat_level = "CRITICAL"
        elif effective_distance < 200:
            threat_level = "HIGH"
        elif effective_distance < 300:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

        # Збільшуємо дистанцію ухилення при високій невизначеності
        evasion_distance = 200 + uncertainty * 0.5

        evade_length = np.sqrt(dx ** 2 + dy ** 2)
        if evade_length > 0:
            evade_dx = dx / evade_length
            evade_dy = dy / evade_length
        else:
            evade_dx = 1
            evade_dy = 0

        # Точка ухилення
        ex = ux + evade_dx * evasion_distance
        ey = uy + evade_dy * evasion_distance

        # Обмеження межами екрану
        margin = 50
        ex = max(margin, min(ex, self.frame_width - margin))
        ey = max(margin, min(ey, self.frame_height - margin))

        return ex, ey, distance, threat_level

    def _generate_evasion_commands(self, ex, ey, vx, vy, distance, pattern):
        """
        Генерація команд з врахуванням патерну
        """
        commands = []

        dx = ex - self.uav_x

        if abs(dx) > COMMAND_THRESHOLD:
            if dx > 0:
                commands.append("EVADE RIGHT")
            else:
                commands.append("EVADE LEFT")

        dy = ey - self.uav_y

        if abs(dy) > COMMAND_THRESHOLD:
            if dy > 0:
                commands.append("EVADE DOWN")
            else:
                commands.append("EVADE UP")

        if pattern == 'straight':
            pass

        elif pattern == 'zigzag':
            commands.append("EVASIVE ZIGZAG")

        elif pattern == 'turning':
            commands.append("PERPENDICULAR")

        elif pattern == 'accelerating':
            if distance < 300:
                commands.insert(0, "⚠️ ACCELERATING THREAT")

        threat_speed = np.sqrt(vx ** 2 + vy ** 2)

        if threat_speed > UAV_SPEED * 0.8:
            commands.append("FULL THROTTLE")


        if distance < 150:
            commands.insert(0, "⚠️ EVADE NOW!")

        if not commands:
            commands.append("SAFE DISTANCE")
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return self.last_commands
        self.last_command_time = current_time
        self.last_commands = commands
        return commands