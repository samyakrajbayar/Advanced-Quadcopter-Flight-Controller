"""
Advanced Quadcopter Flight Controller
Supports multiple flight modes, failsafe, GPS hold, altitude hold, and telemetry
"""

import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum
import json
from collections import deque

class FlightMode(Enum):
    """Available flight modes"""
    STABILIZE = "stabilize"
    ACRO = "acro"
    ALT_HOLD = "alt_hold"
    GPS_HOLD = "gps_hold"
    AUTO = "auto"
    RTL = "return_to_launch"
    FAILSAFE = "failsafe"

class ArmingState(Enum):
    """Arming states"""
    DISARMED = "disarmed"
    ARMED = "armed"
    ARMING = "arming"
    DISARMING = "disarming"

@dataclass
class PIDController:
    """Enhanced PID Controller with feedforward and filtering"""
    kp: float
    ki: float
    kd: float
    kff: float = 0.0  # Feedforward gain
    integral: float = 0.0
    prev_error: float = 0.0
    prev_derivative: float = 0.0
    integral_limit: float = 100.0
    output_limit: float = float('inf')
    lpf_alpha: float = 0.3  # Low-pass filter for derivative
    
    def update(self, error: float, dt: float, feedforward: float = 0.0) -> float:
        """Calculate PID output with feedforward and filtering"""
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term with low-pass filter
        if dt > 0:
            derivative = (error - self.prev_error) / dt
            filtered_derivative = self.lpf_alpha * derivative + (1 - self.lpf_alpha) * self.prev_derivative
            self.prev_derivative = filtered_derivative
            d_term = self.kd * filtered_derivative
        else:
            d_term = 0
        
        self.prev_error = error
        
        # Feedforward term
        ff_term = self.kff * feedforward
        
        # Total output with limiting
        output = p_term + i_term + d_term + ff_term
        return max(min(output, self.output_limit), -self.output_limit)
    
    def reset(self):
        """Reset PID state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

@dataclass
class SensorData:
    """Complete sensor package"""
    # Gyroscope (deg/s)
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    
    # Accelerometer (g)
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 1.0
    
    # Magnetometer (uT)
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 1.0
    
    # Barometer
    pressure: float = 101325.0  # Pa
    temperature: float = 25.0    # Celsius
    
    # GPS
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_alt: float = 0.0
    gps_fix: int = 0  # 0=no fix, 3=3D fix
    gps_satellites: int = 0
    
    # Battery
    battery_voltage: float = 12.6
    battery_current: float = 0.0
    battery_consumed: float = 0.0  # mAh
    
    # Rangefinder
    rangefinder_distance: float = -1.0  # meters, -1 = invalid

@dataclass
class Attitude:
    """Aircraft attitude"""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

@dataclass
class Position:
    """3D position"""
    x: float = 0.0  # meters
    y: float = 0.0
    z: float = 0.0  # altitude

@dataclass
class Velocity:
    """3D velocity"""
    vx: float = 0.0  # m/s
    vy: float = 0.0
    vz: float = 0.0

@dataclass
class ControlInput:
    """Pilot control inputs"""
    throttle: float = 0.0    # 0-100%
    roll: float = 50.0       # 0-100%, 50=center
    pitch: float = 50.0      # 0-100%, 50=center
    yaw: float = 50.0        # 0-100%, 50=center
    aux1: float = 0.0        # Mode switch
    aux2: float = 0.0        # Aux functions

@dataclass
class FlightParameters:
    """Configuration parameters"""
    max_angle: float = 45.0           # degrees
    max_rate: float = 200.0           # deg/s
    max_vertical_speed: float = 3.0   # m/s
    hover_throttle: float = 50.0      # %
    min_throttle: float = 5.0         # %
    max_throttle: float = 95.0        # %
    low_battery_voltage: float = 10.8 # V
    critical_battery_voltage: float = 10.2  # V
    rc_timeout: float = 1.0           # seconds
    gps_timeout: float = 2.0          # seconds

class ExtendedKalmanFilter:
    """EKF for state estimation"""
    def __init__(self):
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.state = np.zeros(9)
        self.P = np.eye(9) * 1.0  # Covariance matrix
        
        # Process noise
        self.Q = np.eye(9) * 0.01
        
        # Measurement noise
        self.R_gps = np.eye(3) * 5.0
        self.R_baro = 2.0
        self.R_accel = np.eye(3) * 0.5
    
    def predict(self, gyro: np.ndarray, dt: float):
        """Prediction step"""
        # Update attitude from gyroscope
        self.state[6] += gyro[0] * dt
        self.state[7] += gyro[1] * dt
        self.state[8] += gyro[2] * dt
        
        # Update position from velocity
        self.state[0] += self.state[3] * dt
        self.state[1] += self.state[4] * dt
        self.state[2] += self.state[5] * dt
        
        # Update covariance
        self.P += self.Q * dt
    
    def update_gps(self, gps_pos: np.ndarray):
        """Update with GPS measurement"""
        H = np.zeros((3, 9))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        
        y = gps_pos - self.state[:3]
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state += K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
    
    def update_barometer(self, altitude: float):
        """Update with barometer measurement"""
        H = np.zeros(9)
        H[2] = 1
        
        y = altitude - self.state[2]
        S = H @ self.P @ H.T + self.R_baro
        K = self.P @ H.T / S
        
        self.state += K * y
        self.P = (np.eye(9) - np.outer(K, H)) @ self.P

class MotorMixer:
    """Advanced motor mixer with thrust curve compensation"""
    def __init__(self, motor_count: int = 4):
        self.motor_count = motor_count
        self.motor_output_history = [deque(maxlen=10) for _ in range(motor_count)]
    
    @staticmethod
    def apply_thrust_curve(throttle: float) -> float:
        """Apply non-linear thrust curve"""
        # Quadratic curve for more linear thrust response
        normalized = throttle / 100.0
        return (normalized ** 1.8) * 100.0
    
    def mix(self, throttle: float, roll: float, pitch: float, yaw: float,
            compensation: bool = True) -> Tuple[float, float, float, float]:
        """Mix controls to motor outputs with optional thrust compensation"""
        
        if compensation:
            throttle = self.apply_thrust_curve(throttle)
        
        # X configuration mixing
        m1 = throttle - roll - pitch + yaw
        m2 = throttle + roll - pitch - yaw
        m3 = throttle + roll + pitch + yaw
        m4 = throttle - roll + pitch - yaw
        
        motors = [m1, m2, m3, m4]
        
        # Check if any motor would saturate
        max_motor = max(motors)
        min_motor = min(motors)
        
        # Reduce all motors proportionally if max exceeds limit
        if max_motor > 100:
            reduction = max_motor - 100
            motors = [m - reduction for m in motors]
        
        # Ensure minimum throttle
        if min_motor < 0:
            boost = -min_motor
            motors = [m + boost for m in motors]
        
        # Final limiting
        motors = [max(0, min(100, m)) for m in motors]
        
        # Store history
        for i, m in enumerate(motors):
            self.motor_output_history[i].append(m)
        
        return tuple(motors)
    
    def get_motor_health(self) -> List[float]:
        """Calculate motor health based on output variance"""
        health = []
        for history in self.motor_output_history:
            if len(history) > 5:
                variance = np.var(list(history))
                # Lower variance = more healthy (0-100 scale)
                health_score = max(0, 100 - variance * 5)
                health.append(health_score)
            else:
                health.append(100.0)
        return health

class FailsafeManager:
    """Manages failsafe conditions and recovery"""
    def __init__(self, params: FlightParameters):
        self.params = params
        self.rc_last_update = time.time()
        self.gps_last_update = time.time()
        self.failsafe_active = False
        self.home_position = Position()
        self.home_set = False
    
    def check_conditions(self, sensor: SensorData) -> Tuple[bool, str]:
        """Check all failsafe conditions"""
        current_time = time.time()
        reasons = []
        
        # Check RC timeout
        if current_time - self.rc_last_update > self.params.rc_timeout:
            reasons.append("RC signal lost")
        
        # Check battery voltage
        if sensor.battery_voltage < self.params.critical_battery_voltage:
            reasons.append("Critical battery voltage")
        elif sensor.battery_voltage < self.params.low_battery_voltage:
            reasons.append("Low battery warning")
        
        # Check GPS health for GPS-dependent modes
        if current_time - self.gps_last_update > self.params.gps_timeout:
            if sensor.gps_fix < 3:
                reasons.append("GPS signal degraded")
        
        if reasons:
            return True, "; ".join(reasons)
        return False, ""
    
    def update_rc_timestamp(self):
        """Update last RC signal time"""
        self.rc_last_update = time.time()
    
    def set_home(self, position: Position):
        """Set home position for RTL"""
        self.home_position = position
        self.home_set = True

class FlightController:
    """Advanced flight controller with multiple modes"""
    def __init__(self, params: Optional[FlightParameters] = None):
        self.params = params or FlightParameters()
        
        # Attitude PIDs (outer loop)
        self.roll_pid = PIDController(kp=2.5, ki=0.08, kd=0.4, output_limit=250)
        self.pitch_pid = PIDController(kp=2.5, ki=0.08, kd=0.4, output_limit=250)
        self.yaw_pid = PIDController(kp=3.0, ki=0.15, kd=0.5, output_limit=200)
        
        # Rate PIDs (inner loop)
        self.roll_rate_pid = PIDController(kp=0.9, ki=0.15, kd=0.06, output_limit=50)
        self.pitch_rate_pid = PIDController(kp=0.9, ki=0.15, kd=0.06, output_limit=50)
        self.yaw_rate_pid = PIDController(kp=1.2, ki=0.2, kd=0.02, output_limit=50)
        
        # Altitude PID
        self.alt_pid = PIDController(kp=2.0, ki=0.3, kd=1.0, output_limit=30)
        self.alt_rate_pid = PIDController(kp=1.5, ki=0.5, kd=0.3, output_limit=30)
        
        # Position PIDs
        self.pos_x_pid = PIDController(kp=1.0, ki=0.1, kd=0.5, output_limit=15)
        self.pos_y_pid = PIDController(kp=1.0, ki=0.1, kd=0.5, output_limit=15)
        
        # Extended Kalman Filter
        self.ekf = ExtendedKalmanFilter()
        
        # Motor mixer
        self.mixer = MotorMixer()
        
        # Failsafe manager
        self.failsafe = FailsafeManager(self.params)
        
        # State
        self.mode = FlightMode.STABILIZE
        self.arming_state = ArmingState.DISARMED
        self.attitude = Attitude()
        self.position = Position()
        self.velocity = Velocity()
        self.last_update = time.time()
        self.flight_time = 0.0
        
        # Telemetry logging
        self.telemetry_log = deque(maxlen=1000)
        
    def update(self, sensor: SensorData, control: ControlInput) -> Tuple[float, float, float, float]:
        """Main control loop"""
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Prevent dt spikes
        dt = min(dt, 0.1)
        if dt < 0.001:
            dt = 0.02
        
        if self.arming_state == ArmingState.ARMED:
            self.flight_time += dt
        
        # Update state estimation
        self._update_state_estimation(sensor, dt)
        
        # Check failsafe conditions
        failsafe_triggered, reason = self.failsafe.check_conditions(sensor)
        if failsafe_triggered and self.arming_state == ArmingState.ARMED:
            self.mode = FlightMode.FAILSAFE
            print(f"FAILSAFE ACTIVATED: {reason}")
        
        # Update RC timestamp
        self.failsafe.update_rc_timestamp()
        
        # Process flight mode
        motors = self._process_flight_mode(sensor, control, dt)
        
        # Log telemetry
        self._log_telemetry(sensor, control, motors)
        
        return motors
    
    def _update_state_estimation(self, sensor: SensorData, dt: float):
        """Update EKF and attitude estimation"""
        # EKF prediction
        gyro = np.array([sensor.gyro_x, sensor.gyro_y, sensor.gyro_z])
        self.ekf.predict(gyro, dt)
        
        # Update with GPS if available
        if sensor.gps_fix >= 3 and sensor.gps_satellites >= 6:
            gps_pos = np.array([sensor.gps_lat, sensor.gps_lon, sensor.gps_alt])
            self.ekf.update_gps(gps_pos)
        
        # Update with barometer
        altitude = self._pressure_to_altitude(sensor.pressure)
        self.ekf.update_barometer(altitude)
        
        # Extract state
        self.position.x = self.ekf.state[0]
        self.position.y = self.ekf.state[1]
        self.position.z = self.ekf.state[2]
        self.velocity.vx = self.ekf.state[3]
        self.velocity.vy = self.ekf.state[4]
        self.velocity.vz = self.ekf.state[5]
        self.attitude.roll = self.ekf.state[6]
        self.attitude.pitch = self.ekf.state[7]
        self.attitude.yaw = self.ekf.state[8]
        
        # Normalize angles
        self.attitude.yaw = self._normalize_angle(self.attitude.yaw)
    
    def _process_flight_mode(self, sensor: SensorData, control: ControlInput, 
                            dt: float) -> Tuple[float, float, float, float]:
        """Process current flight mode"""
        if self.arming_state != ArmingState.ARMED:
            return (0, 0, 0, 0)
        
        if self.mode == FlightMode.STABILIZE:
            return self._stabilize_mode(sensor, control, dt)
        elif self.mode == FlightMode.ACRO:
            return self._acro_mode(sensor, control, dt)
        elif self.mode == FlightMode.ALT_HOLD:
            return self._alt_hold_mode(sensor, control, dt)
        elif self.mode == FlightMode.GPS_HOLD:
            return self._gps_hold_mode(sensor, control, dt)
        elif self.mode == FlightMode.RTL:
            return self._rtl_mode(sensor, control, dt)
        elif self.mode == FlightMode.FAILSAFE:
            return self._failsafe_mode(sensor, control, dt)
        else:
            return (0, 0, 0, 0)
    
    def _stabilize_mode(self, sensor: SensorData, control: ControlInput, 
                       dt: float) -> Tuple[float, float, float, float]:
        """Stabilize mode - self-leveling"""
        # Desired angles from stick input
        desired_roll = (control.roll - 50) * self.params.max_angle / 50
        desired_pitch = (control.pitch - 50) * self.params.max_angle / 50
        desired_yaw_rate = (control.yaw - 50) * self.params.max_rate / 50
        
        # Outer loop (angle)
        roll_error = desired_roll - self.attitude.roll
        pitch_error = desired_pitch - self.attitude.pitch
        
        desired_roll_rate = self.roll_pid.update(roll_error, dt)
        desired_pitch_rate = self.pitch_pid.update(pitch_error, dt)
        
        # Inner loop (rate)
        roll_rate_error = desired_roll_rate - sensor.gyro_x
        pitch_rate_error = desired_pitch_rate - sensor.gyro_y
        yaw_rate_error = desired_yaw_rate - sensor.gyro_z
        
        roll_out = self.roll_rate_pid.update(roll_rate_error, dt)
        pitch_out = self.pitch_rate_pid.update(pitch_rate_error, dt)
        yaw_out = self.yaw_rate_pid.update(yaw_rate_error, dt)
        
        return self.mixer.mix(control.throttle, roll_out, pitch_out, yaw_out)
    
    def _acro_mode(self, sensor: SensorData, control: ControlInput, 
                   dt: float) -> Tuple[float, float, float, float]:
        """Acro mode - rate control only"""
        desired_roll_rate = (control.roll - 50) * self.params.max_rate / 50
        desired_pitch_rate = (control.pitch - 50) * self.params.max_rate / 50
        desired_yaw_rate = (control.yaw - 50) * self.params.max_rate / 50
        
        roll_out = self.roll_rate_pid.update(desired_roll_rate - sensor.gyro_x, dt)
        pitch_out = self.pitch_rate_pid.update(desired_pitch_rate - sensor.gyro_y, dt)
        yaw_out = self.yaw_rate_pid.update(desired_yaw_rate - sensor.gyro_z, dt)
        
        return self.mixer.mix(control.throttle, roll_out, pitch_out, yaw_out)
    
    def _alt_hold_mode(self, sensor: SensorData, control: ControlInput, 
                      dt: float) -> Tuple[float, float, float, float]:
        """Altitude hold mode"""
        # Desired altitude change from throttle stick
        if abs(control.throttle - 50) < 5:  # Deadband
            desired_alt_rate = 0
        else:
            desired_alt_rate = (control.throttle - 50) * self.params.max_vertical_speed / 50
        
        # Altitude rate control
        alt_rate_error = desired_alt_rate - self.velocity.vz
        throttle_adjustment = self.alt_rate_pid.update(alt_rate_error, dt)
        
        adjusted_throttle = self.params.hover_throttle + throttle_adjustment
        adjusted_throttle = max(self.params.min_throttle, 
                               min(self.params.max_throttle, adjusted_throttle))
        
        # Attitude control (same as stabilize)
        desired_roll = (control.roll - 50) * self.params.max_angle / 50
        desired_pitch = (control.pitch - 50) * self.params.max_angle / 50
        desired_yaw_rate = (control.yaw - 50) * self.params.max_rate / 50
        
        roll_out = self.roll_rate_pid.update(
            self.roll_pid.update(desired_roll - self.attitude.roll, dt) - sensor.gyro_x, dt)
        pitch_out = self.pitch_rate_pid.update(
            self.pitch_pid.update(desired_pitch - self.attitude.pitch, dt) - sensor.gyro_y, dt)
        yaw_out = self.yaw_rate_pid.update(desired_yaw_rate - sensor.gyro_z, dt)
        
        return self.mixer.mix(adjusted_throttle, roll_out, pitch_out, yaw_out)
    
    def _gps_hold_mode(self, sensor: SensorData, control: ControlInput, 
                      dt: float) -> Tuple[float, float, float, float]:
        """GPS position hold mode"""
        # Position control
        pos_x_error = 0 - self.velocity.vx  # Hold current position
        pos_y_error = 0 - self.velocity.vy
        
        desired_roll = self.pos_y_pid.update(pos_y_error, dt)
        desired_pitch = -self.pos_x_pid.update(pos_x_error, dt)
        
        # Limit angles
        desired_roll = max(-self.params.max_angle, min(self.params.max_angle, desired_roll))
        desired_pitch = max(-self.params.max_angle, min(self.params.max_angle, desired_pitch))
        
        # Use altitude hold for vertical
        if abs(control.throttle - 50) < 5:
            desired_alt_rate = 0
        else:
            desired_alt_rate = (control.throttle - 50) * self.params.max_vertical_speed / 50
        
        throttle_adjustment = self.alt_rate_pid.update(desired_alt_rate - self.velocity.vz, dt)
        adjusted_throttle = self.params.hover_throttle + throttle_adjustment
        
        # Yaw control from stick
        desired_yaw_rate = (control.yaw - 50) * self.params.max_rate / 50
        
        # Rate control
        roll_out = self.roll_rate_pid.update(
            self.roll_pid.update(desired_roll - self.attitude.roll, dt) - sensor.gyro_x, dt)
        pitch_out = self.pitch_rate_pid.update(
            self.pitch_pid.update(desired_pitch - self.attitude.pitch, dt) - sensor.gyro_y, dt)
        yaw_out = self.yaw_rate_pid.update(desired_yaw_rate - sensor.gyro_z, dt)
        
        return self.mixer.mix(adjusted_throttle, roll_out, pitch_out, yaw_out)
    
    def _rtl_mode(self, sensor: SensorData, control: ControlInput, 
                  dt: float) -> Tuple[float, float, float, float]:
        """Return to launch mode"""
        if not self.failsafe.home_set:
            # If no home position, just land
            return self._failsafe_mode(sensor, control, dt)
        
        # Navigate to home (simplified)
        # In real implementation, use waypoint navigation
        return self._gps_hold_mode(sensor, control, dt)
    
    def _failsafe_mode(self, sensor: SensorData, control: ControlInput, 
                      dt: float) -> Tuple[float, float, float, float]:
        """Failsafe mode - controlled descent"""
        # Descend at safe rate while maintaining level attitude
        desired_descent_rate = -1.0  # m/s
        
        throttle_adjustment = self.alt_rate_pid.update(
            desired_descent_rate - self.velocity.vz, dt)
        adjusted_throttle = self.params.hover_throttle + throttle_adjustment
        
        # Level attitude
        roll_out = self.roll_rate_pid.update(
            self.roll_pid.update(-self.attitude.roll, dt) - sensor.gyro_x, dt)
        pitch_out = self.pitch_rate_pid.update(
            self.pitch_pid.update(-self.attitude.pitch, dt) - sensor.gyro_y, dt)
        yaw_out = 0  # No yaw correction
        
        return self.mixer.mix(adjusted_throttle, roll_out, pitch_out, yaw_out)
    
    def arm(self) -> bool:
        """Arm the flight controller"""
        if self.arming_state != ArmingState.DISARMED:
            return False
        
        self.arming_state = ArmingState.ARMED
        self._reset_all_pids()
        self.flight_time = 0.0
        
        # Set home position
        self.failsafe.set_home(Position(self.position.x, self.position.y, self.position.z))
        
        print("Flight controller ARMED")
        return True
    
    def disarm(self):
        """Disarm the flight controller"""
        self.arming_state = ArmingState.DISARMED
        print("Flight controller DISARMED")
    
    def set_mode(self, mode: FlightMode) -> bool:
        """Change flight mode"""
        # Mode change validation
        if self.arming_state != ArmingState.ARMED:
            if mode not in [FlightMode.STABILIZE, FlightMode.ACRO]:
                print(f"Cannot switch to {mode.value} while disarmed")
                return False
        
        self.mode = mode
        self._reset_all_pids()
        print(f"Flight mode changed to: {mode.value}")
        return True
    
    def _reset_all_pids(self):
        """Reset all PID controllers"""
        for pid in [self.roll_pid, self.pitch_pid, self.yaw_pid,
                    self.roll_rate_pid, self.pitch_rate_pid, self.yaw_rate_pid,
                    self.alt_pid, self.alt_rate_pid, self.pos_x_pid, self.pos_y_pid]:
            pid.reset()
    
    def _pressure_to_altitude(self, pressure: float) -> float:
        """Convert pressure to altitude"""
        # Barometric formula
        P0 = 101325.0  # Sea level pressure
        return 44330.0 * (1.0 - (pressure / P0) ** 0.1903)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to -180 to 180"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def _log_telemetry(self, sensor: SensorData, control: ControlInput, motors: tuple):
        """Log telemetry data"""
        telem = {
            'timestamp': time.time(),
            'mode': self.mode.value,
            'armed': self.arming_state == ArmingState.ARMED,
            'attitude': {'roll': self.attitude.roll, 'pitch': self.attitude.pitch, 'yaw': self.attitude.yaw},
            'position': {'x': self.position.x, 'y': self.position.y, 'z': self.position.z},
            'velocity': {'vx': self.velocity.vx, 'vy': self.velocity.vy, 'vz': self.velocity.vz},
            'motors': motors,
            'battery': sensor.battery_voltage,
            'flight_time': self.flight_time
        }
        self.telemetry_log.append(telem)
    
    def get_telemetry(self) -> dict:
        """Get current telemetry"""
        motor_health = self.mixer.get_motor_health()
        return {
            'mode': self.mode.value,
            'armed': self.arming_state.value,
            'flight_time': round(self.flight_time, 1),
            'attitude': {
                'roll': round(self.attitude.roll, 2),
                'pitch': round(self.attitude.pitch, 2),
                'yaw': round(self.attitude.yaw, 2)
            },
            'position': {
                'x': round(self.position.x, 2),
                'y': round(self.position.y, 2),
                'altitude': round(self.position.z, 2)
            },
            'velocity': {
                'vx': round(self.velocity.vx, 2),
                'vy': round(self.velocity.vy, 2),
                'vz': round(self.velocity.vz, 2)
            },
            'motor_health': [round(h, 1) for h in motor_health]
        }
    
    def save_telemetry_log(self, filename: str):
        """Save telemetry log to file"""
        with open(filename, 'w') as f:
            json.dump(list(self.telemetry_log), f, indent=2)
        print(f"Telemetry log saved to {filename}")
    
    def load_parameters(self, filename: str):
        """Load flight parameters from file"""
        with open(filename, 'r') as f:
            params_dict = json.load(f)
            for key, value in params_dict.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
        print(f"Parameters loaded from {filename}")


# Example usage and advanced simulation
if __name__ == "__main__":
    print("=" * 70)
    print("Advanced Quadcopter Flight Controller Simulation")
    print("=" * 70)
    
    # Create flight controller with custom parameters
    params = FlightParameters(
        max_angle=35.0,
        max_rate=180.0,
        hover_throttle=48.0,
        low_battery_voltage=11.1,
        critical_battery_voltage=10.5
    )
    
    fc = FlightController(params)
    
    # Test different flight modes
    print("\n[TEST 1] STABILIZE MODE")
    print("-" * 70)
    fc.set_mode(FlightMode.STABILIZE)
    fc.arm()
    
    # Simulation parameters
    dt = 0.01  # 100Hz
    
    # Run stabilize mode test
    print("\nTime(s)  Mode       Roll(°)  Pitch(°)  Yaw(°)   Alt(m)  M1   M2   M3   M4")
    print("-" * 70)
    
    for i in range(300):  # 3 seconds
        t = i * dt
        
        # Simulate sensor data with realistic noise
        sensor = SensorData(
            gyro_x=np.random.normal(0, 0.8),
            gyro_y=np.random.normal(0, 0.8),
            gyro_z=np.random.normal(0, 0.3),
            accel_x=np.random.normal(0, 0.02),
            accel_y=np.random.normal(0, 0.02),
            accel_z=np.random.normal(1.0, 0.02),
            pressure=101325.0 - (i * 0.5),  # Slowly ascending
            battery_voltage=12.4 - (i * 0.0001),
            battery_current=15.5,
            gps_fix=3,
            gps_satellites=10,
            gps_lat=25.276987,
            gps_lon=55.296249,
            gps_alt=10.0 + (i * 0.01)
        )
        
        # Control input - hover with slight roll
        control = ControlInput(
            throttle=50,
            roll=52 if t > 1.0 else 50,  # Small roll after 1s
            pitch=50,
            yaw=50
        )
        
        motors = fc.update(sensor, control)
        
        # Print every 0.5s
        if i % 50 == 0:
            telem = fc.get_telemetry()
            print(f"{t:5.2f}   {telem['mode']:10s} {telem['attitude']['roll']:6.2f}  "
                  f"{telem['attitude']['pitch']:7.2f}  {telem['attitude']['yaw']:6.2f}  "
                  f"{telem['position']['altitude']:6.2f} {motors[0]:4.1f} {motors[1]:4.1f} "
                  f"{motors[2]:4.1f} {motors[3]:4.1f}")
        
        time.sleep(dt)
    
    # Test altitude hold mode
    print("\n[TEST 2] ALTITUDE HOLD MODE")
    print("-" * 70)
    fc.set_mode(FlightMode.ALT_HOLD)
    
    current_alt = fc.position.z
    
    for i in range(200):  # 2 seconds
        t = i * dt
        
        sensor = SensorData(
            gyro_x=np.random.normal(0, 0.8),
            gyro_y=np.random.normal(0, 0.8),
            gyro_z=np.random.normal(0, 0.3),
            accel_x=np.random.normal(0, 0.02),
            accel_y=np.random.normal(0, 0.02),
            accel_z=np.random.normal(1.0, 0.02),
            pressure=101325.0 - (current_alt * 12.0),
            battery_voltage=12.3,
            battery_current=16.2,
            gps_fix=3,
            gps_satellites=10
        )
        
        # Climb command
        control = ControlInput(
            throttle=60 if t < 1.0 else 50,  # Climb then hold
            roll=50,
            pitch=50,
            yaw=50
        )
        
        motors = fc.update(sensor, control)
        
        if i % 50 == 0:
            telem = fc.get_telemetry()
            print(f"{t:5.2f}   {telem['mode']:10s} {telem['attitude']['roll']:6.2f}  "
                  f"{telem['attitude']['pitch']:7.2f}  {telem['attitude']['yaw']:6.2f}  "
                  f"{telem['position']['altitude']:6.2f} {motors[0]:4.1f} {motors[1]:4.1f} "
                  f"{motors[2]:4.1f} {motors[3]:4.1f}")
        
        time.sleep(dt)
    
    # Test GPS hold mode
    print("\n[TEST 3] GPS HOLD MODE")
    print("-" * 70)
    fc.set_mode(FlightMode.GPS_HOLD)
    
    for i in range(200):  # 2 seconds
        t = i * dt
        
        # Add simulated wind disturbance
        wind_x = 0.5 * math.sin(t * 2)
        wind_y = 0.3 * math.cos(t * 1.5)
        
        sensor = SensorData(
            gyro_x=wind_x * 5 + np.random.normal(0, 0.8),
            gyro_y=wind_y * 5 + np.random.normal(0, 0.8),
            gyro_z=np.random.normal(0, 0.3),
            accel_x=wind_x + np.random.normal(0, 0.02),
            accel_y=wind_y + np.random.normal(0, 0.02),
            accel_z=np.random.normal(1.0, 0.02),
            pressure=101325.0 - (15.0 * 12.0),
            battery_voltage=12.2,
            battery_current=17.1,
            gps_fix=3,
            gps_satellites=12,
            gps_lat=25.276987 + (wind_x * 0.00001),
            gps_lon=55.296249 + (wind_y * 0.00001),
            gps_alt=15.0
        )
        
        control = ControlInput(
            throttle=50,
            roll=50,
            pitch=50,
            yaw=50
        )
        
        motors = fc.update(sensor, control)
        
        if i % 50 == 0:
            telem = fc.get_telemetry()
            print(f"{t:5.2f}   {telem['mode']:10s} {telem['attitude']['roll']:6.2f}  "
                  f"{telem['attitude']['pitch']:7.2f}  {telem['attitude']['yaw']:6.2f}  "
                  f"{telem['position']['altitude']:6.2f} {motors[0]:4.1f} {motors[1]:4.1f} "
                  f"{motors[2]:4.1f} {motors[3]:4.1f}")
        
        time.sleep(dt)
    
    # Test failsafe
    print("\n[TEST 4] FAILSAFE MODE (Low Battery)")
    print("-" * 70)
    
    for i in range(150):  # 1.5 seconds
        t = i * dt
        
        sensor = SensorData(
            gyro_x=np.random.normal(0, 0.8),
            gyro_y=np.random.normal(0, 0.8),
            gyro_z=np.random.normal(0, 0.3),
            accel_x=np.random.normal(0, 0.02),
            accel_y=np.random.normal(0, 0.02),
            accel_z=np.random.normal(1.0, 0.02),
            pressure=101325.0 - (15.0 * 12.0) + (i * 1.0),  # Descending
            battery_voltage=10.1,  # Critical voltage
            battery_current=18.5,
            gps_fix=3,
            gps_satellites=12
        )
        
        control = ControlInput(throttle=50, roll=50, pitch=50, yaw=50)
        motors = fc.update(sensor, control)
        
        if i % 50 == 0:
            telem = fc.get_telemetry()
            print(f"{t:5.2f}   {telem['mode']:10s} {telem['attitude']['roll']:6.2f}  "
                  f"{telem['attitude']['pitch']:7.2f}  {telem['attitude']['yaw']:6.2f}  "
                  f"{telem['position']['altitude']:6.2f} {motors[0]:4.1f} {motors[1]:4.1f} "
                  f"{motors[2]:4.1f} {motors[3]:4.1f}")
        
        time.sleep(dt)
    
    # Disarm and show summary
    fc.disarm()
    
    print("\n" + "=" * 70)
    print("FLIGHT SUMMARY")
    print("=" * 70)
    telem = fc.get_telemetry()
    print(f"Total Flight Time: {telem['flight_time']}s")
    print(f"Final Mode: {telem['mode']}")
    print(f"Final Attitude - Roll: {telem['attitude']['roll']}° | "
          f"Pitch: {telem['attitude']['pitch']}° | Yaw: {telem['attitude']['yaw']}°")
    print(f"Final Altitude: {telem['position']['altitude']}m")
    print(f"Motor Health: M1={telem['motor_health'][0]}% | M2={telem['motor_health'][1]}% | "
          f"M3={telem['motor_health'][2]}% | M4={telem['motor_health'][3]}%")
    
    # Save telemetry
    fc.save_telemetry_log("flight_log.json")
    
    print("\n✓ Simulation complete!")