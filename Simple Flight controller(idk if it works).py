import time
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PIDController:
    """PID Controller for stabilization"""
    kp: float
    ki: float
    kd: float
    integral: float = 0.0
    prev_error: float = 0.0
    integral_limit: float = 100.0
    
    def update(self, error: float, dt: float) -> float:
        """Calculate PID output"""
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        return p_term + i_term + d_term
    
    def reset(self):
        """Reset PID state"""
        self.integral = 0.0
        self.prev_error = 0.0

@dataclass
class SensorData:
    """Gyroscope and accelerometer data"""
    gyro_x: float = 0.0  # Roll rate (deg/s)
    gyro_y: float = 0.0  # Pitch rate (deg/s)
    gyro_z: float = 0.0  # Yaw rate (deg/s)
    accel_x: float = 0.0  # X acceleration (g)
    accel_y: float = 0.0  # Y acceleration (g)
    accel_z: float = 1.0  # Z acceleration (g)

@dataclass
class Attitude:
    """Current attitude angles"""
    roll: float = 0.0   # degrees
    pitch: float = 0.0  # degrees
    yaw: float = 0.0    # degrees

@dataclass
class ControlInput:
    """Pilot control inputs (0-100%)"""
    throttle: float = 0.0
    roll: float = 50.0     # 50 = center
    pitch: float = 50.0    # 50 = center
    yaw: float = 50.0      # 50 = center

class ComplementaryFilter:
    """Sensor fusion using complementary filter"""
    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha  # Gyro weight (0-1)
        self.attitude = Attitude()
    
    def update(self, sensor: SensorData, dt: float) -> Attitude:
        """Fuse gyro and accelerometer data"""
        # Integrate gyroscope data
        gyro_roll = self.attitude.roll + sensor.gyro_x * dt
        gyro_pitch = self.attitude.pitch + sensor.gyro_y * dt
        gyro_yaw = self.attitude.yaw + sensor.gyro_z * dt
        
        # Calculate angles from accelerometer
        accel_roll = math.degrees(math.atan2(sensor.accel_y, sensor.accel_z))
        accel_pitch = math.degrees(math.atan2(-sensor.accel_x, 
                                              math.sqrt(sensor.accel_y**2 + sensor.accel_z**2)))
        
        # Complementary filter
        self.attitude.roll = self.alpha * gyro_roll + (1 - self.alpha) * accel_roll
        self.attitude.pitch = self.alpha * gyro_pitch + (1 - self.alpha) * accel_pitch
        self.attitude.yaw = gyro_yaw  # Yaw from gyro only (no magnetometer)
        
        # Normalize yaw to -180 to 180
        while self.attitude.yaw > 180:
            self.attitude.yaw -= 360
        while self.attitude.yaw < -180:
            self.attitude.yaw += 360
            
        return self.attitude

class MotorMixer:
    """Convert control outputs to individual motor commands"""
    @staticmethod
    def mix(throttle: float, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """
        Mix control inputs to motor outputs
        Quadcopter layout (X configuration):
              Front
          M1      M2
            \  /
            /  \
          M4      M3
              Back
        M1: Front-left  (CW)
        M2: Front-right (CCW)
        M3: Back-right  (CW)
        M4: Back-left   (CCW)
        """
        # Calculate individual motor values
        m1 = throttle - roll - pitch + yaw  # Front-left
        m2 = throttle + roll - pitch - yaw  # Front-right
        m3 = throttle + roll + pitch + yaw  # Back-right
        m4 = throttle - roll + pitch - yaw  # Back-left
        
        # Constrain motor outputs to 0-100%
        motors = [max(0, min(100, m)) for m in [m1, m2, m3, m4]]
        
        return tuple(motors)

class FlightController:
    """Main flight controller class"""
    def __init__(self):
        # PID controllers for each axis
        self.roll_pid = PIDController(kp=1.5, ki=0.05, kd=0.3)
        self.pitch_pid = PIDController(kp=1.5, ki=0.05, kd=0.3)
        self.yaw_pid = PIDController(kp=2.0, ki=0.1, kd=0.5)
        
        # Rate PIDs (inner loop)
        self.roll_rate_pid = PIDController(kp=0.8, ki=0.1, kd=0.05)
        self.pitch_rate_pid = PIDController(kp=0.8, ki=0.1, kd=0.05)
        self.yaw_rate_pid = PIDController(kp=1.0, ki=0.15, kd=0.0)
        
        # Sensor fusion
        self.filter = ComplementaryFilter(alpha=0.98)
        
        # State
        self.attitude = Attitude()
        self.armed = False
        self.last_update = time.time()
        
        # Limits
        self.max_angle = 45.0  # degrees
        self.max_rate = 200.0  # deg/s
    
    def update(self, sensor: SensorData, control: ControlInput) -> Tuple[float, float, float, float]:
        """
        Main control loop update
        Returns: (motor1, motor2, motor3, motor4) values (0-100%)
        """
        # Calculate dt
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Prevent dt spikes
        if dt > 0.1:
            dt = 0.02  # Default to 50Hz
        
        # Update attitude estimate
        self.attitude = self.filter.update(sensor, dt)
        
        # Convert control inputs to desired angles/rates
        desired_roll = (control.roll - 50) * self.max_angle / 50  # -45 to +45 degrees
        desired_pitch = (control.pitch - 50) * self.max_angle / 50
        desired_yaw_rate = (control.yaw - 50) * self.max_rate / 50  # deg/s
        
        # Outer loop: Angle stabilization
        roll_error = desired_roll - self.attitude.roll
        pitch_error = desired_pitch - self.attitude.pitch
        
        desired_roll_rate = self.roll_pid.update(roll_error, dt)
        desired_pitch_rate = self.pitch_pid.update(pitch_error, dt)
        
        # Inner loop: Rate stabilization
        roll_rate_error = desired_roll_rate - sensor.gyro_x
        pitch_rate_error = desired_pitch_rate - sensor.gyro_y
        yaw_rate_error = desired_yaw_rate - sensor.gyro_z
        
        roll_output = self.roll_rate_pid.update(roll_rate_error, dt)
        pitch_output = self.pitch_rate_pid.update(pitch_rate_error, dt)
        yaw_output = self.yaw_rate_pid.update(yaw_rate_error, dt)
        
        # Constrain outputs
        roll_output = max(-50, min(50, roll_output))
        pitch_output = max(-50, min(50, pitch_output))
        yaw_output = max(-50, min(50, yaw_output))
        
        # Mix to motors
        if self.armed and control.throttle > 5:
            motors = MotorMixer.mix(control.throttle, roll_output, pitch_output, yaw_output)
        else:
            motors = (0, 0, 0, 0)
        
        return motors
    
    def arm(self):
        """Arm the flight controller"""
        self.armed = True
        # Reset PIDs
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.roll_rate_pid.reset()
        self.pitch_rate_pid.reset()
        self.yaw_rate_pid.reset()
        print("Flight controller ARMED")
    
    def disarm(self):
        """Disarm the flight controller"""
        self.armed = False
        print("Flight controller DISARMED")
    
    def get_telemetry(self) -> dict:
        """Get current telemetry data"""
        return {
            'armed': self.armed,
            'roll': round(self.attitude.roll, 2),
            'pitch': round(self.attitude.pitch, 2),
            'yaw': round(self.attitude.yaw, 2)
        }


# Example usage and simulation
if __name__ == "__main__":
    print("Quadcopter Flight Controller Simulation")
    print("=" * 50)
    
    # Create flight controller
    fc = FlightController()
    
    # Arm the controller
    fc.arm()
    
    # Simulation parameters
    sim_time = 5.0  # seconds
    dt = 0.02  # 50Hz update rate
    steps = int(sim_time / dt)
    
    # Initial conditions
    control = ControlInput(throttle=40, roll=50, pitch=50, yaw=50)
    
    print(f"\nRunning simulation for {sim_time}s at {1/dt}Hz")
    print("\nTime(s)  Roll(°)  Pitch(°)  Yaw(°)   M1   M2   M3   M4")
    print("-" * 60)
    
    for i in range(steps):
        t = i * dt
        
        # Simulate sensor data (with some noise and disturbance)
        sensor = SensorData(
            gyro_x=np.random.normal(0, 0.5),
            gyro_y=np.random.normal(0, 0.5),
            gyro_z=np.random.normal(0, 0.2),
            accel_x=np.random.normal(0, 0.01),
            accel_y=np.random.normal(0, 0.01),
            accel_z=np.random.normal(1.0, 0.01)
        )
        
        # Add control input change at 2 seconds (roll right)
        if t > 2.0:
            control.roll = 55  # Roll 5 degrees right
        
        # Update flight controller
        motors = fc.update(sensor, control)
        
        # Print telemetry every 0.5s
        if i % int(0.5 / dt) == 0:
            telem = fc.get_telemetry()
            print(f"{t:5.2f}   {telem['roll']:6.2f}  {telem['pitch']:7.2f}  "
                  f"{telem['yaw']:6.2f}  {motors[0]:4.1f} {motors[1]:4.1f} "
                  f"{motors[2]:4.1f} {motors[3]:4.1f}")
        
        time.sleep(dt)
    
    # Disarm
    fc.disarm()
    print("\nSimulation complete!")
