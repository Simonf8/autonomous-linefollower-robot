# Part 2 of improved test.py - PID Controller, Communication, and Main App

# -----------------------------------------------------------------------------
# --- ADAPTIVE PID CONTROLLER ---
# -----------------------------------------------------------------------------

class AdaptivePIDController:
    """Enhanced PID controller with adaptive parameters and advanced filtering"""
    
    def __init__(self, config):
        self.config = config
        self.kp = config.pid_kp
        self.ki = config.pid_ki
        self.kd = config.pid_kd
        self.integral_max = config.pid_integral_max
        
        # State variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
        # Enhanced filtering
        self.error_history = deque(maxlen=15)
        self.derivative_history = deque(maxlen=5)
        self.output_history = deque(maxlen=10)
        
        # Adaptive parameters
        self.performance_metric = deque(maxlen=50)
        self.last_adaptation_time = time.time()
        
        # Anti-windup
        self.integral_active = True
        
        # Derivative filtering (low-pass filter)
        self.derivative_filter_alpha = 0.1
        self.filtered_derivative = 0.0
    
    def calculate(self, error: float, dt: Optional[float] = None) -> float:
        """Calculate PID output with adaptive parameters"""
        current_time = time.time()
        dt = max(current_time - self.last_time if dt is None else dt, 1e-3)
        
        # Store error for analysis
        self.error_history.append(error)
        
        # Adaptive parameter adjustment
        if self.config.enable_adaptive_pid:
            self._adapt_parameters()
        
        # Proportional term with gain scheduling
        proportional = self.kp * error
        
        # Integral term with conditional integration
        if self.integral_active and abs(error) < 0.5:  # Only integrate for small errors
            self.integral += error * dt
            self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        elif abs(error) > 0.7:  # Reset integral for large errors
            self.integral *= 0.8
        
        integral_term = self.ki * self.integral
        
        # Derivative term with advanced filtering
        derivative = (error - self.prev_error) / dt
        
        # Apply low-pass filter to derivative
        self.filtered_derivative = (self.derivative_filter_alpha * derivative + 
                                  (1 - self.derivative_filter_alpha) * self.filtered_derivative)
        
        self.derivative_history.append(self.filtered_derivative)
        
        # Use median of recent derivatives for robustness
        if len(self.derivative_history) >= 3:
            derivative_term = self.kd * np.median(list(self.derivative_history)[-3:])
        else:
            derivative_term = self.kd * self.filtered_derivative
        
        # Calculate total output
        output = proportional + integral_term + derivative_term
        
        # Output limiting and rate limiting
        if self.output_history:
            max_change = self.config.max_steering_rate * dt
            output = np.clip(output, 
                           self.output_history[-1] - max_change,
                           self.output_history[-1] + max_change)
        
        output = np.clip(output, -1.0, 1.0)
        
        # Store for next iteration
        self.output_history.append(output)
        self.prev_error = error
        self.last_time = current_time
        
        # Update performance metric
        self._update_performance_metric(error, output)
        
        return output
    
    def _adapt_parameters(self):
        """Adapt PID parameters based on performance"""
        if time.time() - self.last_adaptation_time < 2.0:  # Adapt every 2 seconds
            return
            
        if len(self.performance_metric) < 20:
            return
        
        # Calculate recent performance
        recent_performance = np.mean(list(self.performance_metric)[-10:])
        older_performance = np.mean(list(self.performance_metric)[-20:-10])
        
        # Adapt Kp based on performance trend
        if recent_performance > older_performance * 1.1:  # Performance getting worse
            self.kp = max(self.config.pid_kp_range[0], 
                         self.kp - self.config.pid_adaptation_rate)
        elif recent_performance < older_performance * 0.9:  # Performance improving
            self.kp = min(self.config.pid_kp_range[1], 
                         self.kp + self.config.pid_adaptation_rate)
        
        self.last_adaptation_time = time.time()
        logging.debug(f"🎛️ Adapted PID: Kp={self.kp:.3f}")
    
    def _update_performance_metric(self, error: float, output: float):
        """Update performance metric for adaptation"""
        # Performance metric combines error magnitude and output stability
        error_component = abs(error)
        
        # Output stability (penalize rapid changes)
        if len(self.output_history) > 1:
            output_stability = abs(output - self.output_history[-2])
        else:
            output_stability = 0
        
        performance = error_component + 0.5 * output_stability
        self.performance_metric.append(performance)
    
    def reset(self):
        """Reset PID controller state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
        self.error_history.clear()
        self.derivative_history.clear()
        self.output_history.clear()
        self.performance_metric.clear()
        self.last_time = time.time()
        logging.info("🔄 Adaptive PID Controller Reset")

# -----------------------------------------------------------------------------
# --- ENHANCED ESP32 COMMUNICATION ---
# -----------------------------------------------------------------------------

class EnhancedESP32Communicator:
    """Enhanced ESP32 communication with better error handling and monitoring"""
    
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.sock = None
        self.last_command = None
        
        # Connection management
        self.connect_attempts = 0
        self.max_connect_attempts = 5
        self.reconnect_delay = 5.0
        self.last_reconnect_attempt = 0
        
        # Monitoring
        self.commands_sent = 0
        self.connection_errors = 0
        self.last_successful_send = 0
        self.response_times = deque(maxlen=20)
        
        # Command queuing for reliability
        self.command_queue = deque(maxlen=10)
        self.command_lock = threading.Lock()
        
        self.connect()
    
    def connect(self) -> bool:
        """Connect to ESP32 with improved error handling"""
        current_time = time.time()
        
        # Rate limit reconnection attempts
        if current_time - self.last_reconnect_attempt < self.reconnect_delay:
            return False
        
        self.last_reconnect_attempt = current_time
        
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        
        try:
            self.sock = socket.create_connection((self.ip, self.port), timeout=3)
            self.sock.settimeout(1.0)
            logging.info(f"✅ ESP32 connected: {self.ip}:{self.port}")
            self.connect_attempts = 0
            self.last_command = None
            return True
            
        except Exception as e:
            self.connection_errors += 1
            logging.error(f"❌ ESP32 connection failed (Attempt {self.connect_attempts+1}): {e}")
            self.sock = None
            self.connect_attempts += 1
            return False
    
    def send_command_reliable(self, speed_cmd: str, turn_cmd: str) -> bool:
        """Send command with reliability features"""
        command = f"{speed_cmd}:{turn_cmd}\n"
        
        with self.command_lock:
            # Add to queue for retry capability
            self.command_queue.append({
                'command': command,
                'timestamp': time.time(),
                'attempts': 0
            })
        
        return self._send_immediate(command)
    
    def _send_immediate(self, command: str) -> bool:
        """Send command immediately"""
        if not self.sock:
            if self.connect_attempts >= self.max_connect_attempts:
                return False
            if not self.connect():
                return False
        
        try:
            send_start = time.perf_counter()
            
            # Only send if command changed or significant time passed
            if (self.last_command != command or 
                time.time() - self.last_successful_send > 1.0):
                
                self.sock.sendall(command.encode())
                self.last_command = command
                self.last_successful_send = time.time()
                
                # Record response time
                response_time = time.perf_counter() - send_start
                self.response_times.append(response_time)
                
                self.commands_sent += 1
                logging.debug(f"📡 Sent to ESP32: {command.strip()}")
            
            return True
            
        except socket.timeout:
            logging.error("💥 ESP32 Send Timeout")
            self._handle_connection_error()
            return False
            
        except socket.error as e:
            logging.error(f"💥 ESP32 Socket Error: {e}")
            self._handle_connection_error()
            return False
            
        except Exception as e:
            logging.error(f"💥 ESP32 General Error: {e}")
            self._handle_connection_error()
            return False
    
    def _handle_connection_error(self):
        """Handle connection errors and prepare for reconnection"""
        self.connection_errors += 1
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        
        # Reset connection attempts if too many errors
        if self.connection_errors % 10 == 0:
            self.connect_attempts = 0
            logging.warning(f"🔄 Resetting ESP32 connection after {self.connection_errors} errors")
    
    def get_stats(self) -> Dict:
        """Get communication statistics"""
        avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0
        
        return {
            'connected': self.sock is not None,
            'commands_sent': self.commands_sent,
            'connection_errors': self.connection_errors,
            'avg_response_time_ms': avg_response_time * 1000,
            'queue_size': len(self.command_queue),
            'last_successful_send': self.last_successful_send
        }
    
    def close(self):
        """Close connection with proper cleanup"""
        if self.sock:
            try:
                # Send stop command before closing
                stop_command = "H:FORWARD\n"
                logging.info(f"Sending STOP command to ESP32: {stop_command.strip()}")
                self.sock.sendall(stop_command.encode())
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"⚠️ Error sending stop command: {e}")
            finally:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
                logging.info("🔌 ESP32 socket closed.")

# Speed command mapping
SPEEDS = {'FAST':'F', 'NORMAL':'N', 'SLOW':'S', 'TURN':'T', 'STOP':'H'}

# -----------------------------------------------------------------------------
# --- MAIN APPLICATION ---
# -----------------------------------------------------------------------------

def main():
    """Main application with all improvements"""
    
    # Load configuration
    config = ConfigManager()
    
    # Setup logging
    log_level = getattr(logging, config.system.log_level.upper())
    logging.basicConfig(level=log_level, 
                       format='🤖 [%(asctime)s] %(levelname)s: %(message)s', 
                       datefmt='%H:%M:%S')
    logger = logging.getLogger("EnhancedLineFollower")
    
    logger.info("🚀 Starting Enhanced Line Following Robot...")
    
    # Initialize components
    performance_monitor = PerformanceMonitor()
    data_logger = DataLogger(config.system)
    kalman_filter = KalmanLineFilter()
    
    # Initialize camera
    logger.info("📷 Initializing camera...")
    cap = cv2.VideoCapture(config.vision.cam_index)
    if not cap.isOpened():
        logger.error("❌ Camera failed to open")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.vision.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.vision.cam_height)
    cap.set(cv2.CAP_PROP_FPS, config.vision.cam_fps)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"📷 Camera: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS")
    
    # Initialize threaded image processor
    image_processor = ThreadedImageProcessor(config.vision)
    if config.system.enable_threading:
        image_processor.start()
    
    # Initialize PID controller
    pid_controller = AdaptivePIDController(config.control)
    
    # Initialize ESP32 communication
    esp32_comm = EnhancedESP32Communicator(config.system.esp32_ip, config.system.esp32_port)
    
    # Main loop variables
    frame_count = 0
    search_counter = 0
    last_known_offset = 0.0
    last_status_log = time.time()
    
    # Robot state
    current_offset = 0.0
    current_angle = 0.0
    current_steering = 0.0
    current_speed_cmd = SPEEDS['STOP']
    current_turn_cmd = "FORWARD"
    robot_status = "Ready"
    confidence_score = 0.0
    lines_detected = 0
    
    logger.info("🤖 Enhanced Robot READY!")
    
    try:
        while True:
            loop_start_time = time.perf_counter()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ No frame from camera")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Process frame
            if config.system.enable_threading:
                # Threaded processing
                image_processor.process_frame_async(frame)
                result = image_processor.get_result()
                
                if result:
                    final_offset, final_angle, detected_line_segments, confidence, processed_frame = result
                else:
                    # Use prediction if no result available
                    final_offset, final_angle = kalman_filter.predict()
                    detected_line_segments = []
                    confidence = 0.1
                    processed_frame = None
            else:
                # Synchronous processing (fallback)
                processed_frame = image_processor._preprocess_optimized(frame)
                roi_zones_data = image_processor._extract_roi_zones(processed_frame)
                final_offset, final_angle, detected_line_segments, confidence, _ = image_processor._detect_lines_enhanced(roi_zones_data, processed_frame)
            
            # Apply Kalman filtering
            if final_offset is not None and final_angle is not None:
                kalman_offset, kalman_angle = kalman_filter.update(final_offset, final_angle, confidence)
                # Use Kalman filtered values for control
                final_offset, final_angle = kalman_offset, kalman_angle
            
            # Update state
            current_offset = final_offset if final_offset is not None else 0.0
            current_angle = final_angle if final_angle is not None else 0.0
            lines_detected = len(detected_line_segments) if detected_line_segments else 0
            confidence_score = confidence
            
            # Control logic
            if final_offset is not None and confidence > 0.15:
                robot_status = f"Following (C:{confidence:.2f})"
                search_counter = 0
                last_known_offset = final_offset
                
                # PID control
                steering_error = -final_offset
                current_steering = pid_controller.calculate(steering_error)
                
                # Speed selection based on confidence and offset
                abs_offset = abs(final_offset)
                speed_zones = config.control.speed_zones
                
                if confidence > 0.7 and abs_offset < speed_zones['PERFECT']['threshold']:
                    current_speed_cmd = SPEEDS[speed_zones['PERFECT']['speed']]
                elif confidence > 0.4 and abs_offset < speed_zones['GOOD']['threshold']:
                    current_speed_cmd = SPEEDS[speed_zones['GOOD']['speed']]
                elif abs_offset < speed_zones['MODERATE']['threshold']:
                    current_speed_cmd = SPEEDS[speed_zones['MODERATE']['speed']]
                else:
                    current_speed_cmd = SPEEDS[speed_zones['LARGE']['speed']]
                
                # Turn command
                if abs(current_steering) < config.control.steering_deadzone:
                    current_turn_cmd = 'FORWARD'
                elif current_steering < -0.05:
                    current_turn_cmd = 'LEFT'
                elif current_steering > 0.05:
                    current_turn_cmd = 'RIGHT'
                else:
                    current_turn_cmd = 'FORWARD'
                    
            elif final_offset is not None and confidence > 0.05:
                robot_status = f"Weak Signal (C:{confidence:.2f})"
                search_counter = 0
                last_known_offset = final_offset
                
                steering_error = -final_offset * 0.7
                current_steering = pid_controller.calculate(steering_error)
                current_speed_cmd = SPEEDS['SLOW']
                current_turn_cmd = 'LEFT' if current_steering < 0 else 'RIGHT'
                
            else:
                # Search behavior
                robot_status = f"Searching... ({search_counter})"
                current_steering = 0.0
                pid_controller.reset()
                search_counter += 1
                
                if search_counter < 7:
                    current_turn_cmd = 'LEFT' if last_known_offset < 0 else 'RIGHT'
                    current_speed_cmd = SPEEDS['NORMAL']
                elif search_counter < 14:
                    current_turn_cmd = 'RIGHT' if last_known_offset < 0 else 'LEFT'
                    current_speed_cmd = SPEEDS['NORMAL']
                elif search_counter < config.control.search_timeout:
                    current_turn_cmd = 'LEFT' if (search_counter//4) % 2 == 0 else 'RIGHT'
                    current_speed_cmd = SPEEDS['SLOW']
                else:
                    current_turn_cmd = 'FORWARD'
                    current_speed_cmd = SPEEDS['SLOW']
                    if search_counter > config.control.search_timeout + 10:
                        search_counter = 0
            
            # Send command to ESP32
            esp32_comm.send_command_reliable(current_speed_cmd, current_turn_cmd)
            
            # Performance monitoring
            processing_time = time.perf_counter() - loop_start_time
            fps = 1.0 / processing_time if processing_time > 0 else config.vision.cam_fps
            performance_monitor.update(processing_time, fps, confidence_score)
            
            # Data logging
            data_logger.log_data({
                'frame_count': frame_count,
                'offset': current_offset,
                'angle': current_angle,
                'confidence': confidence_score,
                'steering_output': current_steering,
                'speed_cmd': current_speed_cmd,
                'turn_cmd': current_turn_cmd,
                'processing_time_ms': processing_time * 1000,
                'fps': fps,
                'line_count': lines_detected,
                'robot_status': robot_status,
                'esp32_connected': esp32_comm.sock is not None
            })
            
            # Periodic status logging
            if time.time() - last_status_log > 5.0:
                perf_stats = performance_monitor.get_stats()
                esp32_stats = esp32_comm.get_stats()
                
                logger.info(f"Status: {robot_status}, FPS: {perf_stats.get('avg_fps', 0):.1f}, "
                           f"Conf: {confidence_score:.2f}, PID_Kp: {pid_controller.kp:.3f}, "
                           f"ESP32: {'OK' if esp32_stats.get('connected') else 'ERR'}")
                
                last_status_log = time.time()
            
            # Frame rate control
            target_loop_time = 1.0 / config.vision.cam_fps
            if processing_time < target_loop_time:
                time.sleep(target_loop_time - processing_time)
    
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    
    except Exception as e:
        logger.error(f"💥 Fatal error in main loop: {e}", exc_info=True)
    
    finally:
        logger.info("🧹 Cleaning up...")
        
        # Cleanup
        if config.system.enable_threading:
            image_processor.stop()
        
        esp32_comm.close()
        data_logger.close()
        
        if cap.isOpened():
            cap.release()
        
        cv2.destroyAllWindows()
        
        # Save configuration
        config.save_config()
        
        logger.info("✅ Cleanup complete. Exiting.")

if __name__ == "__main__":
    main() 