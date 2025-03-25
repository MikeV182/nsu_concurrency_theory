import cv2
import time
import argparse
import logging
import threading
import queue
import os


LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, 'app.log'), level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Sensor:
    def get(self):
        raise NotImplementedError("Subclass must implement method get()")
    
class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0
    
    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data

class SensorCam(Sensor):
    def __init__(self, cam_name, resolution):
        self.cam_name = cam_name
        self.resolution = resolution
        self.cap = cv2.VideoCapture(cam_name)
        if not self.cap.isOpened():
            logging.error(f"Cannot open camera {cam_name}")
            raise RuntimeError(f"Cannot open camera {cam_name}")
        
        width, height = map(int, resolution.split('x'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to grab frame from camera")
            return None
        return frame
    
    def __del__(self):
        if self.cap:
            self.cap.release()

class WindowImage:
    def __init__(self, fps):
        self.fps = fps
        self.delay = int(1000 / fps)
        self.window_name = "Sensor Display"
        cv2.namedWindow(self.window_name)
    
    def show(self, img):
        cv2.imshow(self.window_name, img)
        if cv2.waitKey(self.delay) & 0xFF == ord('q'):
            return False
        return True
    
    def __del__(self):
        cv2.destroyAllWindows()

def sensor_worker(sensor, q):
    while True:
        data = sensor.get()
        q.put(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, required=True, help='Camera system name, e.g., /dev/video0')
    parser.add_argument('--resolution', type=str, required=True, help='Desired resolution, e.g., 1280x720')
    parser.add_argument('--fps', type=int, required=True, help='Display FPS')
    args = parser.parse_args()
    
    sensor0 = SensorX(0.01)
    sensor1 = SensorX(0.1)
    sensor2 = SensorX(1)
    
    cam = SensorCam(args.camera, args.resolution)
    window = WindowImage(args.fps)
    
    q0, q1, q2, q_cam = queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue()
    
    threads = [
        threading.Thread(target=sensor_worker, args=(sensor0, q0), daemon=True),
        threading.Thread(target=sensor_worker, args=(sensor1, q1), daemon=True),
        threading.Thread(target=sensor_worker, args=(sensor2, q2), daemon=True),
        threading.Thread(target=sensor_worker, args=(cam, q_cam), daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    sensor_data = {0: 0, 1: 0, 2: 0}
    while True:
        try:
            if not q0.empty():
                sensor_data[0] = q0.get()
            if not q1.empty():
                sensor_data[1] = q1.get()
            if not q2.empty():
                sensor_data[2] = q2.get()
            
            frame = q_cam.get() if not q_cam.empty() else None
            
            if frame is not None:
                cv2.putText(frame, f"S0: {sensor_data[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"S1: {sensor_data[1]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"S2: {sensor_data[2]}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if not window.show(frame):
                    break
        except KeyboardInterrupt:
            break
    
    del cam
    del window

if __name__ == "__main__":
    main()
