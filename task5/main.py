import argparse
import time
import cv2
import threading
from queue import Queue
from ultralytics import YOLO
from typing import Dict


class ModelWrapper:
    def __init__(self, model_path: str = 'yolov8s-pose.pt', device: str = 'cpu'):
        self.model = YOLO(model_path)
        self.model.to(device)

    def infer(self, frame) -> any:
        return self.model.predict(frame, verbose=False)[0]


class VideoProcessor:
    def __init__(self, video_path: str, output_path: str):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            output_path, 
            fourcc, 
            self.fps, 
            (self.frame_width, self.frame_height)
        )
        
    def release(self):
        self.cap.release()
        self.out.release()


class SingleThreadProcessor(VideoProcessor):
    def process(self):
        model = ModelWrapper()
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            result = model.infer(frame)
            annotated = result.plot(boxes=False, labels=False)
            self.out.write(annotated)
            frame_count += 1
            
        self.release()
        end_time = time.time()
        
        print(f"[Single Thread] Processed {frame_count} frames in {end_time - start_time:.2f} seconds")


class MultiThreadProcessor(VideoProcessor):
    def __init__(self, video_path: str, output_path: str, num_workers: int = 2):
        super().__init__(video_path, output_path)
        self.num_workers = num_workers
        self.input_queue = Queue()
        self.output_dict: Dict[int, any] = {}
        self.lock = threading.Lock()
        self.finished_event = threading.Event()

    def worker(self, worker_id: int):
        model = ModelWrapper()
        while not self.finished_event.is_set():
            try:
                index, frame = self.input_queue.get(timeout=1)
            except:
                continue
                
            result = model.infer(frame)
            annotated = result.plot(boxes=False, labels=False)
            
            with self.lock:
                self.output_dict[index] = annotated
                
            self.input_queue.task_done()

    def process(self):
        threads = [
            threading.Thread(target=self.worker, args=(i,), daemon=True) 
            for i in range(self.num_workers)
        ]
        
        for t in threads:
            t.start()

        index = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.input_queue.put((index, frame))
            index += 1

        self.input_queue.join()
        self.finished_event.set()

        for i in range(index):
            while i not in self.output_dict:
                time.sleep(0.01)
            self.out.write(self.output_dict[i])
            
        self.release()
        end_time = time.time()
        
        print(f"[Multi Thread ({self.num_workers})] Processed {index} frames in {end_time - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Estimation Video Processor")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--instances", type=int, default=1, help="Number of model instances (1 = single thread)")
    parser.add_argument("--output", type=str, required=True, help="Output video filename")

    args = parser.parse_args()

    try:
        if args.instances == 1:
            processor = SingleThreadProcessor(args.video, args.output)
        else:
            processor = MultiThreadProcessor(args.video, args.output, args.instances)
            
        processor.process()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()