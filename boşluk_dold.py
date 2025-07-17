import cv2
from ultralytics import YOLO
import time
import datetime


class Utils:
    @staticmethod
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x1_1, y1_1, x1_2, y1_2 = x1, y1, x1 + w1, y1 + h1
        x2_1, y2_1, x2_2, y2_2 = x2, y2, x2 + w2, y2 + h2

        xA = max(x1_1, x2_1)
        yA = max(y1_1, y2_1)
        xB = min(x1_2, x2_2)
        yB = min(y1_2, y2_2)

        intersection_width = max(0, xB - xA + 1)
        intersection_height = max(0, yB - yA + 1)
        intersection_area = intersection_width * intersection_height
        box1_area = w1 * h1
        box2_area = w2 * h2

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou


class ObjectTracker:
    def __init__(self, model_path, video_source, device="cuda:0"):
        self.model = YOLO(model_path, verbose=False).to(device)
        self.cap = cv2.VideoCapture(video_source)

        self.tracker = None
        self.tracking = False
        self.matching_threshold = 0.4
        self.last_yolo_time = time.time()
        self.yolo_interval_frame = 1
        self.frame_counter = 0
        self.tracker_start_time = None
        self.start_time = time.time()
        self.bbox_tracker = None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # --- Tracker güncelle ---
            if self.tracking:
                success, self.bbox_tracker = self.tracker.update(frame)
                if success:
                    x, y, w, h = map(int, self.bbox_tracker)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    elapsed_time = time.time() - self.tracker_start_time
                    elapsed_text = f"{elapsed_time:.1f} sec"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, elapsed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    self.tracking = False
                    self.last_yolo_time = 0

            # --- YOLO belirli frame aralığında çalışır ---
            self.frame_counter += 1
            if self.frame_counter >= self.yolo_interval_frame:
                results = self.model(source=frame, conf=0.4, verbose=False)
                self.frame_counter = 0

                boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes.xyxy is not None else []

                if len(boxes) > 0:
                    confidences = results[0].boxes.conf.cpu().numpy()
                    best_id = confidences.argmax()
                    x1, y1, x2, y2 = boxes[best_id]
                    bbox_yolo = (x1, y1, x2 - x1, y2 - y1)
                    x, y, w, h = map(int, bbox_yolo)

                    iou = 0
                    if self.bbox_tracker is not None:
                        iou = Utils.calculate_iou(bbox_yolo, self.bbox_tracker)

                    if (iou < self.matching_threshold) or not self.tracking:
                        self.tracker = cv2.legacy.TrackerKCF_create()
                        self.tracker.init(frame, bbox_yolo)
                        self.tracking = True
                        self.tracker_start_time = time.time()
                        self.bbox_tracker = bbox_yolo

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # --- FPS sabitle ---
            gecen_sure = time.time() - self.start_time
            fps_adjust = max(0, (1 / 15) - gecen_sure)
            time.sleep(fps_adjust)
            now = datetime.datetime.now()
            timestamp = now.strftime("%M:%S.") + f"{int(now.microsecond / 1000):03d}"
            print(f"[{timestamp}] FPS: {round(1 / (time.time() - self.start_time))}")
            self.start_time = time.time()
        
            cv2.imshow("YOLO + Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    model_path = r"C:\Users\Zengin\Desktop\Gökbörü\2024-25 Savaşan iha\takip_goruntu isleme\best_h.pt"
    video_source = "udp://127.0.0.1:5001"

    app = ObjectTracker(model_path, video_source)
    app.run()

