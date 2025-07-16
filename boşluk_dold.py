import cv2
from ultralytics import YOLO
import time
import datetime

def calculate_intersection_over_union(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_1, y1_1, x1_2, y1_2 = x1, y1, x1 + w1, y1 + h1
    x2_1, y2_1, x2_2, y2_2 = x2, y2, x2 + w2, y2 + h2

    xA = max(x1_1, x2_1)
    yA = max(y1_1, y2_1)
    xB = min(x1_2, x2_2)
    yB = min(y1_2, y2_2)

    intersection_width = max(0, xB - xA )
    intersection_height = max(0, yB - yA)
    intersection_area = intersection_width * intersection_height
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


model = YOLO(r"C:\Users\Zengin\Desktop\Gökbörü\2024-25 Savaşan iha\takip_goruntu isleme\best_h.pt", verbose=False).to("cuda:0")
ffmpeg="udp://127.0.0.1:5001"
cap = cv2.VideoCapture(0)

tracker = None
tracking = False
matching_threshold = 0.1  #ıou eşiği
last_yolo_time = time.time()
# yolo_interval = 1.0   #YOLO kaç saniyede bir çalıştırılsın
yolo_interval_frame=30
frame_counter=0
frame_fps=0

tracker_start_time = None
start_time = time.time()

bbox_tracker = None  

while True:
    ret, frame = cap.read()
    frame_fps +=1
    if not ret:
        break

    # Tracker sürekli çalışır
    if tracking:
        success, bbox_tracker = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox_tracker)
            elapsed_time = time.time() - tracker_start_time
            elapsed_text = f"{elapsed_time:.1f} sec"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, elapsed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            tracking = False
            last_yolo_time= 0
    frame_counter +=1
    """ Bu kısım saniye cinsinden çalıştırmak için
    # YOLO belirlenen aralıkta çalışır
    current_time = time.time()
    if current_time - last_yolo_time >= yolo_interval:
    """
    if frame_counter >= yolo_interval_frame:
        results = model.predict(source=frame, conf=0.4, verbose=False)
        frame_counter = 0 
    #     last_yolo_time = current_time

        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes.xyxy is not None else []

        if len(boxes) > 0:
            confidences= results[0].boxes.conf.cpu().numpy()
            best_id= confidences.argmax()
            x1, y1, x2, y2 = boxes[best_id]
            bbox_yolo = (x1, y1, x2 - x1, y2 - y1)
            x, y, w, h = map(int, bbox_yolo)

            iou = 0
            if bbox_tracker is not None:
                iou = calculate_intersection_over_union(bbox_yolo, bbox_tracker)
                # print(f"IoU: {iou:.2f}")

            # IoU düşükse veya takip yoksa yeni takip başlatılır
            if (iou < matching_threshold) or not tracking:
                tracker = cv2.legacy.TrackerKCF_create()
                tracker.init(frame, bbox_yolo)
                tracking = True
                tracker_start_time = time.time()
                bbox_tracker = bbox_yolo

            # Yeni bulunan YOLO kutusu (mavi olarak çizilir)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    gecen_sure = time.time() - start_time
    fps_adjust = max(0, (1 / 15) - gecen_sure)
    time.sleep(fps_adjust)
    # now = datetime.datetime.now()
    # timestamp = now.strftime("%M:%S.") + f"{int(now.microsecond / 1000):03d}"
    # print(f"[{timestamp}] FPS: {round(1 / (time.time() - start_time))}")
    if gecen_sure > 1.0:
        avg= gecen_sure/frame_fps
        print("fps:",frame_fps/gecen_sure)
        frame_fps=0
        start_time = time.time()

    cv2.imshow("YOLO + Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
