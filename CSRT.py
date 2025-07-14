from ultralytics import YOLO
import time
import datetime 
import cv2


detection_interval=1
last_detection_time = time.time()

def calculate_intersection_over_union( box1, box2):
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
        iou = (intersection_area /
               float(box1_area
                     + box2_area
                     - intersection_area))

        return iou


# YOLOv11 modelini yükle
model = YOLO("best_h.pt", verbose= False).to("cuda:0")

# Kamera kaynağını başlat (0 = varsayılan webcam)
p=r"C:\Users\Zengin\Desktop\Gökbörü\2024-25 Savaşan iha\takip_goruntu isleme\video2.mp4"
cap = cv2.VideoCapture("udp://127.0.0.1:5001")

# Tracker ve takip durumu
csrt_params = cv2.TrackerCSRT_Params()

#FPS’e en büyük etkiyi yapan ayarlar —
csrt_params.use_hog            = False     # HOG kapalı → hız ↑, doğruluk ↓
csrt_params.use_color_names    = False     # Color Names kapalı → hız ↑
csrt_params.use_segmentation   = False     # Segmentasyon kapalı → hız ↑
csrt_params.admm_iterations    = 1         # ADMM iterasyon sayısını indirerek hız ↑
csrt_params.number_of_scales   = 17        # Daha az ölçek → hız ↑
csrt_params.scale_step         = 1.05      # Ölçek adım büyüklüğü
csrt_params.padding            = 2.0       # Daha küçük arama bölgesi → hız ↑
csrt_params.template_size      = 100       # Küçük template → hız ↑

tracker = None
tracking = False
matching_threshold = 0.1
tracker_life = 0

start_time = time.time()
frame_count = 0
while True:
    ret, frame = cap.read()
    frame_count += 1
    curr = time.time()
    elapsed_time=curr-last_detection_time
    
    if not ret:
        break
    if tracker_life == 0:
        tracker = None
        
    if elapsed_time >= detection_interval:
        
        results = model(source=frame, conf=0.4, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes.xyxy is not None else []

        if len (boxes) >0:
           
            # YOLO ile nesne tespiti
            if tracker is None:
                tracker = cv2.legacy.TrackerKCF_create()
                # İlk nesnenin bounding box’unu al
                x1, y1, x2, y2 = boxes[0]
                x1, y1, x2, y2 = map(int,(x1,y1,x2,y2))
                bbox = (x1, y1, x2 - x1, y2 - y1)
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                tracker_life = 100
                tracker.init(frame, bbox)
                
            else:
                success, bbox_tracker = tracker.update(frame)  
                tracker_life = tracker_life -1
                          
                if success:               
                    iou = calculate_intersection_over_union(bbox, bbox_tracker)
                    # print(iou)
                    if iou >= matching_threshold:
                        x, y, w, h = map(int, bbox_tracker)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "Takip Ediliyor", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        
                        tracker.init(frame, bbox)
                
        elif tracker is not None:
            success, bbox_tracker = tracker.update(frame)
            tracker_life = tracker_life -1
            if success:
                x, y, w, h = map(int, bbox_tracker)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Takip Ediliyor", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                tracker = None
                
        else:
            tracker = None
            
        
        last_detection_time = time.time()

    else :
        
        if tracker is not None:
            success, bbox_tracker = tracker.update(frame)
            tracker_life =tracker_life -1
            if success:
                
                x, y, w, h = map(int, bbox_tracker)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Takip Ediliyor", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else: 
                tracker=None
                
                
    gecen_sure = time.time() - start_time
    fps_adjust = max(0, (1 / 15) - gecen_sure)
    time.sleep(fps_adjust)
    # now = datetime.datetime.now()
    # timestamp = now.strftime("%M:%S.") + f"{int(now.microsecond / 1000):03d}"
    # print(f"[{timestamp}] FPS: {round(1 / (time.time() - start_time))}")
    # start_time = time.time()
    elpsd_time = time.time() - start_time
                    
    if elpsd_time > 1.0:
        avg = elpsd_time / frame_count
        print(frame_count/elpsd_time)
        frame_count = 0
        start_time = time.time()

    cv2.imshow("YOLO + Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
