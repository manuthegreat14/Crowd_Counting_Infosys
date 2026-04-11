import cv2
import json
import numpy as np
from ultralytics import YOLO

# Load zones
def load_zones():
    try:
        with open("zones.json","r") as f:
            return json.load(f)
    except:
        return []

# Load model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Track unique IDs per zone
zone_id_map = {}

def point_in_zone(point, zone):
    polygon = np.array(zone, np.int32)
    return cv2.pointPolygonTest(polygon, point, False) >= 0


while True:

    ret, frame = cap.read()
    if not ret:
        break

    zones = load_zones()

    # initialize zone_id_map
    for i in range(len(zones)):
        if i not in zone_id_map:
            zone_id_map[i] = set()

    # YOLO tracking
    results = model.track(frame, classes=[0], persist=True)

    zone_counts = [0]*len(zones)

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # draw box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID {int(track_id)}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

            # check zones
            for i, zone in enumerate(zones):

                if point_in_zone((cx,cy), zone):

                    zone_id_map[i].add(int(track_id))

    # compute counts
    for i in range(len(zones)):
        zone_counts[i] = len(zone_id_map[i])

    # save for Flask
    with open("counts.json","w") as f:
        json.dump(zone_counts,f)

    # draw zones + counts
    for i, zone in enumerate(zones):

        pts = np.array(zone,np.int32)
        cv2.polylines(frame,[pts],True,(255,0,0),2)

        cv2.putText(frame,
                    f"Zone {i+1}: {zone_counts[i]}",
                    tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

    # total unique count
    total_unique = len(set().union(*zone_id_map.values())) if zone_id_map else 0

    cv2.putText(frame,
                f"Total Unique: {total_unique}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2)

    cv2.imshow("YOLOv8 Zone Tracking",frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()