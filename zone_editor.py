import cv2
import json
import numpy as np
import os

cap = cv2.VideoCapture(0)

zones = []
current_zone = []

# Load previously saved zones
def load_existing_zones():
    if os.path.exists("zones.json"):
        try:
            with open("zones.json", "r") as f:
                return json.load(f)
        except:
            return []
    return []

zones = load_existing_zones()
print("Loaded zones:", len(zones))


def mouse_callback(event, x, y, flags, param):
    global current_zone

    if event == cv2.EVENT_LBUTTONDOWN:
        current_zone.append((x, y))


cv2.namedWindow("Zone Editor")
cv2.setMouseCallback("Zone Editor", mouse_callback)


while True:

    ret, frame = cap.read()
    if not ret:
        break

    temp = frame.copy()

    # Draw saved zones
    for i, zone in enumerate(zones):
        pts = np.array(zone, np.int32)
        cv2.polylines(temp, [pts], True, (0,255,0), 2)
        cv2.putText(temp, f"Zone {i+1}", tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Draw current points
    for p in current_zone:
        cv2.circle(temp, p, 5, (0,0,255), -1)

    # Draw current polygon (in-progress)
    if len(current_zone) > 1:
        cv2.polylines(temp, [np.array(current_zone)], False, (255,0,0), 2)

    # Instructions
    cv2.putText(temp, "Click: Add Points", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(temp, "N: New Zone | S: Save | C: Clear | R: Reset | Q: Quit",
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Zone Editor", temp)

    key = cv2.waitKey(1) & 0xFF

    # Finalize current zone
    if key == ord('n'):
        if len(current_zone) >= 3:
            zones.append(current_zone.copy())
            current_zone = []
            print("Zone added")
        else:
            print("Need at least 3 points")

    # Save zones (APPEND SAFE)
    elif key == ord('s'):
        try:
            existing = load_existing_zones()
        except:
            existing = []

        # Avoid duplicate zones (optional)
        updated = existing.copy()

        for z in zones:
            if z not in updated:
                updated.append(z)

        with open("zones.json", "w") as f:
            json.dump(updated, f)

        print("Zones saved (no overwrite)")
        zones = updated  # sync memory

    # Clear current drawing only
    elif key == ord('c'):
        current_zone = []
        print("Current zone cleared")

    # Reset ALL zones (file + memory)
    elif key == ord('r'):
        zones = []
        current_zone = []

        with open("zones.json", "w") as f:
            json.dump([], f)

        print("All zones reset")

    # Exit
    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()