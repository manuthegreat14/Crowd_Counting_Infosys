from flask import Flask, render_template, Response, jsonify, request, redirect, session, send_file
import cv2, json, numpy as np, time, csv
from ultralytics import YOLO
from reportlab.platypus import SimpleDocTemplate, Table
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key="1234"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

heatmap=None


# ======================
# UTIL FUNCTIONS
# ======================
def load_zones():
    try:
        with open("zones.json") as f:
            return json.load(f)
    except:
        return []

def point_in_zone(p,z):
    return cv2.pointPolygonTest(np.array(z,np.int32), p, False)>=0


# ======================
# LOGIN
# ======================
@app.route('/login',methods=["GET","POST"])
def login():
    if request.method=="POST":
        if request.form["username"]=="admin" and request.form["password"]=="1234":
            session["user"]="admin"
            return redirect("/")
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.pop("user",None)
    return redirect("/login")


# ======================
# VIDEO + HEATMAP + HISTORY
# ======================
def generate():
    global heatmap

    while True:
        ret,frame = cap.read()
        zones = load_zones()

        if heatmap is None:
            heatmap = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.float32)

        results = model.track(frame,classes=[0],persist=True)
        counts=[0]*len(zones)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for b in boxes:
                x1,y1,x2,y2 = map(int,b)
                cx,cy = (x1+x2)//2,(y1+y2)//2

                heatmap[cy,cx]+=1

                for i,z in enumerate(zones):
                    if point_in_zone((cx,cy),z):
                        counts[i]+=1

        # SAVE CURRENT COUNTS
        with open("counts.json","w") as f:
            json.dump(counts,f)

        # ======================
        # SAVE HISTORY
        # ======================
        record = {
            "timestamp": time.time(),
            "counts": counts
        }

        try:
            with open("history.json","r") as f:
                history = json.load(f)
        except:
            history = []

        history.append(record)

        # keep last 500 entries
        history = history[-500:]

        with open("history.json","w") as f:
            json.dump(history,f)

        # HEATMAP
        heat = cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(heat.astype(np.uint8),cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame,0.7,heat,0.3,0)

        _,buf=cv2.imencode('.jpg',frame)
        yield(b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(),mimetype='multipart/x-mixed-replace; boundary=frame')


# ======================
# STATS
# ======================
@app.route('/stats')
def stats():
    zones = load_zones()
    try:
        counts=json.load(open("counts.json"))
    except:
        counts=[]

    data={}
    for i,c in enumerate(counts):
        data[f"Zone {i+1}"]={
            "count":c,
            "status":"Overcrowded" if c>5 else "Normal"
        }

    return jsonify({"total":sum(counts),"zones":data,"timestamp":time.time()})


# ======================
# EXPORT CSV
# ======================
@app.route('/export_csv')
def export_csv():
    try:
        with open("history.json") as f:
            history = json.load(f)
    except:
        return "No data available"

    filename = "report.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Dynamic header
        max_zones = max(len(entry["counts"]) for entry in history) if history else 0
        header = ["Time"] + [f"Zone {i+1}" for i in range(max_zones)]
        writer.writerow(header)

        for entry in history:
            row = [time.strftime('%H:%M:%S', time.localtime(entry["timestamp"]))]
            row.extend(entry["counts"])
            writer.writerow(row)

    return send_file(filename, as_attachment=True)


# ======================
# EXPORT PDF
# ======================
@app.route('/export_pdf')
def export_pdf():
    try:
        with open("history.json") as f:
            history = json.load(f)
    except:
        return "No data available"

    filename = "report.pdf"
    doc = SimpleDocTemplate(filename)

    max_zones = max(len(entry["counts"]) for entry in history) if history else 0
    data = [["Time"] + [f"Zone {i+1}" for i in range(max_zones)]]

    for entry in history:
        row = [time.strftime('%H:%M:%S', time.localtime(entry["timestamp"]))]
        row.extend(entry["counts"])
        data.append(row)

    table = Table(data)
    table.setStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ])

    doc.build([table])

    return send_file(filename, as_attachment=True)


# ======================
# ZONE SAVE
# ======================
@app.route('/save_zones',methods=['POST'])
def save_zones():
    new_zones = request.json["zones"]

    try:
        with open("zones.json","r") as f:
            existing = json.load(f)
    except:
        existing = []

    updated = existing + new_zones

    with open("zones.json","w") as f:
        json.dump(updated,f)

    return "ok"

@app.route('/get_zones')
def get_zones():
    return jsonify(load_zones())


# ======================
# HOME
# ======================
@app.route('/')
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")


# ======================
# RUN
# ======================
if __name__=="__main__":
    app.run(debug=True)