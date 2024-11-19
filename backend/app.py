import cv2
from flask.templating import _render
from ultralytics import YOLO
from flask import Flask, Response, jsonify, render_template

# Initialize Flask app
app = Flask(__name__ , template_folder="../frontend/templates",
    static_folder="../frontend/static")

# Load YOLO model
model = YOLO('model/best6.pt')

# Class colors for bounding boxes
colors = {
    0: (0, 255, 0),  # Helmet
    1: (0, 0, 255),  # No Helmet
}

helmet_count = 0
no_helmet_count = 0

def generate_frames():
    global helmet_count, no_helmet_count
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Run YOLO detection
        results = model(resized_frame, device='cpu', conf=0.6, iou=0.4)

        helmet_count = 0
        no_helmet_count = 0

        # Process detections
        for result in results[0].boxes:
            class_id = int(result.cls.item())
            confidence = result.conf.item()

            if (class_id == 0 and confidence < 0.65) or (class_id == 1 and confidence < 0.5):
                continue

            # Increment counts
            if class_id == 0:
                helmet_count += 1
            elif class_id == 1:
                no_helmet_count += 1

            # Bounding box coordinates
            box = result.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # Draw bounding box and label
            color = colors.get(class_id, (255, 255, 255))
            label = f"{'No_helmet' if class_id == 1 else 'Helmet'}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Home route for webpage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts():
    """Return current counts as JSON."""
    total_count = helmet_count + no_helmet_count
    return jsonify({
        "helmet": helmet_count,
        "no_helmet": no_helmet_count
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
