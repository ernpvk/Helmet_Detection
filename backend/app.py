import cv2
from flask.templating import _render
from ultralytics import YOLO
from flask import Flask, Response, jsonify, render_template
import platform

# Initialize Flask app
app = Flask(__name__, template_folder="../frontend/templates",
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

    # Check the OS
    os_name = platform.system()

    # Resolution based on OS
    if os_name == "Darwin":  # macOS
        resize_width, resize_height = 1215, 720
    else:  # Windows or other OS
        resize_width, resize_height = 640, 480

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the frame.")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Resize frame for detection
        resized_frame = cv2.resize(frame, (resize_width, resize_height))

        # Run YOLO detection
        results = model(resized_frame, device='cpu', conf=0.6, iou=0.4)

        helmet_count = 0
        no_helmet_count = 0

        # Process detections
        for result in results[0].boxes:
            if result is None:
                print("No results found in the frame")
                continue

            class_id = int(result.cls.item())
            confidence = result.conf.item()

            # Set confidence thresholds
            if (class_id == 0 and confidence < 0.65) or (class_id == 1 and confidence < 0.5):
                continue

            # Increment counts
            if class_id == 0:
                helmet_count += 1
            elif class_id == 1:
                no_helmet_count += 1

            # Bounding box coordinates
            box = result.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = map(int, box)

            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            # Scale bounding box coordinates to match the resized frame size
            scale_x = resize_width / frame.shape[1]
            scale_y = resize_height / frame.shape[0]

            # Adjust coordinates based on the new resolution
            x1 = int(x1 / scale_x)
            y1 = int(y1 / scale_y)
            x2 = int(x2 / scale_x)
            y2 = int(y2 / scale_y)

            # Draw bounding box and label
            color = colors.get(class_id, (255, 255, 255))
            label = f"{'No_helmet' if class_id == 1 else 'Helmet'}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode the frame.")
            break

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
    app.run(host='0.0.0.0', port=8000, debug=True)
