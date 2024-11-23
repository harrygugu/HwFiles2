import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectDetection():
    def __init__(self, video_path, output_path="result/tracked.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.model = self.load_model()
        self.CLASS_NAME_DICT = self.model.model.names

    def load_model(self):
        model = YOLO('yolov8n.pt')
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img, stream=True)
        return results

    def plot_boxes(self, results, img):
        # Prepare detections for DeepSORT
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Classname
                class_id = int(box.cls[0])
                class_name = self.CLASS_NAME_DICT[class_id]

                # Confidence
                conf = np.ceil(box.conf[0]*100)/100

                if conf > 0.5:  # Confidence threshold
                    detections.append((([x1, y1, w, h]), conf, class_name))

        return detections, img

    def track_detect(self, detections, img, tracker):
        tracks = tracker.update_tracks(detections, frame=img)

        # Draw results on the frame
        for track in tracks:
            if not track.is_confirmed():
                continue
            bbox = track.to_tlbr()  # Bounding box
            track_id = track.track_id  # Track ID

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return img

    def __call__(self):
        cap = cv2.VideoCapture(self.video_path)
        tracker = DeepSort(max_age=5,
                            n_init=2,
                            nms_max_overlap=1.0,
                            max_cosine_distance=0.3,
                            nn_budget=None,
                            override_track_class=None,
                            embedder="mobilenet",
                            half=True,
                            bgr=True,
                            embedder_gpu=True,
                            embedder_model_name=None,
                            embedder_wts=None,
                            polygon=False,
                            today=None)

        # Get video properties for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = self.track_detect(detections, frames, tracker)

            # Write and display frame
            out.write(detect_frame)
            cv2.imshow("Tracking", detect_frame)

            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(video_path="./data/Task4/tracking.mp4")
detector()