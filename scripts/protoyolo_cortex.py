from __future__ import annotations

import argparse

import cv2  # type: ignore
import numpy as np


class ObjectMemory:
    """
    Maintains object permanence. If a moving blob stops, this memory keeps the box alive.
    """

    def __init__(self, persistence: int = 30) -> None:
        self.targets: dict[int, list] = {}
        self.next_id = 0
        self.max_patience = int(persistence)

    def update(self, detected_boxes):
        active_ids = list(self.targets.keys())
        for obj_id in active_ids:
            self.targets[obj_id][2] -= 1
            if self.targets[obj_id][2] < 0:
                del self.targets[obj_id]

        for box in detected_boxes:
            x, y, w, h = box
            cx, cy = x + w // 2, y + h // 2
            matched = False
            for obj_id, data in self.targets.items():
                saved_cx, saved_cy = data[0]
                dist = np.linalg.norm(np.array([cx, cy]) - np.array([saved_cx, saved_cy]))
                if dist < 50:
                    self.targets[obj_id] = [(cx, cy), box, self.max_patience, data[3]]
                    matched = True
                    break
            if not matched:
                self.targets[self.next_id] = [(cx, cy), box, self.max_patience, "Scanning..."]
                self.next_id += 1

        return self.targets


class DeepCortex:
    """
    Lightweight classifier (MobileNet SSD). Only runs on tracked crops.
    """

    def __init__(self) -> None:
        try:
            self.net = cv2.dnn.readNetFromCaffe(
                "MobileNetSSD_deploy.prototxt.txt",
                "MobileNetSSD_deploy.caffemodel",
            )
            self.active = True
            self.classes = [
                "Background",
                "Plane",
                "Bicycle",
                "Bird",
                "Boat",
                "Bottle",
                "Bus",
                "Car",
                "Cat",
                "Chair",
                "Cow",
                "Table",
                "Dog",
                "Horse",
                "Motorbike",
                "Person",
                "Plant",
                "Sheep",
                "Sofa",
                "Train",
                "Monitor",
            ]
        except Exception:
            print("[WARNING] MobileNet files not found. Cortex is running in blind mode.")
            self.active = False

    def classify(self, frame, box):
        if not self.active:
            return "Unknown"
        x, y, w, h = box
        roi = frame[y : y + h, x : x + w]
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            return "Noise"
        blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        best_conf = 0.0
        best_label = "Unknown"
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if confidence > best_conf:
                    best_conf = float(confidence)
                    best_label = self.classes[idx]
        return best_label


def _parse_source(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def run_full_evolution(source: str | int = 0) -> int:
    cap = cv2.VideoCapture(source)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to grab first frame.")
        return 1

    memory = ObjectMemory(persistence=40)
    cortex = DeepCortex()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    print("--- AGENT EVOLVED: CORTEX ONLINE ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        motion = cv2.absdiff(prev_gray, gray_blurred)
        _, motion_mask = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            if 500 < cv2.contourArea(cnt) < 15000:
                detections.append(cv2.boundingRect(cnt))

        tracked_objects = memory.update(detections)

        for obj_id, data in tracked_objects.items():
            centroid, box, patience, current_label = data
            x, y, w, h = box

            if current_label in ["Scanning...", "Unknown"]:
                new_label = cortex.classify(frame, box)
                tracked_objects[obj_id][3] = new_label

            label = tracked_objects[obj_id][3]
            if label == "Person":
                color = (0, 255, 0)
            elif label in ["Unknown", "Scanning..."]:
                color = (0, 255, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"ID:{obj_id} {label}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            bar_len = int((patience / 40) * w)
            cv2.line(frame, (x, y + h + 5), (x + bar_len, y + h + 5), color, 3)

        cv2.imshow("ProtoYolo: Full Stack", frame)
        prev_gray = gray_blurred
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProtoYolo Phase III/IV cortex demo.")
    parser.add_argument("--source", default="0", help="Video path or camera index (e.g. 0)")
    args = parser.parse_args(argv)
    return run_full_evolution(_parse_source(args.source))


if __name__ == "__main__":
    raise SystemExit(main())
