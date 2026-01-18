from __future__ import annotations

import argparse

import cv2  # type: ignore


class LoomingDetector:
    def __init__(self, expansion_threshold: float = 1.1) -> None:
        self.prev_area = 0.0
        self.expansion_thresh = float(expansion_threshold)
        self.threat_level = 0
        self.tracking_center: tuple[int, int] | None = None

    def detect(self, saliency_map, debug_frame):
        """
        Scans the saliency map for looming threats.
        Returns: motor_command (str), updated_frame (image)
        """
        contours, _ = cv2.findContours(
            saliency_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motor_command = "HOVER"
        color = (0, 255, 0)

        if not contours:
            self.prev_area = 0.0
            return motor_command, debug_frame

        largest_blob = max(contours, key=cv2.contourArea)
        current_area = float(cv2.contourArea(largest_blob))

        if current_area < 100.0:
            return motor_command, debug_frame

        x, y, w, h = cv2.boundingRect(largest_blob)
        center_x = x + w // 2

        if self.prev_area > 0.0:
            expansion_rate = current_area / self.prev_area
            if expansion_rate > self.expansion_thresh:
                self.threat_level += 1
            else:
                self.threat_level = max(0, self.threat_level - 1)
        else:
            expansion_rate = 1.0

        height, width = debug_frame.shape[:2]

        if self.threat_level > 3:
            color = (0, 0, 255)
            cv2.putText(
                debug_frame,
                "COLLISION IMMINENT!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3,
            )
            motor_command = "EVADE_RIGHT" if center_x < width // 2 else "EVADE_LEFT"

        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug_frame,
            f"Expansion: {expansion_rate:.2f}x",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        self.prev_area = current_area

        return motor_command, debug_frame


def _parse_source(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def run_collider_simulation(source: str | int = 0) -> int:
    cap = cv2.VideoCapture(source)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to grab first frame.")
        return 1

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    brain = LoomingDetector(expansion_threshold=1.05)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        motion = cv2.absdiff(prev_gray, gray_blurred)
        _, motion_mask = cv2.threshold(motion, 30, 255, cv2.THRESH_BINARY)

        action, output_view = brain.detect(motion_mask, frame.copy())
        height, _ = output_view.shape[:2]
        cv2.putText(
            output_view,
            f"ACTION: {action}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.imshow("ProtoYolo: Phase I (Collider)", output_view)

        prev_gray = gray_blurred
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProtoYolo Phase I collider demo.")
    parser.add_argument("--source", default="0", help="Video path or camera index (e.g. 0)")
    args = parser.parse_args(argv)
    return run_collider_simulation(_parse_source(args.source))


if __name__ == "__main__":
    raise SystemExit(main())
