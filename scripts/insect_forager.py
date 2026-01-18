from __future__ import annotations

import argparse

import cv2  # type: ignore
import numpy as np

from scripts.insect_collider import LoomingDetector


class InsectForager:
    def __init__(self, prey_min_size: float = 10.0, prey_max_size: float = 300.0) -> None:
        self.min_size = float(prey_min_size)
        self.max_size = float(prey_max_size)
        self.locked_target: tuple[int, int] | None = None
        self.patience = 0

    def hunt(self, saliency_map, debug_frame):
        """
        Scans for prey (small moving blobs).
        Returns: motor_command (str), updated_frame (image)
        """
        contours, _ = cv2.findContours(
            saliency_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motor_command = "SEARCHING"
        height, width = debug_frame.shape[:2]
        center_screen = (width // 2, height // 2)

        potential_prey = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_size < area < self.max_size:
                potential_prey.append(cnt)

        if potential_prey:
            best_target = min(
                potential_prey,
                key=lambda cnt: np.linalg.norm(
                    np.array(cv2.boundingRect(cnt)[:2]) - np.array(center_screen)
                ),
            )
            x, y, w, h = cv2.boundingRect(best_target)
            target_center = (x + w // 2, y + h // 2)

            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.line(debug_frame, center_screen, target_center, (0, 255, 255), 1)

            error_x = target_center[0] - center_screen[0]
            if abs(error_x) < 50:
                motor_command = "FORWARD (CHASE)"
            elif error_x < 0:
                motor_command = "TURN_LEFT"
            else:
                motor_command = "TURN_RIGHT"

            self.locked_target = target_center
            self.patience = 10
        else:
            if self.patience > 0:
                self.patience -= 1
                motor_command = "SCANNING (LOST TARGET)"
            else:
                motor_command = "ROAMING"

        return motor_command, debug_frame


def _parse_source(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def run_protoyolo_agent(source: str | int = 0) -> int:
    cap = cv2.VideoCapture(source)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to grab first frame.")
        return 1

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    collider = LoomingDetector(expansion_threshold=1.1)
    forager = InsectForager(prey_min_size=20, prey_max_size=400)

    print("ProtoYolo Agent Online. [RED = Threat] [YELLOW = Prey]")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        motion = cv2.absdiff(prev_gray, gray_blurred)
        _, motion_mask = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)

        threat_cmd, frame = collider.detect(motion_mask, frame)
        if "EVADE" in threat_cmd:
            final_action = f"** {threat_cmd} **"
        else:
            hunt_cmd, frame = forager.hunt(motion_mask, frame)
            final_action = hunt_cmd

        cv2.putText(
            frame,
            f"STATE: {final_action}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ProtoYolo: Integrated Agent", frame)

        prev_gray = gray_blurred
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProtoYolo Phase II forager demo.")
    parser.add_argument("--source", default="0", help="Video path or camera index (e.g. 0)")
    args = parser.parse_args(argv)
    return run_protoyolo_agent(_parse_source(args.source))


if __name__ == "__main__":
    raise SystemExit(main())
