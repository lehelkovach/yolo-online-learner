from __future__ import annotations

import argparse

import cv2  # type: ignore
import numpy as np


def _parse_source(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def insect_retina_stream(source: str | int = 0) -> int:
    cap = cv2.VideoCapture(source)

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to grab first frame.")
        return 1

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    print("Insect Retina Online. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_energy = cv2.absdiff(prev_gray, gray_blurred)
        _, motion_mask = cv2.threshold(motion_energy, 25, 255, cv2.THRESH_BINARY)

        edges = cv2.Laplacian(gray_blurred, cv2.CV_64F)
        edges = cv2.convertScaleAbs(edges)

        saliency = cv2.addWeighted(motion_mask, 0.8, edges, 0.2, 0)

        display_shape = (320, 240)
        view_raw = cv2.resize(frame, display_shape)
        view_motion = cv2.resize(motion_mask, display_shape)
        view_saliency = cv2.resize(saliency, display_shape)

        view_motion_c = cv2.cvtColor(view_motion, cv2.COLOR_GRAY2BGR)
        view_saliency_c = cv2.cvtColor(view_saliency, cv2.COLOR_GRAY2BGR)

        combined_view = np.hstack((view_raw, view_motion_c, view_saliency_c))
        cv2.imshow("ProtoYolo Retina: [Human] vs [Motion] vs [Saliency]", combined_view)

        prev_gray = gray_blurred

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProtoYolo insect retina viewer.")
    parser.add_argument("--source", default="0", help="Video path or camera index (e.g. 0)")
    args = parser.parse_args(argv)
    return insect_retina_stream(_parse_source(args.source))


if __name__ == "__main__":
    raise SystemExit(main())
