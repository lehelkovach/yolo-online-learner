from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2  # type: ignore
import numpy as np


class MovingObject:
    def __init__(
        self,
        *,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        color: Tuple[int, int, int],
        radius: int,
        shape: str,
    ) -> None:
        self.x, self.y = position
        self.vx, self.vy = velocity
        self.color = color
        self.radius = int(radius)
        self.shape = shape

    def step(self, width: int, height: int) -> None:
        self.x += self.vx
        self.y += self.vy

        if self.x < self.radius or self.x > width - self.radius:
            self.vx *= -1
            self.x = float(np.clip(self.x, self.radius, width - self.radius))
        if self.y < self.radius or self.y > height - self.radius:
            self.vy *= -1
            self.y = float(np.clip(self.y, self.radius, height - self.radius))

    def draw(self, frame: np.ndarray) -> None:
        if self.shape == "circle":
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
        else:
            x1 = int(self.x - self.radius)
            y1 = int(self.y - self.radius)
            x2 = int(self.x + self.radius)
            y2 = int(self.y + self.radius)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)


def generate_video(
    *,
    output_path: Path,
    width: int,
    height: int,
    frames: int,
    fps: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {output_path}")

    obj_a = MovingObject(
        position=(width * 0.25, height * 0.3),
        velocity=(2.0, 1.5),
        color=(0, 255, 255),
        radius=18,
        shape="circle",
    )
    obj_b = MovingObject(
        position=(width * 0.7, height * 0.6),
        velocity=(-1.8, 2.2),
        color=(255, 0, 255),
        radius=22,
        shape="rect",
    )
    objects = [obj_a, obj_b]

    for _ in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for obj in objects:
            obj.step(width, height)
            obj.draw(frame)
        writer.write(frame)

    writer.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic two-object video.")
    parser.add_argument("--output", default="outputs/synthetic_two_objects.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    generate_video(
        output_path=Path(args.output),
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        seed=args.seed,
    )
    print(str(Path(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
