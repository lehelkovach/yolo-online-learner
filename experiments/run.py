from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from attention.scheduler import AttentionScheduler, empty_attention_metrics  # noqa: E402
from experiments.cognition import PerceptualLearner  # noqa: E402
from experiments.config import ExperimentConfig  # noqa: E402
from experiments.preview import OpenCvPreview  # noqa: E402
from features.encoder import SimpleCropEncoder  # noqa: E402
from features.simple_embedding import (  # noqa: E402
    attended_embedding_metrics,
    simple_embedding_schema,
)
from memory.prototypes import PrototypeMemoryConfig  # noqa: E402
from objects.binder import BinderConfig  # noqa: E402
from objects.memory import PermanenceConfig  # noqa: E402
from perception.recorded import RecordedBbpGenerator  # noqa: E402
from perception.video import iter_frames  # noqa: E402
from perception.yolo_adapter import YoloBbpGenerator  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def graph_snapshot_path(session_path: Path) -> Path:
    """Sidecar JSON holding the percept-graph snapshot for a session log."""
    return session_path.with_name(f"{session_path.stem}_graph.json")


def run_session(cfg: ExperimentConfig) -> Path:
    """Run a single recording session and write JSONL events."""
    _seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"session_seed{cfg.seed}_{int(time.time())}.jsonl"

    preview = None
    try:
        if cfg.detections is not None:
            gen = RecordedBbpGenerator(cfg.detections)
        else:
            gen = YoloBbpGenerator(
                model=cfg.yolo_model,
                device=cfg.yolo_device,
                conf=cfg.yolo_conf,
                iou=cfg.yolo_iou,
            )
        if cfg.preview:
            preview = OpenCvPreview()
        attention = AttentionScheduler()
        encoder = SimpleCropEncoder()
        learner = PerceptualLearner(cfg, encoder_space_id=encoder.space_id)
        stop_reason = "completed"

        with out_path.open("w", encoding="utf-8") as f:
            config_metrics = asdict(cfg)
            if not cfg.preview:
                config_metrics.pop("preview")
            start_event = {
                "event": "session_start",
                "config": config_metrics,
                "embedding_schema": simple_embedding_schema(),
                "cognition_schema": learner.schema(),
            }
            if cfg.preview:
                start_event["preview_enabled"] = True
            f.write(
                json.dumps(start_event)
                + "\n"
            )

            for fr in iter_frames(cfg.source, stride=cfg.stride, max_frames=cfg.max_frames):
                bbps = gen.detect_bbps(
                    frame_idx=fr.frame_idx,
                    timestamp_s=fr.timestamp_s,
                    frame_bgr=fr.image,
                )
                selection = attention.select(bbps)
                embedding_result = None
                embedding = None
                selected_bbp_index = None
                if selection is not None:
                    selected_bbp_index = selection.bbp_index
                    embedding_result = encoder.encode_crop(fr.image, selection.bbp.bbox)
                    embedding = encoder.to_result(embedding_result)
                    if embedding_result is not None:
                        enriched_bbp = replace(
                            selection.bbp,
                            embedding=embedding_result.vector,
                        )
                        bbps[selection.bbp_index] = enriched_bbp
                        selection = replace(selection, bbp=enriched_bbp)
                cognition = learner.step(
                    frame_idx=fr.frame_idx,
                    timestamp_s=fr.timestamp_s,
                    bbps=bbps,
                    selected_index=selected_bbp_index,
                    embedding=embedding,
                )

                attention_metrics = (
                    selection.to_metrics(len(bbps))
                    if selection is not None
                    else empty_attention_metrics()
                )
                embedding_metrics = attended_embedding_metrics(
                    embedding_result,
                    selected_bbp_index=selected_bbp_index,
                )
                f.write(
                    json.dumps(
                        {
                            "event": "frame",
                            "frame_idx": fr.frame_idx,
                            "timestamp_s": fr.timestamp_s,
                            "bbps": [b.to_dict() for b in bbps],
                            "attention": attention_metrics,
                            "attended_embedding": embedding_metrics,
                            **cognition,
                        }
                    )
                    + "\n"
                )

                if preview is not None and not preview.show(
                    fr.image,
                    bbps,
                    selected_bbp_index=selected_bbp_index,
                    priority=None if selection is None else selection.priority,
                    inhibited_count=0 if selection is None else selection.inhibited_count,
                ):
                    stop_reason = "operator_quit"
                    break

            graph_path = graph_snapshot_path(out_path)
            graph_path.write_text(json.dumps(learner.graph_snapshot()), encoding="utf-8")
            f.write(
                json.dumps(
                    {
                        "event": "cognition_summary",
                        **learner.summary(),
                        "graph_snapshot_path": graph_path.name,
                    }
                )
                + "\n"
            )
            end_event = {"event": "session_end"}
            if cfg.preview:
                end_event["stop_reason"] = stop_reason
            f.write(json.dumps(end_event) + "\n")
    finally:
        if preview is not None:
            preview.close()

    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run an experiment session and log JSONL events.")
    p.add_argument("--source", required=True, help="Video path or camera index (e.g. 0)")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--yolo-device", default=None)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou", type=float, default=0.7)
    p.add_argument("--preview", action="store_true", help="Show live BBPs and WTA attention")
    p.add_argument(
        "--detections",
        default=None,
        help=(
            "Replay recorded detections (a detections JSONL or an earlier session log) "
            "instead of running YOLO on the frames"
        ),
    )
    binder_defaults = BinderConfig()
    permanence_defaults = PermanenceConfig()
    prototype_defaults = PrototypeMemoryConfig(embedding_space_id=SimpleCropEncoder().space_id)
    p.add_argument("--match-threshold", type=float, default=binder_defaults.match_threshold)
    p.add_argument("--reid-threshold", type=float, default=binder_defaults.reid_threshold)
    p.add_argument(
        "--occluded-after", type=int, default=permanence_defaults.occluded_after_frames
    )
    p.add_argument("--lost-after", type=int, default=permanence_defaults.lost_after_frames)
    p.add_argument(
        "--dormant-after", type=int, default=permanence_defaults.dormant_after_frames
    )
    p.add_argument(
        "--prototype-threshold", type=float, default=prototype_defaults.match_threshold
    )
    p.add_argument("--max-prototypes", type=int, default=prototype_defaults.max_prototypes)
    p.add_argument(
        "--prototype-update-rule", choices=("running_mean", "learning_rate"),
        default=prototype_defaults.update_rule,
    )
    p.add_argument(
        "--prototype-learning-rate", type=float, default=prototype_defaults.learning_rate
    )
    args = p.parse_args(argv)

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    cfg = ExperimentConfig(
        seed=args.seed,
        source=source,
        max_frames=args.max_frames,
        stride=args.stride,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        preview=args.preview,
        output_dir=args.output_dir,
        detections=args.detections,
        binder=replace(
            binder_defaults,
            match_threshold=args.match_threshold,
            reid_threshold=args.reid_threshold,
        ),
        permanence=replace(
            permanence_defaults,
            occluded_after_frames=args.occluded_after,
            lost_after_frames=args.lost_after,
            dormant_after_frames=args.dormant_after,
        ),
        prototypes=replace(
            prototype_defaults,
            update_rule=args.prototype_update_rule,
            learning_rate=args.prototype_learning_rate,
            match_threshold=args.prototype_threshold,
            max_prototypes=args.max_prototypes,
        ),
    )
    out_path = run_session(cfg)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
