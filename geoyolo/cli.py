import argparse
import os
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from geoyolo.core.inference import detect


def main():
    parser = argparse.ArgumentParser(description="GeoYOLO CLI")
    subparsers = parser.add_subparsers(dest="command")
    detect_parser = subparsers.add_parser("detect")

    detect_parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Image source, either single image path or directory of images",
    )

    detect_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model file path.",
    )

    detect_parser.add_argument(
        "--window_size",
        type=int,
        default=1024,
        help="Sliding window size.",
    )

    detect_parser.add_argument(
        "--stride",
        type=float,
        default=0.20,
        help="Sliding window overlap in horizontal and vertical direction.",
    )

    detect_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold.",
    )

    detect_parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IOU threshold.",
    )

    detect_parser.add_argument(
        "--agnostic",
        action="store_true",
        help="Run model in agnostic mode.",
    )

    detect_parser.add_argument(
        "--multi_label",
        action="store_true",
        help="Run model in multi-label mode.",
    )

    detect_parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="List of YOLO class indices to detect.",
    )

    detect_parser.add_argument(
        "--max_det",
        type=int,
        default=1000,
        help="Maximum number of detections to return.",
    )

    detect_parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device to use, e.g., 0, cpu, mps.",
    )

    detect_parser.add_argument(
        "--half",
        action="store_true",
        help="Run model in fp16/half precision mode.",
    )

    detect_parser.add_argument(
        "--export",
        type=str,
        default="geojson",
        choices=["geojson", "parquet", "database"],
        help="Export format. Options: geojson, parquet, database",
    )

    detect_parser.add_argument(
        "--export_dir",
        type=str,
        default=os.path.join(Path.home(), "detects"),
        help="Detection export directory for file export.",
    )

    detect_parser.add_argument(
        "--database_creds",
        type=str,
        help="Path to JSON containing database information.",
    )

    detect_parser.add_argument("--table", type=str, help="Database table name.")

    detect_parser.add_argument(
        "--bands",
        nargs="*",
        default=False,
        help="1-indexed bands to use to use for inference.",
    )

    detect_parser.add_argument(
        "--encode_chip",
        action="store_true",
        help="base64 encode detection chip",
    )

    detect_parser.set_defaults(func=detect)

    args = parser.parse_args()

    if args.command == "detect":
        info = f"""
GeoYOLO: Detect
Version: {metadata.metadata('geoyolo')['Version']}
Model Name: {os.path.basename(args.model_path).split('.')[0]}, Created: {datetime.fromtimestamp(os.path.getmtime(args.model_path))}
Device: {args.device}
Window Size: {args.window_size}x{args.window_size}
Stride: {args.stride}
Confidence: {args.confidence}
NMS: {args.iou}
"""
        print(info)
    else:
        print("Unrecognized command supplied!")

    if args.command:
        func_args = {
            k: v for k, v in vars(args).items() if k not in ("command", "func")
        }
        try:
            args.func(**func_args)
        except NotImplementedError as e:
            print(e)
            sys.exit(-1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
