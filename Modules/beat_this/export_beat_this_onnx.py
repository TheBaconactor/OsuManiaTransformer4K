from __future__ import annotations

import argparse
from pathlib import Path

from beat_this.onnx_inference import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Beat This! checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default="final0",
        help="Checkpoint name/path/url (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX file path. Default: beat_this/onnx_models/<checkpoint>.onnx",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    args = parser.parse_args()

    if args.output is None:
        out = (
            Path(__file__).resolve().parent
            / "beat_this"
            / "onnx_models"
            / f"{args.checkpoint}.onnx"
        )
    else:
        out = Path(args.output)

    out = export_onnx(checkpoint_path=args.checkpoint, onnx_path=out, opset=args.opset)
    print(str(out))


if __name__ == "__main__":
    main()

