from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from beat_this.inference import aggregate_prediction, split_piece
from beat_this.inference import load_model as load_torch_model
from beat_this.inference import _refine_peak_times
from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect, load_audio


class _BeatThisOnnxWrapper(torch.nn.Module):
    def __init__(self, model: BeatThis):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["beat"], out["downbeat"]


def export_onnx(
    *,
    checkpoint_path: str = "final0",
    onnx_path: str | Path,
    opset: int = 17,
    chunk_size: int = 1500,
    spect_dim: int = 128,
) -> Path:
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_torch_model(checkpoint_path, device="cpu")
    wrapper = _BeatThisOnnxWrapper(model).eval()

    dummy = torch.zeros((1, chunk_size, spect_dim), dtype=torch.float32, device="cpu")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["spect"],
        output_names=["beat", "downbeat"],
        opset_version=opset,
        dynamic_axes={
            "spect": {1: "time"},
            "beat": {1: "time"},
            "downbeat": {1: "time"},
        },
    )
    return onnx_path


def get_session(
    onnx_path: str | Path,
    *,
    provider: str = "DmlExecutionProvider",
) -> ort.InferenceSession:
    onnx_path = str(onnx_path)
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(
            f"Requested ONNX Runtime provider '{provider}' not available. "
            f"Available: {available}"
        )

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Reduce noisy warnings (some nodes may still fall back to CPU).
    sess_options.log_severity_level = 3
    return ort.InferenceSession(onnx_path, sess_options=sess_options, providers=[provider])


def _onnx_predict(
    session: ort.InferenceSession, *, spect_chunk: torch.Tensor
) -> dict[str, torch.Tensor]:
    if spect_chunk.ndim != 2:
        raise ValueError(f"Expected (time, freq) spect chunk, got {spect_chunk.shape}")
    inp = spect_chunk.to(dtype=torch.float32, device="cpu").unsqueeze(0).numpy()
    beat, downbeat = session.run(None, {"spect": inp})
    return {
        "beat": torch.from_numpy(np.asarray(beat))[0],
        "downbeat": torch.from_numpy(np.asarray(downbeat))[0],
    }


class OnnxSpect2Frames:
    def __init__(
        self,
        *,
        checkpoint_path: str = "final0",
        onnx_path: str | Path | None = None,
        provider: str = "DmlExecutionProvider",
        float16: bool = False,
    ):
        self.float16 = float16
        if float16:
            raise ValueError("float16 is not supported for ONNX inference yet")

        if onnx_path is None:
            onnx_path = (
                Path(__file__).resolve().parent / "onnx_models" / f"{checkpoint_path}.onnx"
            )
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            export_onnx(checkpoint_path=checkpoint_path, onnx_path=onnx_path)

        self.session = get_session(onnx_path, provider=provider)

    def spect2frames(self, spect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        chunks, starts = split_piece(
            spect, chunk_size=1500, border_size=6, avoid_short_end=True
        )
        pred_chunks = [_onnx_predict(self.session, spect_chunk=chunk) for chunk in chunks]

        beat, downbeat = aggregate_prediction(
            pred_chunks=pred_chunks,
            starts=starts,
            full_size=spect.shape[0],
            chunk_size=1500,
            border_size=6,
            overlap_mode="keep_first",
            device="cpu",
        )
        return beat.float(), downbeat.float()

    def __call__(self, spect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spect2frames(spect)


class OnnxAudio2Frames(OnnxSpect2Frames):
    def __init__(
        self,
        *,
        checkpoint_path: str = "final0",
        onnx_path: str | Path | None = None,
        provider: str = "DmlExecutionProvider",
        float16: bool = False,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            provider=provider,
            float16=float16,
        )
        self.spect = LogMelSpect(device="cpu")

    def signal2spect(self, signal, sr) -> torch.Tensor:
        import soxr

        if signal.ndim == 2:
            signal = signal.mean(1)
        elif signal.ndim != 1:
            raise ValueError(f"Expected 1D or 2D signal, got shape {signal.shape}")
        if sr != 22050:
            signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
        signal = torch.tensor(signal, dtype=torch.float32, device="cpu")
        return self.spect(signal)

    def __call__(self, signal, sr) -> tuple[torch.Tensor, torch.Tensor]:
        spect = self.signal2spect(signal, sr)
        return self.spect2frames(spect)


class OnnxAudio2Beats(OnnxAudio2Frames):
    def __init__(
        self,
        *,
        checkpoint_path: str = "final0",
        onnx_path: str | Path | None = None,
        provider: str = "DmlExecutionProvider",
        float16: bool = False,
        dbn: bool = False,
        refine: bool = False,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            provider=provider,
            float16=float16,
        )
        self.frames2beats = Postprocessor(type="dbn" if dbn else "minimal")
        self.refine = refine

    def __call__(self, signal, sr):
        beat_logits, downbeat_logits = super().__call__(signal, sr)
        beat_time, downbeat_time = self.frames2beats(beat_logits, downbeat_logits)
        if not self.refine:
            return beat_time, downbeat_time

        fps = int(getattr(self.frames2beats, "fps", 50))
        beat_ref = _refine_peak_times(beat_logits, beat_time, fps=fps)

        if len(downbeat_time) == 0 or len(beat_time) == 0:
            down_ref = downbeat_time
        else:
            beat_time = np.asarray(beat_time, dtype=np.float64)
            downbeat_time = np.asarray(downbeat_time, dtype=np.float64)
            down_ref = np.empty_like(downbeat_time)
            for i, dt in enumerate(downbeat_time):
                j = int(np.argmin(np.abs(beat_time - dt)))
                down_ref[i] = beat_ref[j]
            down_ref = np.unique(down_ref)
        return beat_ref, down_ref


class OnnxFile2Beats(OnnxAudio2Beats):
    def __call__(self, audio_path):
        signal, sr = load_audio(audio_path)
        return super().__call__(signal, sr)
