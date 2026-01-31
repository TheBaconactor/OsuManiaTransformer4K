"""
Fine-tune Beat This! on osu-derived supervision (automatic: mapper redlines).

Workflow:
  1) Build the dataset (writes to Modules/beat_this/data by default):
       python scripts/build_beat_this_osu_dataset.py --val-artist Rousseau

  2) Fine-tune from a pretrained checkpoint (downloads final0 if needed):
       python scripts/finetune_beat_this_osu.py --init-checkpoint final0 --tune heads

  3) Use the resulting .ckpt in beat_to_osu.py:
       python Modules/beat_this/beat_to_osu.py <audio> --checkpoint <path_to_ckpt> ...
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BEAT_THIS_ROOT = REPO_ROOT / "Modules" / "beat_this"
if str(BEAT_THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(BEAT_THIS_ROOT))


def _filter_kwargs_for(callable_obj, kwargs: dict) -> dict:
    sig = inspect.signature(callable_obj)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _freeze_module(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Beat This on osu-derived beats.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(BEAT_THIS_ROOT / "data"),
        help="Beat This data dir (contains audio/spectrograms + annotations).",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="final0",
        help="Beat This checkpoint name/path/URL to initialize from (use 'none' to train from scratch).",
    )
    parser.add_argument(
        "--tune",
        type=str,
        default="heads",
        choices=["heads", "frontend+heads", "all"],
        help="Which parts to train (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-length", type=int, default=1500, help="Sequence length in frames.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(BEAT_THIS_ROOT / "checkpoints" / "osu_finetune"),
        help="Where to write Lightning checkpoints (gitignored).",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Lightning fast dev run (1 train/val/test batch).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU index for CUDA training, or -1 for CPU (default).",
    )
    args = parser.parse_args()

    # Local imports (keep CLI usable even if deps are missing until you actually run training).
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    from beat_this.dataset import BeatDataModule
    from beat_this.inference import load_checkpoint
    from beat_this.model.pl_module import PLBeatThis

    seed_everything(int(args.seed), workers=True)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise RuntimeError(f"Missing data dir: {data_dir}")

    datamodule = BeatDataModule(
        data_dir,
        batch_size=int(args.batch_size),
        train_length=int(args.train_length),
        spect_fps=50,
        num_workers=int(args.num_workers),
        test_dataset="gtzan",  # can be missing; test split will be empty and skipped.
        augmentations={},
        hung_data=False,
        no_val=False,
        fold=None,
        length_based_oversampling_factor=0,
    )
    datamodule.setup(stage="fit")

    pos_weights = datamodule.get_train_positive_weights(widen_target_mask=3)
    print("[INFO] Positive weights:", pos_weights)

    init_checkpoint = str(args.init_checkpoint).strip()
    if init_checkpoint.lower() == "none":
        init_checkpoint = ""

    if init_checkpoint:
        ckpt = load_checkpoint(init_checkpoint, device="cpu")
        hparams = dict(ckpt.get("hyper_parameters", {}))
        # Build a PLBeatThis that matches the checkpoint architecture, but override training hparams.
        pl_kwargs = _filter_kwargs_for(PLBeatThis.__init__, hparams)
        pl_kwargs.update(
            dict(
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                pos_weights=pos_weights,
                max_epochs=int(args.max_epochs),
                use_dbn=False,
                eval_trim_beats=0,
            )
        )
        pl_model = PLBeatThis(**pl_kwargs)
        missing, unexpected = pl_model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print("[WARN] Missing keys when loading:", missing[:8], ("..." if len(missing) > 8 else ""))
        if unexpected:
            print("[WARN] Unexpected keys when loading:", unexpected[:8], ("..." if len(unexpected) > 8 else ""))
    else:
        pl_model = PLBeatThis(
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            pos_weights=pos_weights,
            max_epochs=int(args.max_epochs),
            use_dbn=False,
            eval_trim_beats=0,
        )

    # Freeze according to tuning mode.
    tune = str(args.tune)
    if tune == "heads":
        _freeze_module(pl_model.model.frontend)
        _freeze_module(pl_model.model.transformer_blocks)
    elif tune == "frontend+heads":
        _freeze_module(pl_model.model.transformer_blocks)
    elif tune == "all":
        pass
    else:
        raise ValueError(f"Unknown --tune={tune!r}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"osu_finetune-{tune}-S{int(args.seed)}",
            save_last=True,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
        ),
    ]

    if int(args.gpu) >= 0:
        accelerator = "gpu"
        devices = [int(args.gpu)]
        precision = "16-mixed"
    else:
        accelerator = "cpu"
        devices = 1
        precision = 32

    trainer = Trainer(
        max_epochs=int(args.max_epochs),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=None,
        log_every_n_steps=1,
        precision=precision,
        fast_dev_run=bool(args.fast_dev_run),
    )

    trainer.fit(pl_model, datamodule)
    print(f"[OK] Checkpoints written to: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

