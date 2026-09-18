#!/usr/bin/env python

"""
Local wrapper for resuming MLCast U-Net training without modifying the base MLCast code.

It injects MLCAST_CKPT_PATH into Lightning Trainer.fit(..., ckpt_path=...)
at runtime, then runs the normal `python -m mlcast ...` entry point.

Usage:
  export MLCAST_CKPT_PATH=/path/to/checkpoint.ckpt
  python scripts/Minh_mlcast_train_resume_wrapper.py train --config ...
"""

import os
import runpy


def patch_trainer_fit(trainer_cls, label):
    original_fit = trainer_cls.fit

    if getattr(original_fit, "_mlcast_resume_wrapper_patched", False):
        return

    def fit_with_env_checkpoint(self, *args, **kwargs):
        ckpt_path = os.environ.get("MLCAST_CKPT_PATH") or None

        if ckpt_path is not None and kwargs.get("ckpt_path") is None:
            print(
                f"[resume wrapper] Injecting ckpt_path into {label}.Trainer.fit: {ckpt_path}",
                flush=True,
            )
            kwargs["ckpt_path"] = ckpt_path

        return original_fit(self, *args, **kwargs)

    fit_with_env_checkpoint._mlcast_resume_wrapper_patched = True
    trainer_cls.fit = fit_with_env_checkpoint


try:
    import pytorch_lightning as pl

    patch_trainer_fit(pl.Trainer, "pytorch_lightning")
except Exception as e:
    print(f"[resume wrapper] Could not patch pytorch_lightning.Trainer.fit: {e}", flush=True)

try:
    import lightning.pytorch as L

    patch_trainer_fit(L.Trainer, "lightning.pytorch")
except Exception:
    pass


# Run the normal MLCast package entry point.
runpy.run_module("mlcast", run_name="__main__", alter_sys=True)
