import argparse
import gc

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from radar_dataset_sampled import RadarDatasetNew
from mlcast.modules import ConvGRU


def radar_pair_collate(batch):
    x = torch.stack([sample["radar_past"] for sample in batch], dim=0)
    y = torch.stack([sample["radar_future"] for sample in batch], dim=0)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    y = y.permute(0, 2, 1, 3, 4).contiguous()
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--region", default="belgium")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tin", type=int, default=12)
    ap.add_argument("--tout", type=int, default=12)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed"])
    ap.add_argument("--use_dbz_norm", action="store_true")
    ap.add_argument("--allowed_nan_fraction", type=float, default=1.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on a GPU node.")

    torch.set_float32_matmul_precision("high")

    with open(args.config, "r") as f:
        data_sources = yaml.safe_load(f)

    if args.region not in data_sources:
        raise KeyError(f"Region {args.region!r} not found in config")

    if args.use_dbz_norm:
        data_sources[args.region]["radar"]["transform_precip_to_dbz"] = True

    train_ds = RadarDatasetNew(
        mode="train",
        regions=[args.region],
        is_primary_source=True,
        inputs=["radar_past"],
        targets=["radar_future"],
        data_sources=data_sources,
        number_of_past_radar_time_steps=args.tin,
        number_of_future_radar_time_steps=args.tout,
        crop_size=args.crop,
        allowed_nan_fraction=args.allowed_nan_fraction,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=radar_pair_collate,
        drop_last=True,
    )

    model = ConvGRU().cuda()
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    use_amp = args.precision == "16-mixed"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"dataset_len={len(train_ds)}")
    print(f"batch={args.batch}, workers={args.workers}, steps={args.steps}, precision={args.precision}")

    iterator = iter(train_loader)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for step_idx in range(args.steps):
        x, y = next(iterator)
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x, y.shape[1])
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(
            f"step={step_idx + 1}/{args.steps} "
            f"loss={loss.item():.6f} "
            f"peak_alloc_gb={peak_alloc:.2f} "
            f"peak_reserved_gb={peak_reserved:.2f}"
        )

    del model, optimizer, train_loader, train_ds
    gc.collect()
    torch.cuda.empty_cache()

    print(f"SUCCESS batch={args.batch}")


if __name__ == "__main__":
    main()
