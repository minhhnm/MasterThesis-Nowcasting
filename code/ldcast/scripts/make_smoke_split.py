import argparse
import csv
import random
from pathlib import Path


def sample_csv(input_csv: Path, output_csv: Path, fraction: float, seed: int):
    rng = random.Random(seed)
    total = 0
    kept = 0

    with input_csv.open("r", newline="") as f_in, output_csv.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"{input_csv} has no header")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            if rng.random() < fraction:
                writer.writerow(row)
                kept += 1

    return total, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", required=True)
    ap.add_argument("--val_in", required=True)
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--val_out", required=True)
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    train_total, train_kept = sample_csv(
        Path(args.train_in), Path(args.train_out), args.fraction, args.seed
    )
    val_total, val_kept = sample_csv(
        Path(args.val_in), Path(args.val_out), args.fraction, args.seed + 1
    )

    with open(args.report, "w") as f:
        f.write(f"fraction={args.fraction}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"train_in={args.train_in}\n")
        f.write(f"train_out={args.train_out}\n")
        f.write(f"train_total={train_total}\n")
        f.write(f"train_kept={train_kept}\n")
        f.write(f"val_in={args.val_in}\n")
        f.write(f"val_out={args.val_out}\n")
        f.write(f"val_total={val_total}\n")
        f.write(f"val_kept={val_kept}\n")

    print("Done")
    print(f"train: kept {train_kept} / {train_total}")
    print(f"val:   kept {val_kept} / {val_total}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
