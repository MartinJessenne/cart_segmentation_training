"""Live progress dashboard for RF-DETR Nano 480 training."""
import csv
import sys

def main():
    path = "output/seg_nano_480/metrics.csv"
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print("Waiting for first metric step...")
            return

        last = rows[-1]
        last_loss = next((r for r in reversed(rows) if r.get("train/loss")), None)

        epoch = int(last.get("epoch", 0))
        step = int(last.get("step", 0))
        steps_per_epoch = 617
        step_in_epoch = step % steps_per_epoch
        pct = (step_in_epoch / steps_per_epoch) * 100

        bar_len = 35
        filled = int(bar_len * step_in_epoch // steps_per_epoch)
        arrow = ">" if filled < bar_len else ""
        bar = "=" * filled + arrow + " " * max(0, bar_len - filled - len(arrow))

        print("⚡ RF-DETR Nano 480 Live Training Monitor")
        print("=" * 64)
        print(f"Epoch: {epoch:3d} / 100   [{bar}] {pct:5.1f}%")
        print(f"Step : {step:5d} / {steps_per_epoch * 100} ({step_in_epoch:3d}/{steps_per_epoch} in epoch)")
        print("-" * 64)

        if last_loss:
            print(f"Summary for Epoch {last_loss['epoch']}:")
            print(f"  • Total Loss     : {float(last_loss['train/loss']):.4f}")
            print(f"  • BBox Loss      : {float(last_loss['train/loss_bbox']):.4f}")
            print(f"  • GIoU Loss      : {float(last_loss['train/loss_giou']):.4f}")
            print(f"  • Mask CE Loss   : {float(last_loss['train/loss_mask_ce']):.4f}")
            print(f"  • Mask Dice Loss : {float(last_loss['train/loss_mask_dice']):.4f}")
            print("-" * 64)

        lr = last.get("train/lr") or last.get("train/lr_max") or "N/A"
        try:
            lr_str = f"{float(lr):.2e}"
        except Exception:
            lr_str = str(lr)
        print(f"Current Learning Rate: {lr_str}")
        print("=" * 64)

    except FileNotFoundError:
        print(f"Awaiting metrics file at {path}...")
    except Exception as e:
        print(f"Error reading metrics: {e}")

if __name__ == "__main__":
    main()
