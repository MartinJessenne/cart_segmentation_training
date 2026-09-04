"""Score an exported ONNX detector on REAL camera frames, by cart class.

Why this exists. The Isaac validation split is saturated: mAP@50-95 measures
0.9862 at epoch 4 and 0.9868 at epoch 9, and per-class AP sits at 0.985 for all
three carts with precision 1.000. Every checkpoint looks identical and perfect.
Yet the epoch-9 checkpoint that split selected classifies a real leanflow as
picanol on 296 of 297 bag frames. A metric that cannot separate a deployable
model from an unusable one cannot be used to choose between checkpoints, and
that is exactly what `BestModelCallback` was doing -- it preferred epoch 9 over
epoch 14 on a segm mAP difference of 0.0003.

This probe supplies the missing signal WITHOUT any hand labelling: each
recorded bag contains exactly one cart whose type is already known from the
bag, so the directory name is the label and the score is simply how often the
detector agrees.

WHAT THIS METRIC CAN AND CANNOT SAY. Every bag recorded so far is a leanflow,
so the probe has ONE class and a model that answered "leanflow" unconditionally
would score 1.000 on it. It is therefore only meaningful READ TOGETHER WITH the
Isaac validation mAP, which is what rules that degenerate model out: sim mAP
stays near 0.99 only while the network still separates all three carts. Use the
pair -- "sim mAP held AND real accuracy rose" -- never this number alone. The
day a colruyt or picanol bag is recorded, drop it in its own directory and the
ambiguity disappears.

Layout, one directory per known cart type:

    real_probe/leanflow/*.jpg
    real_probe/colruyt/*.jpg      # when such a bag exists

Usage:
    python3 eval_real_probe.py model.onnx --probe real_probe
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

# Ascending model class index. Read from the ONNX's own rfdetr_notes metadata
# when present, so a model exported with a different ordering cannot be scored
# against the wrong names.
DEFAULT_CLASS_NAMES = ["picanol", "colruyt", "leanflow"]

# RF-DETR normalises with the ImageNet statistics (rfdetr/detr.py) and resizes
# without preserving aspect (torchvision F.resize on the target shape), so the
# preprocessing below is the model's own, not a choice made here.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def class_names_of(session, path):
    for entry in session.get_modelmeta().custom_metadata_map.items():
        if entry[0] == "rfdetr_notes":
            names = json.loads(entry[1]).get("class_names")
            if names:
                return names
    print(f"{path}: no rfdetr_notes metadata, assuming {DEFAULT_CLASS_NAMES}",
          file=sys.stderr)
    return DEFAULT_CLASS_NAMES


def predict(session, input_name, net_w, net_h, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
    x = ((resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
    x = x.transpose(2, 0, 1)[None]
    _, labels, _ = session.run(None, {input_name: x})
    # The last logit is the background slot: RF-DETR's head is num_classes + 1
    # wide and reserves the final column (rfdetr/models/lwdetr.py, where
    # foreground_num_classes = detection_num_classes - 1). Scoring must ignore
    # it, or a confidently-empty query outranks every real detection.
    scores = 1.0 / (1.0 + np.exp(-labels[0][:, :-1]))
    query = int(np.argmax(scores.max(axis=1)))
    return int(np.argmax(scores[query])), float(scores[query].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--probe", default="real_probe")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    _, _, net_h, net_w = session.get_inputs()[0].shape
    names = class_names_of(session, args.model)
    print(f"{args.model}: input {net_w}x{net_h}, classes {names}\n")

    report, weighted_hits, weighted_total = {}, 0, 0
    for truth in sorted(os.listdir(args.probe)):
        directory = os.path.join(args.probe, truth)
        if not os.path.isdir(directory):
            continue
        files = sorted(f for f in os.listdir(directory)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not files:
            continue
        votes, scores = {}, []
        for name in files:
            image = cv2.imread(os.path.join(directory, name))
            if image is None:
                continue
            idx, score = predict(session, input_name, net_w, net_h, image)
            predicted = names[idx] if idx < len(names) else f"idx{idx}"
            votes[predicted] = votes.get(predicted, 0) + 1
            scores.append(score)
        hits = votes.get(truth, 0)
        total = sum(votes.values())
        weighted_hits += hits
        weighted_total += total
        report[truth] = {"accuracy": hits / total, "n": total, "votes": votes,
                         "mean_score": float(np.mean(scores))}
        print(f"  {truth:10s} {hits:4d}/{total:<4d} = {hits/total:6.1%}   "
              f"mean score {np.mean(scores):.3f}   {votes}")

    if not weighted_total:
        raise SystemExit(f"{args.probe}: no images found")
    overall = weighted_hits / weighted_total
    print(f"\n  REAL-DOMAIN ACCURACY  {weighted_hits}/{weighted_total} = {overall:.1%}")
    if len(report) == 1:
        print("  (single class in the probe -- read together with the Isaac "
              "validation mAP, which is what excludes a constant predictor)")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"overall": overall, "per_class": report}, fh, indent=2)
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
