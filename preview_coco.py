"""Ecrit des apercus annotes, pour verifier a l'oeil que classe et geometrie collent.

Deux erreurs passent toutes les metriques sans laisser de trace :
 - une permutation des noms de classe, qui donne un modele parfaitement entraine
   a repondre le mauvais nom ;
 - un masque decale ou comble, dont la mAP reste honorable tant que la bbox tombe
   juste.

Les deux se voient immediatement sur une image annotee, et sur rien d'autre.
Chaque apercu porte le nom de classe issu du COCO, le masque decode depuis le RLE
et la bbox, pour que le nom puisse etre confronte au chariot reellement visible.
"""
import argparse
import json
import os

import cv2
import numpy as np
import pycocotools.mask as coco_mask

# Teinte d'incrustation par nom de classe, independante des couleurs du masque
# source : l'apercu doit rester lisible meme si le masque est faux.
OVERLAY = {"picanol": (60, 60, 230), "colruyt": (60, 210, 60),
           "leanflow": (230, 120, 60)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="_rfdetr_dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--out", default="_preview")
    args = ap.parse_args()

    split_dir = os.path.join(args.root, args.split)
    with open(os.path.join(split_dir, "_annotations.coco.json")) as fh:
        coco = json.load(fh)

    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}

    par_classe = {cid: [] for cid in names}
    for ann in coco["annotations"]:
        par_classe[ann["category_id"]].append(ann)

    os.makedirs(args.out, exist_ok=True)
    for cid, anns in par_classe.items():
        if not anns:
            print(f"{names[cid]}: aucune instance")
            continue
        # Repartis sur tout le split plutot que les premiers : les shards sont
        # ordonnes, prendre le debut ne montrerait qu'une poignee de scenes.
        pas = max(1, len(anns) // args.per_class)
        for k, ann in enumerate(anns[::pas][:args.per_class]):
            meta = images[ann["image_id"]]
            img = cv2.imread(os.path.join(split_dir, meta["file_name"]),
                             cv2.IMREAD_COLOR)
            rle = dict(ann["segmentation"])
            rle["counts"] = rle["counts"].encode("ascii")
            mask = coco_mask.decode(rle).astype(bool)

            couleur = OVERLAY[names[cid]]
            calque = img.copy()
            calque[mask] = couleur
            img = cv2.addWeighted(calque, 0.45, img, 0.55, 0)

            x, y, w, h = (int(v) for v in ann["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), couleur, 2)
            texte = f"{names[cid]}  aire={int(ann['area'])}px"
            cv2.putText(img, texte, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 5)
            cv2.putText(img, texte, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        couleur, 2)

            cible = os.path.join(args.out, f"{names[cid]}_{k}.png")
            cv2.imwrite(cible, img)
            print(f"{cible}  <- {meta['file_name']}")


if __name__ == "__main__":
    main()
