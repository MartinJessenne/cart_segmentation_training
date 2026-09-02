"""Mesure precision et latence d'une variante entrainee, sur le split test.

Precision : mAP COCO en boite ET en masque. Les deux sont necessaires -- une
boite juste autour d'un masque comble donne une mAP boite honorable et une mAP
masque mediocre, et c'est le masque qui alimente l'etage suivant.

Latence : lot de 1, apres chauffe, avec synchronisation CUDA explicite. Sans
torch.cuda.synchronize les appels CUDA sont asynchrones et l'on chronometre le
temps de mise en file, pas le temps de calcul -- l'erreur donne des latences
absurdement basses.

Cette latence est celle de la carte de developpement, pas celle du Jetson Orin
Nano vise. L'ordre entre variantes se conserve generalement, les valeurs
absolues non : le choix final se valide sur la cible, sous TensorRT.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pycocotools.mask as coco_mask
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

VARIANTES = {"nano": RFDETRSegNano, "small": RFDETRSegSmall,
             "medium": RFDETRSegMedium}

# Seuil de confiance bas : la mAP integre la courbe precision/rappel et a besoin
# des detections peu sures pour en decrire la queue. Un seuil de deploiement
# (0.5) tronquerait la courbe et sous-estimerait le modele.
SEUIL_MAP = 0.05


def trouver_checkpoint(output_dir):
    for motif in ("checkpoint_best_total.pth", "checkpoint_best*.pth",
                  "checkpoint.pth", "*.pth"):
        trouves = sorted(glob.glob(os.path.join(output_dir, motif)))
        if trouves:
            return trouves[0]
    raise SystemExit(f"ABORT: aucun checkpoint dans {output_dir}")


def detecter_decalage(modele, chemins, ids_gt):
    """Compare les ids de classe predits aux ids annotes et deduit le decalage.

    Le modele raisonne en indices contigus, le fichier COCO en ids de categorie.
    Deviner ce decalage ferait s'effondrer la mAP en silence, donc on l'observe.
    """
    vus = set()
    for chemin in chemins[:200]:
        det = modele.predict(chemin, threshold=SEUIL_MAP)
        if det.class_id is not None and len(det.class_id):
            vus.update(int(c) for c in det.class_id)
    if not vus:
        raise SystemExit("ABORT: aucune detection sur l'echantillon, decalage "
                         "de classe indeterminable")
    if vus <= ids_gt:
        print(f"ids predits {sorted(vus)} deja dans les ids annotes "
              f"{sorted(ids_gt)} : aucun decalage")
        return 0
    decalage = min(ids_gt) - min(vus)
    print(f"ids predits {sorted(vus)} contre annotes {sorted(ids_gt)} : "
          f"decalage applique {decalage:+d}")
    return decalage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variante", choices=sorted(VARIANTES))
    ap.add_argument("--dataset-dir", default="_rfdetr_dataset")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--latence-iters", type=int, default=100)
    ap.add_argument("--latence-chauffe", type=int, default=20)
    args = ap.parse_args()

    output_dir = args.output_dir or f"output/seg_{args.variante}"
    checkpoint = trouver_checkpoint(output_dir)
    test_dir = os.path.join(args.dataset_dir, "test")
    ann_file = os.path.join(test_dir, "_annotations.coco.json")

    coco_gt = COCO(ann_file)
    ids_gt = {int(c) for c in coco_gt.getCatIds()}
    images = coco_gt.loadImgs(coco_gt.getImgIds())
    chemins = [os.path.join(test_dir, im["file_name"]) for im in images]

    print(f"variante   : {args.variante}")
    print(f"checkpoint : {checkpoint}")
    print(f"images test: {len(images)}\n")

    modele = VARIANTES[args.variante](pretrain_weights=checkpoint,
                                      num_classes=len(ids_gt))
    decalage = detecter_decalage(modele, chemins, ids_gt)

    # --- precision -----------------------------------------------------------
    resultats = []
    for n, (meta, chemin) in enumerate(zip(images, chemins), 1):
        det = modele.predict(chemin, threshold=SEUIL_MAP)
        if det.mask is None:
            raise SystemExit("ABORT: le modele ne renvoie pas de masque ; "
                             "verifier qu'il s'agit bien d'une variante Seg")
        for k in range(len(det.xyxy)):
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[k])
            rle = coco_mask.encode(
                np.asfortranarray(det.mask[k].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            resultats.append({
                "image_id": meta["id"],
                "category_id": int(det.class_id[k]) + decalage,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": rle,
                "score": float(det.confidence[k]),
            })
        if n % 500 == 0:
            print(f"  {n}/{len(images)}")

    if not resultats:
        raise SystemExit("ABORT: aucune detection sur le split test")

    resume = {"variante": args.variante, "checkpoint": checkpoint,
              "images_test": len(images), "detections": len(resultats)}

    coco_dt = coco_gt.loadRes(resultats)
    for type_iou in ("bbox", "segm"):
        ev = COCOeval(coco_gt, coco_dt, type_iou)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        resume[f"mAP_{type_iou}"] = round(float(ev.stats[0]), 4)
        resume[f"mAP50_{type_iou}"] = round(float(ev.stats[1]), 4)

    # --- latence -------------------------------------------------------------
    # Une seule image reutilisee : on mesure le modele, pas la lecture disque.
    img = Image.open(chemins[0]).convert("RGB")
    for _ in range(args.latence_chauffe):
        modele.predict(img, threshold=0.5)
    torch.cuda.synchronize()
    debut = time.perf_counter()
    for _ in range(args.latence_iters):
        modele.predict(img, threshold=0.5)
    torch.cuda.synchronize()
    latence_ms = (time.perf_counter() - debut) / args.latence_iters * 1000

    resume["latence_ms_lot1"] = round(latence_ms, 2)
    resume["gpu"] = torch.cuda.get_device_name(0)

    chemin_resume = os.path.join(output_dir, "resume_bench.json")
    with open(chemin_resume, "w") as fh:
        json.dump(resume, fh, indent=2)

    print(f"\n{args.variante}: mAP boite {resume['mAP_bbox']:.4f} | "
          f"mAP masque {resume['mAP_segm']:.4f} | "
          f"latence {latence_ms:.1f} ms")
    print(f"ecrit dans {chemin_resume}")


if __name__ == "__main__":
    main()
