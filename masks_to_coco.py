"""Convertit les masques colorises Isaac en annotations COCO d'instance.

Encodage RLE et non polygones. Un chariot industriel est une structure ajouree :
le fond est visible a travers le cadre, et les polygones COCO sont additifs,
incapables de representer un trou. Un contour externe comblerait le cadre et
sur-estimerait le masque. Le RLE compresse de pycocotools represente les ajours
exactement, et le chargeur de RF-DETR le decode nativement
(convert_coco_poly_to_mask accepte les deux formes).

Ids de categorie 1..3 et non 0..2. RF-DETR derive ses indices de classe par
   {category["id"]: label for label, category in enumerate(kept)}
sur les seules categories annotees du split train, triees par id. L'ordre
croissant des ids reproduit donc CLASS_MAPPING a l'identique, et l'id 0 reste
libre pour le noeud parent que la convention Roboflow y place.

Une classe absente du split train ferait glisser tous les indices des autres :
le script s'arrete si les trois classes n'y sont pas toutes presentes.

Produit <racine>/<split>/_annotations.coco.json, la disposition attendue par
build_roboflow_from_coco.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pycocotools.mask as coco_mask
from PIL import Image

# Couleur de masque par id interne, telle qu'ecrite par generate_dataset_6_0_1.py.
SEMANTIC_COLORS = {
    0: (220, 50, 50),    # picanol
    1: (50, 200, 50),    # colruyt
    2: (50, 100, 220),   # leanflow
}
CLASS_NAMES = {0: "picanol", 1: "colruyt", 2: "leanflow"}

# Decalage entre l'id interne du generateur et l'id de categorie COCO.
COCO_ID_OFFSET = 1

SPLITS = ("train", "valid", "test")


def annotations_for(args):
    """Un masque -> une annotation par classe presente, en RLE.

    Retourne (nom de fichier, largeur, hauteur, liste d'annotations sans id).
    """
    img_path, mask_path, label_path = args

    with open(label_path) as fh:
        labels = json.load(fh)

    mask = np.array(Image.open(mask_path).convert("RGB"))
    height, width = mask.shape[:2]

    out = []
    for sem_id_str in labels:
        sem_id = int(sem_id_str)
        colour = SEMANTIC_COLORS[sem_id]
        # Les couleurs sont posees par affectation directe, sans reechantillonnage :
        # l'egalite exacte est la bonne comparaison, une tolerance ne ferait
        # qu'absorber une eventuelle erreur d'encodage sans la signaler.
        binary = np.all(mask == colour, axis=-1)
        if not binary.any():
            continue
        rle = coco_mask.encode(np.asfortranarray(binary.astype(np.uint8)))
        # counts est un bytes ; JSON exige du texte. RF-DETR redecode cette forme.
        rle["counts"] = rle["counts"].decode("ascii")
        x, y, w, h = (float(v) for v in coco_mask.toBbox(rle))
        out.append({
            "category_id": sem_id + COCO_ID_OFFSET,
            "segmentation": rle,
            "area": float(coco_mask.area(rle)),
            "bbox": [x, y, w, h],
            "iscrowd": 0,
        })

    return os.path.basename(img_path), width, height, out


def build_split(root, split, workers):
    img_dir = os.path.join(root, split)
    msk_dir = os.path.join(root, "_masks", split)
    if not os.path.isdir(img_dir):
        return None

    names = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    if not names:
        return None

    tasks = []
    for name in names:
        stem = name[:-len(".png")]
        mask_path = os.path.join(msk_dir, name)
        label_path = os.path.join(msk_dir, stem + ".json")
        if not (os.path.exists(mask_path) and os.path.exists(label_path)):
            sys.exit(f"ABORT: masque ou labels manquants pour {split}/{name}")
        tasks.append((os.path.join(img_dir, name), mask_path, label_path))

    images, annotations = [], []
    per_class = {cid: 0 for cid in CLASS_NAMES}
    empty = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for image_id, (file_name, width, height, anns) in enumerate(
                pool.map(annotations_for, tasks, chunksize=16)):
            images.append({"id": image_id, "file_name": file_name,
                           "width": width, "height": height})
            if not anns:
                empty += 1
            for ann in anns:
                ann["id"] = len(annotations)
                ann["image_id"] = image_id
                per_class[ann["category_id"] - COCO_ID_OFFSET] += 1
                annotations.append(ann)
            if (image_id + 1) % 2000 == 0:
                print(f"  {split}: {image_id + 1}/{len(tasks)}")

    coco = {
        "info": {"description": "industrial carts, Isaac Sim, instance masks"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid + COCO_ID_OFFSET, "name": CLASS_NAMES[cid],
                        "supercategory": "cart"}
                       for cid in sorted(CLASS_NAMES)],
    }

    out_path = os.path.join(img_dir, "_annotations.coco.json")
    with open(out_path, "w") as fh:
        json.dump(coco, fh)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"{split}: {len(images)} images, {len(annotations)} instances, "
          f"{empty} sans annotation, {size_mb:.0f} MB")
    for cid, n in sorted(per_class.items()):
        print(f"    {CLASS_NAMES[cid]:9s} {n:6d}")
    return per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="_rfdetr_dataset")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    print(f"{args.workers} processus\n")
    train_counts = None
    for split in SPLITS:
        counts = build_split(args.root, split, args.workers)
        if split == "train":
            train_counts = counts

    if train_counts is None:
        sys.exit("ABORT: aucun split train, RF-DETR n'a pas de quoi construire "
                 "son mapping de classes")
    missing = [CLASS_NAMES[c] for c, n in train_counts.items() if n == 0]
    if missing:
        sys.exit(f"ABORT: classes absentes du split train : {missing}. RF-DETR "
                 "derive ses indices des seules categories annotees dans train, "
                 "leur absence decalerait les indices des autres classes.")
    print("\ntrois classes presentes dans train, indices RF-DETR stables")


if __name__ == "__main__":
    main()
