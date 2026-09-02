"""Entraine une variante de segmentation RF-DETR sur le dataset de chariots.

Les trois variantes ne different pas seulement par leur profondeur : leur
resolution d'entree est figee dans leurs poids pre-entraines
(positional_encoding_size = resolution / patch_size), 312 pour nano, 384 pour
small, 432 pour medium. Cette resolution ne se regle donc pas, elle fait partie
de la variante -- et c'est elle qui porte l'essentiel de l'arbitrage
precision/latence que ce script sert a mesurer.

Tout le reste est tenu identique d'une variante a l'autre (memes epochs, meme
taille de lot effective, meme dataset, meme graine), sans quoi la comparaison
ne mesurerait plus la variante mais le protocole.

Le script va jusqu'a l'ONNX et jusqu'au Hub, dans le meme processus. Ce n'est
pas une commodite : la machine d'entrainement n'a pas de stockage persistant,
donc un checkpoint qui n'est pas sorti d'ici n'existe pas. Decouper export et
televersement en etapes separees rouvrirait une fenetre ou une variante est
entrainee mais pas livree.
"""
import argparse
import json
import os
import time

import torch
from huggingface_hub import HfApi, create_repo
from rfdetr import RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

VARIANTES = {
    "nano": (RFDETRSegNano, 312),
    "small": (RFDETRSegSmall, 384),
    "medium": (RFDETRSegMedium, 432),
}

# Ce qui merite de survivre a la machine : les poids retenus, le graphe exporte,
# les resumes et les journaux. Les checkpoints par epoch sont deliberement
# exclus, ils pesent lourd et ne portent aucune conclusion.
MOTIFS_PUBLIES = ["*.json", "*.txt", "*.onnx", "checkpoint_best*.pth"]


def noms_de_classes(dataset_dir):
    """Ordre des classes tel que RF-DETR le derive, lu du COCO et non retape.

    RF-DETR construit ses indices par
        {category["id"]: label for label, category in enumerate(kept)}
    sur les categories du split train triees par id. Trier ici par le meme id
    reproduit donc exactement l'ordre interne du modele. Cette liste part dans
    les metadonnees de l'ONNX : c'est le seul endroit ou l'ordre des classes
    accompagne le fichier, au lieu d'etre retape dans le detecteur C++ ou une
    permutation passe toutes les metriques en silence.
    """
    chemin = os.path.join(dataset_dir, "train", "_annotations.coco.json")
    with open(chemin) as fh:
        coco = json.load(fh)
    return [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]


def publier(output_dir, depot, variante):
    """Televerse le contenu utile de output_dir sous seg_<variante>/ du depot."""
    api = HfApi()
    create_repo(depot, exist_ok=True, repo_type="model", private=True)
    api.upload_folder(
        folder_path=output_dir,
        path_in_repo=f"seg_{variante}",
        repo_id=depot,
        repo_type="model",
        allow_patterns=MOTIFS_PUBLIES,
    )
    print(f"publie -> https://huggingface.co/{depot}/tree/main/seg_{variante}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variante", choices=sorted(VARIANTES))
    ap.add_argument("--dataset-dir", default="_rfdetr_dataset")
    ap.add_argument("--epochs", type=int, default=10)
    # Taille de lot effective = batch_size * grad_accum_steps. RF-DETR est reglee
    # pour 16 en affinage ; la carte a 96 GB, donc les 16 tiennent en un seul lot
    # et grad_accum reste a 1 pour les trois variantes.
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--depot-hf", default="UItraviolet/cart_segmentation_rfdetr")
    args = ap.parse_args()

    cls, resolution = VARIANTES[args.variante]
    output_dir = args.output_dir or f"output/seg_{args.variante}"
    os.makedirs(output_dir, exist_ok=True)

    classes = noms_de_classes(args.dataset_dir)

    print(f"variante   : {args.variante}  (resolution {resolution})")
    print(f"gpu        : {torch.cuda.get_device_name(0)}")
    print(f"capability : {torch.cuda.get_device_capability()}")
    print(f"dataset    : {args.dataset_dir}")
    print(f"classes    : {classes}")
    print(f"sortie     : {output_dir}\n")

    debut = time.time()
    modele = cls()
    modele.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=output_dir,
    )
    duree = time.time() - debut

    resume = {
        "variante": args.variante,
        "resolution": resolution,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "duree_entrainement_s": round(duree, 1),
        "gpu": torch.cuda.get_device_name(0),
        "class_names": classes,
    }
    with open(os.path.join(output_dir, "resume_entrainement.json"), "w") as fh:
        json.dump(resume, fh, indent=2)
    print(f"\nentraine en {duree/60:.1f} min")

    # `notes` est le seul canal de metadonnees que RF-DETR expose a l'export :
    # son contenu est serialise en JSON sous la cle `rfdetr_notes` du fichier
    # ONNX. Y placer l'ordre des classes rend le graphe auto-descriptif.
    chemin_onnx = modele.export(
        output_dir=output_dir,
        format="onnx",
        notes={"class_names": classes, "variante": args.variante,
               "resolution": resolution},
    )
    print(f"exporte -> {chemin_onnx}")

    publier(output_dir, args.depot_hf, args.variante)
    print(f"\ntermine en {(time.time() - debut)/60:.1f} min -> {output_dir}")


if __name__ == "__main__":
    main()
