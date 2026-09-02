"""Rapatrie depuis le Hub les seules colonnes utiles a l'entrainement RGB.

Le parquet est colonnaire : ne selectionner que rgb, semantic et semantic_labels
fait que les plages d'octets de la colonne depth ne sont jamais demandees au
serveur. Mesure sur un shard : rgb + semantic pesent 1.29 MB par frame contre
3.3 MB pour une ligne complete, donc les 24705 frames des deux plages
representent ~32 GB au lieu des 82 GB du depot entier.

Le travail est fait shard par shard et marque par un fichier .done : une
interruption reseau reprend ou elle s'est arretee au lieu de tout refaire.

Disposition produite, celle qu'attend RF-DETR (les masques sont un
intermediaire, supprimable une fois le COCO ecrit) :

    <out>/train/<nom>.png            image RGB
    <out>/valid/<nom>.png
    <out>/test/<nom>.png
    <out>/_masks/<split>/<nom>.png   masque colorise par classe
    <out>/_masks/<split>/<nom>.json  id de classe -> nom de classe
"""
import argparse
import json
import os
import sys

import duckdb
from huggingface_hub import HfApi

REPO = "UItraviolet/industrial_cart"

# Le split "validation" du Hub s'appelle "valid" cote RF-DETR.
SPLIT_DIRS = {"train": "train", "validation": "valid", "test": "test"}

# Les trois colonnes qui portent la supervision RGB. rgb et semantic sont des
# features Image() de `datasets`, stockees en parquet comme struct<bytes, path> ;
# struct_extract evite l'ambiguite entre "colonne.champ" et "table.colonne".
QUERY = """
SELECT struct_extract(rgb, 'bytes')      AS rgb_bytes,
       struct_extract(semantic, 'bytes') AS sem_bytes,
       semantic_labels                   AS labels
FROM read_parquet('hf://datasets/{repo}/{path}')
"""


def shard_split(name):
    """Le split est le premier segment du nom de shard, par construction du nom."""
    return name.rsplit("/", 1)[-1].split("-")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_rfdetr_dataset")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="n'en traiter que N (mise au point) ; 0 = tous")
    args = ap.parse_args()

    api = HfApi()
    shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
                    if f.startswith("data/") and f.endswith(".parquet"))
    if not shards:
        sys.exit("ABORT: aucun shard parquet trouve sur le depot")

    unknown = {shard_split(s) for s in shards} - set(SPLIT_DIRS)
    if unknown:
        sys.exit(f"ABORT: splits inattendus {sorted(unknown)}")

    if args.limit_shards:
        shards = shards[:args.limit_shards]

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    done_dir = os.path.join(args.out, "_done")
    os.makedirs(done_dir, exist_ok=True)
    for d in set(SPLIT_DIRS.values()):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
        os.makedirs(os.path.join(args.out, "_masks", d), exist_ok=True)

    total_rows = 0
    total_bytes = 0
    for n, path in enumerate(shards, 1):
        stem = os.path.basename(path)[:-len(".parquet")]
        marker = os.path.join(done_dir, stem)
        if os.path.exists(marker):
            print(f"[{n}/{len(shards)}] {stem} deja fait")
            continue

        split_dir = SPLIT_DIRS[shard_split(path)]
        img_dir = os.path.join(args.out, split_dir)
        msk_dir = os.path.join(args.out, "_masks", split_dir)

        reader = con.execute(QUERY.format(repo=REPO, path=path)).to_arrow_reader(64)
        rows = 0
        written = 0
        for batch in reader:
            cols = batch.to_pydict()
            for rgb, sem, labels in zip(cols["rgb_bytes"], cols["sem_bytes"],
                                        cols["labels"]):
                if rgb is None or sem is None:
                    sys.exit(f"ABORT: octets manquants dans {stem}, ligne {rows}")
                name = f"{stem}_{rows:03d}"
                for blob, target in ((rgb, os.path.join(img_dir, name + ".png")),
                                     (sem, os.path.join(msk_dir, name + ".png"))):
                    with open(target, "wb") as fh:
                        fh.write(blob)
                    written += len(blob)
                with open(os.path.join(msk_dir, name + ".json"), "w") as fh:
                    fh.write(labels if isinstance(labels, str) else json.dumps(labels))
                rows += 1

        open(marker, "w").close()
        total_rows += rows
        total_bytes += written
        print(f"[{n}/{len(shards)}] {stem} -> {split_dir}  {rows} frames  "
              f"{written/1e6:.0f} MB  (cumul {total_bytes/1e9:.2f} GB)")

    print(f"\n{total_rows} frames ecrites, {total_bytes/1e9:.2f} GB sur disque")


if __name__ == "__main__":
    main()
