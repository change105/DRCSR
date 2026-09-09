"""
General Amazon Dataset Preparation Script (DRCSR )
Supports the four datasets used in the paper:
    Baby
    Sports
    Games
    Office
Usage:
    cd /root/DRCSR
    python prepare_amazon.py --dataset Baby
    python prepare_amazon.py --dataset Sports
    python prepare_amazon.py --dataset Games
    python prepare_amazon.py --dataset Office
Optional arguments:
    --workers 16
    --device cuda
------------------------------------------------------------
Please download the original Amazon files yourself and place them under ./dataset/:

Baby:
    reviews_Baby_5.json.gz
    meta_Baby.json.gz

Sports:
    reviews_Sports_and_Outdoors_5.json.gz
    meta_Sports_and_Outdoors.json.gz

Games:
    reviews_Video_Games_5.json.gz
    meta_Video_Games.json.gz

Office:
    reviews_Office_Products_5.json.gz
    meta_Office_Products.json.gz

You also need to prepare the pretrained models yourself:
    ./pretrained/bert-base-uncased/pytorch_model.bin
    ./pretrained/vit_base_patch16_clip_224.pth

This script does not download the original Amazon datasets or the pretrained models above.
Product images are downloaded automatically from URLs in the metadata.
------------------------------------------------------------
"""

import argparse
import gzip
import os
import pickle
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add dataprocess to sys.path
sys.path.insert(0, "./dataprocess")

# Allow PIL to load truncated images
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Dataset configuration
# ============================================================

DATASET_SPECS = {
    "Baby": {
        "raw_name": "Baby",
    },
    "Sports": {
        "raw_name": "Sports_and_Outdoors",
    },
    "Games": {
        "raw_name": "Video_Games",
    },
    "Office": {
        "raw_name": "Office_Products",
    },
}

BERT_PATH = "./pretrained/bert-base-uncased/pytorch_model.bin"
VIT_PATH = "./pretrained/vit_base_patch16_clip_224.pth"


# ============================================================
# Utilities
# ============================================================

def compress_to_gz(json_path, gz_path):
    """Compress a .json file into .json.gz."""
    print(f"  Compressing: {json_path} -> {gz_path}")
    with open(json_path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(
        f"  Done, size: "
        f"{os.path.getsize(gz_path) / 1024 / 1024:.1f} MB"
    )


def load_filtered_items(item_dict_path):
    """Load item2id and return the filtered ASIN set."""
    with open(item_dict_path, "rb") as f:
        item_dict = pickle.load(f)

    if isinstance(item_dict, dict):
        return set(item_dict.keys())

    return set(item_dict)


def _first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _ensure_alias(source_path, alias_path):
    """
    Create a short dataset-name symlink expected by dataprocess.

    Example:
        reviews_Sports_and_Outdoors_5.json.gz
            ->
        reviews_Sports_5.json.gz
    """
    source_abs = os.path.abspath(source_path)
    alias_abs = os.path.abspath(alias_path)

    if source_abs == alias_abs:
        return

    if os.path.lexists(alias_path):
        # Keep an existing correct symlink or real file
        if os.path.islink(alias_path):
            current = os.path.realpath(alias_path)
            if current == source_abs:
                return
        return

    os.symlink(source_abs, alias_path)
    print(f"  ✓ Symlink: {alias_path} → {source_path}")


def resolve_raw_files(dataset):
    """
    Locate the downloaded Amazon files and create the short-name aliases required by dataprocess.
    """
    raw_name = DATASET_SPECS[dataset]["raw_name"]

    canonical_reviews = f"./dataset/reviews_{dataset}_5.json.gz"
    canonical_meta = f"./dataset/meta_{dataset}.json.gz"

    review_candidates = [
        canonical_reviews,
        f"./dataset/reviews_{raw_name}_5.json.gz",
    ]

    meta_candidates = [
        canonical_meta,
        f"./dataset/meta_{raw_name}.json.gz",
    ]

    reviews_source = _first_existing(review_candidates)
    meta_source = _first_existing(meta_candidates)

    if reviews_source is None:
        print("  ✗ Review file not found. Tried:")
        for p in review_candidates:
            print(f"      {p}")
        return None, None

    if meta_source is None:
        print("  ✗ Metadata file not found. Tried:")
        for p in meta_candidates:
            print(f"      {p}")
        return None, None

    print(f"  ✓ Reviews: {reviews_source}")
    print(f"  ✓ Metadata: {meta_source}")

    _ensure_alias(reviews_source, canonical_reviews)
    _ensure_alias(meta_source, canonical_meta)

    return canonical_reviews, canonical_meta


def _get_image_url(row):
    """Support image URL fields used by different Amazon metadata versions."""

    if "imUrl" in row.index:
        value = row["imUrl"]
        if isinstance(value, str) and value.strip():
            return value.strip()

    for field in ("imageURLHighRes", "image"):
        if field not in row.index:
            continue

        value = row[field]

        if isinstance(value, (list, tuple)) and len(value) > 0:
            url = value[0]
            if isinstance(url, str) and url.strip():
                return url.strip()

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def download_images_filtered(
    meta_gz_path,
    image_base_dir,
    filtered_items,
    num_workers=16,
):
    """
    Download images with multiple threads, only for filtered items.

    Image storage format:
        image_base_dir/{asin}/{asin}.jpg
    """
    from get_df import get_df

    os.makedirs(image_base_dir, exist_ok=True)

    print("  Loading metadata...")
    data_df = get_df(meta_gz_path)
    total_meta = len(data_df)

    tasks = []
    matched = 0
    skip = 0
    no_url = 0

    for i in range(total_meta):
        row = data_df.iloc[i]

        if "asin" not in row.index:
            continue

        asin = row["asin"]

        if asin not in filtered_items:
            continue

        matched += 1

        item_dir = os.path.join(image_base_dir, asin)
        save_path = os.path.join(item_dir, f"{asin}.jpg")

        # Skip if the image file already exists
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            skip += 1
            continue

        img_url = _get_image_url(row)

        if img_url is None:
            no_url += 1
            continue

        tasks.append((asin, img_url, item_dir, save_path))

    total_to_download = len(tasks)

    print(
        f"  Filtered items: {len(filtered_items)}, "
        f"matched in metadata: {matched}"
    )
    print(
        f"  to download: {total_to_download}, "
        f"existing: {skip}, no URL: {no_url}"
    )

    if total_to_download == 0:
        print("  No images need to be downloaded. Skipping.")
        return

    def _download_one(task):
        asin, img_url, item_dir, save_path = task

        try:
            os.makedirs(item_dir, exist_ok=True)

            request = urllib.request.Request(
                img_url,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                with open(save_path, "wb") as f:
                    shutil.copyfileobj(response, f)

            if os.path.getsize(save_path) == 0:
                raise RuntimeError("empty image")

            return asin, True

        except Exception:
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except OSError:
                    pass

            if os.path.exists(item_dir):
                try:
                    if len(os.listdir(item_dir)) == 0:
                        os.rmdir(item_dir)
                except OSError:
                    pass

            return asin, False

    success = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_download_one, task): task[0]
            for task in tasks
        }

        for i, future in enumerate(as_completed(futures), 1):
            _, ok = future.result()

            if ok:
                success += 1
            else:
                fail += 1

            if i % 500 == 0 or i == total_to_download:
                print(
                    f"    Progress: {i}/{total_to_download}, "
                    f"success={success}, failed={fail}"
                )

    print(
        f"  Image download finished: success {success}, "
        f"failed {fail}, existing {skip}, no URL {no_url}"
    )


# ============================================================
# Dataset preparation
# ============================================================

def prepare_dataset(dataset, device="cuda", num_workers=16):
    if dataset not in DATASET_SPECS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Supported: {list(DATASET_SPECS)}"
        )

    print("\n" + "=" * 70)
    print(f"  {dataset} dataset preparation")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Check raw data and create short-name aliases
    # --------------------------------------------------------
    print("\n[Step 1] Checking raw data...")

    reviews_gz, meta_gz = resolve_raw_files(dataset)

    if reviews_gz is None or meta_gz is None:
        print(f"\n✗ {dataset} raw data is incomplete. Stopping.")
        return False

    # --------------------------------------------------------
    # Step 2: Create directories
    # --------------------------------------------------------
    print("\n[Step 2] Creating directories...")

    for directory in [
        f"./dataset/{dataset}",
        f"./ckpt/{dataset}",
        "./log",
        "./saved",
    ]:
        os.makedirs(directory, exist_ok=True)

    print("  ✓ Done")

    # --------------------------------------------------------
    # Step 3: Generate .inter
    # --------------------------------------------------------
    print(f"\n[Step 3] Generating {dataset}.inter...")

    inter_path = f"./dataset/{dataset}/{dataset}.inter"

    if os.path.exists(inter_path):
        print("  ✓ already exists. Skipping.")
    else:
        from data_process import amazon, inter2txt
        from args import getArgs

        args = getArgs()
        args.dataset = dataset

        inter_df = amazon(args)
        inter2txt(inter_df, inter_path)

        print("  ✓ Done")

    # --------------------------------------------------------
    # Step 4: RecBole preprocessing
    # --------------------------------------------------------
    print(
        "\n[Step 4] RecBole preprocessing "
        "(item2id, interval_num, minmax_num)..."
    )

    item_dict_path = f"./dataset/{dataset}/item2id"
    data_path = f"./dataset/{dataset}/00_seq"

    preprocess_outputs = [
        item_dict_path,
        f"./dataset/{dataset}/interval_num",
        f"./dataset/{dataset}/minmax_num",
    ]

    if all(os.path.exists(path) for path in preprocess_outputs):
        print("  ✓ already exists. Skipping.")
    else:
        from data_process import prepare_seq
        from args import getArgs

        args = getArgs()
        args.dataset = dataset
        args.item_dict_path = item_dict_path
        args.data_path = data_path
        args.inter_path = inter_path

        prepare_seq(args)

        print("  ✓ Done")

    # --------------------------------------------------------
    # Step 5: Text embeddings
    # --------------------------------------------------------
    print("\n[Step 5] Extracting text embeddings (BERT)...")

    txt_emb_path = f"./dataset/{dataset}/txt_emb.pt"

    if os.path.exists(txt_emb_path):
        print("  ✓ already exists. Skipping.")
    else:
        if not os.path.exists(BERT_PATH):
            print(f"  ✗ BERT weights not found: {BERT_PATH}")
            print(
                "  Please download bert-base-uncased yourself and "
                "place it under ./pretrained/bert-base-uncased/."
            )
            return False

        from data_process import prepare_txt_emb
        from args import getArgs

        args = getArgs()
        args.dataset = dataset
        args.device = device
        args.txt_emb = txt_emb_path
        args.item_dict_path = item_dict_path
        args.txt_path = f"./dataset/{dataset}/content.txt"
        args.meta_path = meta_gz

        prepare_txt_emb(args)

        print("  ✓ Done")

    # --------------------------------------------------------
    # Step 6: Images + image embeddings
    # --------------------------------------------------------
    print("\n[Step 6] Extracting image embeddings...")

    img_emb_path = f"./dataset/{dataset}/img_emb.pt"

    if os.path.exists(img_emb_path):
        print("  ✓ already exists. Skipping.")
    else:
        if not os.path.exists(VIT_PATH):
            print(f"  ✗ ViT weights not found: {VIT_PATH}")
            print(
                "  Please download vit_base_patch16_clip_224.pth yourself and "
                "place it under ./pretrained/."
            )
            return False

        if not os.path.exists(item_dict_path):
            print(f"  ✗ item2id not found: {item_dict_path}")
            return False

        image_dir = f"./dataset/{dataset}/image/"
        filtered_items = load_filtered_items(item_dict_path)

        print(f"  Filtered items: {len(filtered_items)}")
        print(
            f"  Checking and downloading missing images "
            f"({num_workers} workers, filtered items only)..."
        )

        download_images_filtered(
            meta_gz_path=meta_gz,
            image_base_dir=image_dir,
            filtered_items=filtered_items,
            num_workers=num_workers,
        )

        from data_process import prepare_img_emb
        from args import getArgs

        args = getArgs()
        args.dataset = dataset
        args.device = device
        args.img_emb = img_emb_path
        args.item_dict_path = item_dict_path

        prepare_img_emb(args)

        print("  ✓ Done")

    # --------------------------------------------------------
    # Step 7: Category embeddings
    # --------------------------------------------------------
    print("\n[Step 7] Generating cat.pt (required by HM4SR)...")

    cat_path = f"./dataset/{dataset}/cat.pt"

    if os.path.exists(cat_path):
        print("  ✓ already exists. Skipping.")
    else:
        from data_process import prepare_category
        from args import getArgs

        args = getArgs()
        args.dataset = dataset
        args.item_dict_path = item_dict_path
        args.data_path = data_path
        args.meta_path = meta_gz

        prepare_category(args)

        print("  ✓ Done")

    # --------------------------------------------------------
    # Final check
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{dataset} final check:")
    print("=" * 70)

    files = {
        f"{dataset}.inter": inter_path,
        "txt_emb.pt": txt_emb_path,
        "img_emb.pt": img_emb_path,
        "cat.pt": cat_path,
        "item2id": item_dict_path,
        "interval_num": f"./dataset/{dataset}/interval_num",
        "minmax_num": f"./dataset/{dataset}/minmax_num",
    }

    all_ok = True

    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            unit = "KB"

            if size > 1024:
                size /= 1024
                unit = "MB"

            print(f"  ✓ {name:20s} {size:.1f} {unit}")
        else:
            print(f"  ✗ {name:20s} missing!")
            all_ok = False

    if all_ok:
        dataset_lower = dataset.lower()

        print(f"\n✓ {dataset} dataset preparation completed!")
        print("\nNext steps:")
        print(
            f"  python run_hm4sr_{dataset_lower}.py"
            f"       # HM4SR on {dataset}"
        )
        print(
            f"  python run_drcsr_{dataset_lower}.py"
            f"        # DRCSR on {dataset}"
        )
        print(
            f"  python run_baselines_{dataset_lower}.py --model all"
            f"  # baselines"
        )

        return True

    print("\n✗ Some files are missing. Please check the error messages.")
    return False


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Amazon datasets for DRCSR / HM4SR."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Baby", "Sports", "Games", "Office", "all"],
        help="Dataset to prepare.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used to extract multimodal embeddings.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of threads used for image downloading.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "all":
        datasets = ["Office", "Baby", "Games", "Sports"]
    else:
        datasets = [args.dataset]

    results = {}

    for dataset in datasets:
        try:
            results[dataset] = prepare_dataset(
                dataset=dataset,
                device=args.device,
                num_workers=args.workers,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            raise
        except Exception as exc:
            print(f"\n✗ {dataset} preparation failed: {exc}")
            results[dataset] = False

    if len(datasets) > 1:
        print("\n" + "=" * 70)
        print("Dataset preparation summary")
        print("=" * 70)

        for dataset in datasets:
            status = "✓ Done" if results.get(dataset) else "✗ failed"
            print(f"  {dataset:10s} {status}")


if __name__ == "__main__":
    main()
