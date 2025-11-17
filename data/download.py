"""
Download, organize the Amazon ML Challenge 2024 dataset.
Expects the dataset CSV to be placed in data/raw/ manually from Kaggle.
handles downloading the product images from their URLs.
"""

import asyncio
import os
import json
import hashlib
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm


RAW_DIR = Path("data/raw")
IMAGE_DIR = RAW_DIR / "images"
CSV_PATH = RAW_DIR / "train.csv"


async def download_image(session, url, save_path, semaphore):
    """Download a single image with rate limiting."""
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    save_path.write_bytes(content)
                    return True
        except Exception:
            pass
        return False


def get_image_filename(url):
    """Create a stable filename from the image URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = url.split(".")[-1].split("?")[0]
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    return f"{url_hash}.{ext}"


async def download_all_images(df, max_concurrent=50):
    """Download all images from the dataframe."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in df.iterrows():
            url = row["image_link"]
            filename = get_image_filename(url)
            save_path = IMAGE_DIR / filename

            if save_path.exists():
                continue

            task = download_image(session, url, save_path, semaphore)
            tasks.append(task)

        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"):
            result = await coro
            results.append(result)

    success = sum(results)
    print(f"Downloaded {success}/{len(results)} images")


def main():
    if not CSV_PATH.exists():
        print(f"download the dataset CSV and place it at: {CSV_PATH}")
        print("get it from the Amazon ML Challenge 2024 on Kaggle.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from dataset")
    print(f"Columns: {list(df.columns)}")
    print(f"Entity types: {df['entity_name'].unique()}")

    asyncio.run(download_all_images(df))


if __name__ == "__main__":
    main()
