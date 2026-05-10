"""
scrape_stylegan3.py
Scrapes thispersondoesnotexist.com for StyleGAN3 fake faces.
Each page refresh = new unique AI-generated face.

Usage:
    python scrape_stylegan3.py --count 5000 --output data/train/fake/stylegan3
"""

import requests
import os
import time
import argparse
from pathlib import Path


def scrape(count: int, output_dir: str, delay: float = 0.5):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://thispersondoesnotexist.com/",
    }

    url = "https://thispersondoesnotexist.com/image"
    saved = 0
    failed = 0

    print(f"Scraping {count} images → {output_dir}")
    print(f"Delay: {delay}s between requests\n")

    while saved < count:
        try:
            # Cache-bust with timestamp param
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
                filename = os.path.join(output_dir, f"sg3_{saved:05d}.jpg")
                with open(filename, "wb") as f:
                    f.write(r.content)
                saved += 1
                if saved % 100 == 0:
                    print(f"  {saved}/{count} saved...")
            else:
                failed += 1
                print(f"  Bad response: {r.status_code}")

            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            failed += 1
            print(f"  Request error: {e}")
            time.sleep(2)  # Back off on error

        except KeyboardInterrupt:
            print(f"\nInterrupted. Saved {saved} images.")
            break

    print(f"\nDone. Saved: {saved} | Failed: {failed}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000, help="Number of images to download")
    parser.add_argument("--output", type=str, default="data/train/fake/stylegan3", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds")
    args = parser.parse_args()

    scrape(args.count, args.output, args.delay)
