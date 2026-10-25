"""
Script 3 — Create the exploratory plots used before model training.

Outputs:
- Outputs/eda_image_counts.png
- Outputs/eda_sample_images.png

Required packages: Pillow, matplotlib
"""

from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_DIR / "Data" / "Final_Datasets"
OUTPUT_DIR = PROJECT_DIR / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPECIES_FOLDERS = {
    "Native Ladybug\n(Coccinella novemnotata)": FINAL_DIR / "Coccinella_novemnotata",
    "Asian Lady Beetle\n(Harmonia axyridis)": FINAL_DIR / "Harmonia_axyridis",
}

# EDA plot 1: Count the processed images available for each species.
species_names = []
image_counts = []

for species_name, folder in SPECIES_FOLDERS.items():
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    count = sum(
        1 for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )
    species_names.append(species_name)
    image_counts.append(count)

plt.figure(figsize=(8, 5))
bars = plt.bar(species_names, image_counts)
plt.title("Number of Processed Images by Species")
plt.xlabel("Species")
plt.ylabel("Number of Images")
plt.bar_label(bars, padding=4)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_image_counts.png", dpi=300, bbox_inches="tight")
plt.show()
print("Image counts:", dict(zip(species_names, image_counts)))

# EDA plot 2: Show a small, reproducible sample of images from each species.
random.seed(42)
fig, axes = plt.subplots(2, 4, figsize=(12, 7))

for row, (species_name, folder) in enumerate(SPECIES_FOLDERS.items()):
    image_files = sorted([
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ])
    sample_files = random.sample(image_files, min(4, len(image_files)))

    for col in range(4):
        ax = axes[row, col]
        if col < len(sample_files):
            with Image.open(sample_files[col]) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"Example {col + 1}")
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[row, 0].set_ylabel(species_name, fontsize=10, labelpad=10)

fig.suptitle("Sample Images by Ladybug Species", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_sample_images.png", dpi=300, bbox_inches="tight")
plt.show()

print("EDA plots saved to:", OUTPUT_DIR)
