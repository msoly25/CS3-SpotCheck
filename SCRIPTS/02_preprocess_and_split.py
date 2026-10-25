"""
Purpose: Resize images, verify dimensions, and create train/test folders

Outputs:
- Data/Final_Datasets/<species>/ : RGB JPEG images resized to 224 x 224
- Data/Split_Datasets/train/ and test/ : stratified 80/20 split
- train_manifest.csv and test_manifest.csv : records of the split

Required packages: Pillow, pandas, scikit-learn
"""

from pathlib import Path
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "Data"
ORIGINAL_DIR = DATA_DIR / "Original_Datasets"
FINAL_DIR = DATA_DIR / "Final_Datasets"
SPLIT_DIR = DATA_DIR / "Split_Datasets"

SPECIES = ["Harmonia_axyridis", "Coccinella_novemnotata"]
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Step 1: Resize originals into a separate finalized dataset.
for species in SPECIES:
    source_folder = ORIGINAL_DIR / species
    if not source_folder.exists():
        raise FileNotFoundError(f"Missing original image folder: {source_folder}")

    output_folder = FINAL_DIR / species
    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path for path in source_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    resized_count = 0

    for image_path in image_paths:
        try:
            # Convert images to RGB so every file has three color channels.
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                resized_image = image.resize(IMAGE_SIZE)
                # Use the original filename stem with a .jpg extension.
                output_path = output_folder / f"{image_path.stem}.jpg"
                resized_image.save(output_path, "JPEG")
                resized_count += 1
        except Exception as error:
            print(f"Could not process {image_path.name}: {error}")

    print(f"{species}: {resized_count} images resized")
  # Step 2: Check that the saved images have the intended dimensions.
for species in SPECIES:
    folder = FINAL_DIR / species
    images = list(folder.glob("*.jpg"))
    wrong_size = []

    for image_path in images:
        with Image.open(image_path) as image:
            if image.size != IMAGE_SIZE:
                wrong_size.append(image_path.name)

    print(f"\n{species}: {len(images)} resized images")
    print(f"Images with incorrect dimensions: {len(wrong_size)}")

# Step 3: Make a table of paths and labels for the split.
image_paths = []
labels = []

for species in SPECIES:
    species_folder = FINAL_DIR / species
    for image_path in sorted(species_folder.glob("*.jpg")):
        image_paths.append(str(image_path))
        labels.append(species)

image_data = pd.DataFrame({"image_path": image_paths, "label": labels})

# Step 4: Split 80% for training and 20% for testing.
# Stratification keeps the species proportions balanced in each subset.
train_data, test_data = train_test_split(
    image_data,
    test_size=0.20,
    random_state=42,
    stratify=image_data["label"],
)

# Clear the old split only after the new split has been calculated.
if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

# Copy images into folders that TensorFlow can load by class name.
for split_name, split_data in [("train", train_data), ("test", test_data)]:
    for species in SPECIES:
        destination_folder = SPLIT_DIR / split_name / species
        destination_folder.mkdir(parents=True, exist_ok=True)
        species_images = split_data[split_data["label"] == species]

        for _, row in species_images.iterrows():
            source_path = Path(row["image_path"])
            shutil.copy2(source_path, destination_folder / source_path.name)

# Save manifests so the assignment can document which images entered each split.
train_data.assign(split="train").to_csv(SPLIT_DIR / "train_manifest.csv", index=False)
test_data.assign(split="test").to_csv(SPLIT_DIR / "test_manifest.csv", index=False)

print("\nSplit complete.")
print("Training images:", len(train_data))
print("Testing images:", len(test_data))
print("\nTraining counts by species:")
print(train_data["label"].value_counts())
print("\nTesting counts by species:")
print(test_data["label"].value_counts())
