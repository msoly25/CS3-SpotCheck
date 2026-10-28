"""
Purpose:
    Validate the existing original lady beetle image dataset
    and its corresponding CSV file.
"""

# 1. Import libraries
# --------------------------------------------------

from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image

# 2. Project paths and settings
# --------------------------------------------------

# Google Drive project location
PROJECT_DIR = Path(
    "/content/drive/MyDrive/SpotCheck_Backup"
)

DATA_DIR = PROJECT_DIR / "Data"

ORIGINAL_DIR = DATA_DIR / "Original_Datasets"

CSV_PATH = ORIGINAL_DIR / "original_dataset.csv"

SPECIES = [
    "Coccinella_novemnotata",
    "Harmonia_axyridis"
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".tif", ".tiff", ".webp"
}

# 3. Verify dataset folder and CSV exist
# --------------------------------------------------

def check_dataset_paths():

    print("\nCHECKING DATASET PATHS")
    print("-" * 40)

    if not ORIGINAL_DIR.is_dir():
        raise FileNotFoundError(
            f"Original dataset folder not found: {ORIGINAL_DIR}"
        )

    print("Original_Datasets folder found.")

    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    print("original_dataset.csv found.")

# 4. Load and validate CSV
# --------------------------------------------------

def load_dataset_csv():

    print("\nLOADING ORIGINAL DATASET CSV")
    print("-" * 40)

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "filename",
        "label",
        "relative_path"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    print("CSV loaded successfully.")

    print(f"Total CSV records: {len(df)}")

    print("\nImages per species:")

    print(df["label"].value_counts(dropna=False))

    print("\nMissing CSV values:")

    print(df[required_columns].isnull().sum())

    return df

# 5. Check image readability
# --------------------------------------------------

def check_images():

    print("\nCHECKING IMAGE READABILITY")
    print("-" * 40)

    image_records = []
    unreadable_images = []

    for species in SPECIES:

        species_dir = ORIGINAL_DIR / species

        if not species_dir.is_dir():
            print(f"WARNING: Missing species folder: {species}")
            continue

        image_files = sorted(
            path for path in species_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        readable = 0

        for image_path in image_files:

            relative_path = image_path.relative_to(
                ORIGINAL_DIR
            ).as_posix()

            image_records.append(relative_path)

            try:
                with Image.open(image_path) as image:
                    image.verify()

                readable += 1

            except Exception as error:

                unreadable_images.append(
                    (relative_path, str(error))
                )

        print(f"\n{species}")
        print(f"Images found: {len(image_files)}")
        print(f"Readable images: {readable}")
        print(
            f"Unreadable images: {len(image_files) - readable}"
        )

    print("\nTOTAL IMAGE CHECK")

    print(f"Total images: {len(image_records)}")

    print(
        f"Total readable: "
        f"{len(image_records) - len(unreadable_images)}"
    )

    print(f"Total unreadable: {len(unreadable_images)}")

    if unreadable_images:

        print("\nUNREADABLE FILES:")

        for filename, error in unreadable_images:
            print(f"{filename}: {error}")

    return image_records, unreadable_images

# 6. Verify CSV matches image folders
# --------------------------------------------------

def validate_csv_paths(df, image_records):

    print("\nVALIDATING CSV AGAINST IMAGE FILES")
    print("-" * 40)

    missing_files = []
    incorrect_labels = []
    incorrect_filenames = []

    for _, row in df.iterrows():

        if pd.isna(row["relative_path"]):
            missing_files.append("<missing relative_path>")
            continue

        relative_path = str(row["relative_path"])

        image_path = ORIGINAL_DIR / relative_path

        if not image_path.is_file():
            missing_files.append(relative_path)

        # Check species label
        path_parts = Path(relative_path).parts

        if row["label"] not in SPECIES or (
            not path_parts or path_parts[0] != row["label"]
        ):
            incorrect_labels.append(relative_path)

        # Check filename
        if image_path.name != row["filename"]:
            incorrect_filenames.append(relative_path)

    csv_paths = set(
        df["relative_path"].dropna().astype(str)
    )

    folder_paths = set(image_records)

    unlisted_images = folder_paths - csv_paths

    duplicate_paths = df[
        df["relative_path"].duplicated(keep=False)
        & df["relative_path"].notna()
    ]

    print(f"CSV records: {len(df)}")

    print(f"Images found in folders: {len(image_records)}")

    print(f"Missing image files: {len(missing_files)}")

    print(f"Incorrect species labels: {len(incorrect_labels)}")

    print(f"Incorrect filenames: {len(incorrect_filenames)}")

    print(f"Unlisted images: {len(unlisted_images)}")

    print(f"Duplicate CSV paths: {len(duplicate_paths)}")

    if missing_files:
        print("\nMissing files:")
        for path in missing_files:
            print(path)

    if incorrect_labels:
        print("\nIncorrect labels:")
        for path in incorrect_labels:
            print(path)

    if incorrect_filenames:
        print("\nIncorrect filenames:")
        for path in incorrect_filenames:
            print(path)

    if unlisted_images:
        print("\nUnlisted images:")
        for path in sorted(unlisted_images):
            print(path)

    if not duplicate_paths.empty:
        print("\nDuplicate CSV paths:")
        print(duplicate_paths["relative_path"].tolist())

    return {
        "missing_files": missing_files,
        "incorrect_labels": incorrect_labels,
        "incorrect_filenames": incorrect_filenames,
        "unlisted_images": unlisted_images,
        "duplicate_paths": duplicate_paths
    }

# 7. Generate final validation report
# --------------------------------------------------

def generate_report(df, image_records, unreadable, results):

    print("\n" + "=" * 50)
    print("ORIGINAL DATASET VALIDATION REPORT")
    print("=" * 50)

    expected_total = 240
    expected_per_species = 120

    species_counts = Counter(df["label"].dropna())

    counts_valid = (
        len(df) == expected_total
        and len(image_records) == expected_total
        and all(
            species_counts[species] == expected_per_species
            for species in SPECIES
        )
    )

    csv_complete = not df[
        ["filename", "label", "relative_path"]
    ].isnull().any().any()

    paths_valid = (
        not results["missing_files"]
        and not results["incorrect_labels"]
        and not results["incorrect_filenames"]
        and not results["unlisted_images"]
        and results["duplicate_paths"].empty
    )

    images_readable = len(unreadable) == 0

    print(f"Expected image count verified: {counts_valid}")
    print(f"CSV completeness verified: {csv_complete}")
    print(f"CSV paths and labels verified: {paths_valid}")
    print(f"Image readability verified: {images_readable}")

    all_valid = (
        counts_valid
        and csv_complete
        and paths_valid
        and images_readable
    )

    print("\nFINAL STATUS:")

    if all_valid:
        print("PASS — Original dataset validation successful.")
    else:
        print("REVIEW REQUIRED — Dataset issues detected.")

    print("=" * 50)

    return all_valid

# 8. Run validation
# --------------------------------------------------

def main():

    check_dataset_paths()

    df = load_dataset_csv()

    image_records, unreadable = check_images()

    results = validate_csv_paths(
        df,
        image_records
    )

    valid = generate_report(
        df,
        image_records,
        unreadable,
        results
    )

    if not valid:
        raise ValueError(
            "Original dataset validation failed. "
            "Review the report above."
        )


if __name__ == "__main__":
    main()
