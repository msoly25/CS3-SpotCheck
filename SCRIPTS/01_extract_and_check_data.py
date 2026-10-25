"""
Purpose: Extract and Check Original Datasets
    1. Reassemble the original dataset ZIP if it was split into parts.
    2. Extract the two lady beetle species datasets.
    3. Check image counts and readability.

Expected input:
    Data/Original_Datasets.zip.part1
    Data/Original_Datasets.zip.part2

Alternatively, provide:
    Data/Original_Datasets.zip

Output:
    Data/Original_Datasets/
        Harmonia_axyridis/
        Coccinella_novemnotata/
"""

from pathlib import Path
from zipfile import ZipFile, BadZipFile
import shutil
import tempfile

from PIL import Image


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "Data"

ARCHIVE_PATH = DATA_DIR / "Original_Datasets.zip"
PART_FILES = [
    DATA_DIR / "Original_Datasets.zip.part1",
    DATA_DIR / "Original_Datasets.zip.part2",
]

ORIGINAL_DIR = DATA_DIR / "Original_Datasets"

SPECIES = [
    "Harmonia_axyridis",
    "Coccinella_novemnotata",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".tif", ".tiff", ".webp",
}

# 2. Reassemble the dataset archive if necessary

def merge_archive_parts():
    """Combine split archive files into one ZIP archive."""

    if ARCHIVE_PATH.exists():
        print("Original_Datasets.zip already exists.")
        return

    missing_parts = [part.name for part in PART_FILES if not part.is_file()]

    if missing_parts:
        raise FileNotFoundError(
            "Cannot find the complete dataset archive or all its parts.\n"
            f"Missing files: {', '.join(missing_parts)}"
        )

    print("Reassembling Original_Datasets.zip...")

    # Write to a temporary file so a failed merge does not leave
    # an incomplete archive at the expected output path.
    temporary_path = DATA_DIR / "Original_Datasets.zip.tmp"

    try:
        with temporary_path.open("wb") as output:
            for part in PART_FILES:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output)

        # Verify that the merged file is a valid ZIP archive.
        with ZipFile(temporary_path, "r") as archive:
            bad_file = archive.testzip()

            if bad_file is not None:
                raise BadZipFile(f"Corrupted archive member: {bad_file}")

        temporary_path.replace(ARCHIVE_PATH)
        print("Archive successfully reassembled.")

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

# 3. Extract the species folders

def extract_datasets():
    """Extract missing species folders without overwriting existing ones."""

    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)

    missing_species = [
        species for species in SPECIES
        if not (ORIGINAL_DIR / species).is_dir()
    ]

    if not missing_species:
        print("Both species folders already exist. Extraction skipped.")
        return

    with tempfile.TemporaryDirectory() as temporary_directory:

        temporary_directory = Path(temporary_directory)
        extraction_directory = temporary_directory / "extracted"
        extraction_directory.mkdir()

        print("Extracting the dataset archive...")

        with ZipFile(ARCHIVE_PATH, "r") as archive:
            # Check archive paths before extracting.
            for member in archive.infolist():
                target = (extraction_directory / member.filename).resolve()

                if not target.is_relative_to(extraction_directory.resolve()):
                    raise ValueError(
                        f"Unsafe path in archive: {member.filename}"
                    )

            archive.extractall(extraction_directory)

        # Look for species folders in the extracted archive.
        found_folders = {}

        for species in missing_species:
            matches = [
                path for path in extraction_directory.rglob(species)
                if path.is_dir() and path.name == species
            ]

            if matches:
                found_folders[species] = matches[0]

        # If the archive contains a nested ZIP, extract and check it.
        still_missing = [
            species for species in missing_species
            if species not in found_folders
        ]

        if still_missing:
            nested_archives = list(extraction_directory.rglob("*.zip"))

            for index, nested_archive in enumerate(nested_archives):

                nested_directory = temporary_directory / f"nested_{index}"
                nested_directory.mkdir()

                try:
                    with ZipFile(nested_archive, "r") as archive:
                        for member in archive.infolist():
                            target = (
                                nested_directory / member.filename
                            ).resolve()

                            if not target.is_relative_to(
                                nested_directory.resolve()
                            ):
                                raise ValueError(
                                    f"Unsafe path in archive: "
                                    f"{member.filename}"
                                )

                        archive.extractall(nested_directory)

                    for species in still_missing:
                        matches = [
                            path for path in nested_directory.rglob(species)
                            if path.is_dir() and path.name == species
                        ]

                        if matches:
                            found_folders[species] = matches[0]

                except BadZipFile:
                    print(f"Skipping invalid nested ZIP: {nested_archive.name}")

                still_missing = [
                    species for species in missing_species
                    if species not in found_folders
                ]

                if not still_missing:
                    break

        if still_missing:
            raise FileNotFoundError(
                "The following species folders were not found:\n"
                + "\n".join(still_missing)
                + "\nCheck the contents of Original_Datasets.zip."
            )

        # Copy only missing species folders.
        for species in missing_species:
            destination = ORIGINAL_DIR / species

            shutil.copytree(found_folders[species], destination)

            print(f"Extracted {species}.")

    print("Dataset extraction complete.")

# 4. Check image counts and readability

def check_images():
    """Count images and report files that cannot be opened."""

    total_images = 0
    total_readable = 0
    total_unreadable = 0

    print("\nIMAGE CHECK")
    print("-" * 40)

    for species in SPECIES:
        species_directory = ORIGINAL_DIR / species

        if not species_directory.is_dir():
            print(f"\nMissing folder: {species}")
            continue

        image_files = [
            path for path in species_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        readable = 0
        unreadable = []

        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    image.verify()

                readable += 1

            except Exception:
                unreadable.append(image_path.name)

        total_images += len(image_files)
        total_readable += readable
        total_unreadable += len(unreadable)

        print(f"\n{species}")
        print(f"  Images found:      {len(image_files)}")
        print(f"  Readable images:   {readable}")
        print(f"  Unreadable images: {len(unreadable)}")

        if unreadable:
            print("  Files needing review:")

            for filename in unreadable:
                print(f"    {filename}")

    print("\n" + "-" * 40)
    print("TOTALS")
    print(f"Images found:      {total_images}")
    print(f"Readable images:   {total_readable}")
    print(f"Unreadable images: {total_unreadable}")

# 5. Run the script

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    merge_archive_parts()
    extract_datasets()
    check_images()
```
