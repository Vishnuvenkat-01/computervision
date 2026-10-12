"""
Submission Packaging Script for GateID Vision.
Packages the project into a clean, distributable ZIP archive
excluding virtual environments, caches, temporary files, large weights, and private data.
"""
from pathlib import Path
import zipfile
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP = PROJECT_ROOT.parent / "gateid-vision-submission.zip"

# Exclusion patterns
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    "ENV",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "staging",
}

EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".pt",
    ".pth",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".zip",
}

# Explicitly excluded files or paths
EXCLUDED_NAMES = {
    ".DS_Store",
    "gateid-vision-submission.zip",
}


def should_exclude(file_path: Path) -> bool:
    """Checks whether a file or directory should be excluded from packaging."""
    # Check directory parts
    for part in file_path.parts:
        if part in EXCLUDED_DIRS:
            return True

    # Check extension
    if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    # Check filename
    if file_path.name in EXCLUDED_NAMES:
        return True

    # Exclude generated snapshot images and generated reports while keeping .gitkeep
    if "snapshots" in file_path.parts and file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return True
    if "reports" in file_path.parts and file_path.suffix.lower() == ".csv":
        return True

    return False


def build_submission_zip() -> None:
    print(f"Creating submission archive from: {PROJECT_ROOT}")
    print(f"Destination: {OUTPUT_ZIP}")

    included_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            rel_dir = Path(root).relative_to(PROJECT_ROOT)

            # Skip excluded directories
            if any(part in EXCLUDED_DIRS for part in rel_dir.parts):
                continue

            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT)

                if should_exclude(rel_path):
                    continue

                # Add file to ZIP with gateid-vision/ prefix
                archive_name = Path("gateid-vision") / rel_path
                zip_file.write(file_path, arcname=str(archive_name.as_posix()))
                included_count += 1
                total_uncompressed_bytes += file_path.stat().st_size

    zip_size_kb = OUTPUT_ZIP.stat().st_size / 1024
    print("\nPackage Summary:")
    print(f"- Total Files Included: {included_count}")
    print(f"- Uncompressed Size: {total_uncompressed_bytes / 1024:.1f} KB")
    print(f"- Compressed ZIP Size: {zip_size_kb:.1f} KB")
    print(f"- Archive Location: {OUTPUT_ZIP}")

    # Verify archive readability
    print("\nVerifying archive integrity...")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as test_zip:
        corrupt = test_zip.testzip()
        if corrupt is not None:
            print(f"ERROR: Corrupted file in archive: {corrupt}")
            sys.exit(1)
        names = test_zip.namelist()
        print(f"Archive verified successfully ({len(names)} entries).")


if __name__ == "__main__":
    build_submission_zip()
