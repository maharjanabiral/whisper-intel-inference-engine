import sys
sys.path.insert(0, ".")
from huggingface_hub import snapshot_download
from pathlib import Path

MODELS = {
    "1": ("OpenVINO/whisper-tiny-fp16-ov",   "models/whisper-tiny"),
    "2": ("OpenVINO/whisper-base-fp16-ov",   "models/whisper-base"),
    "3": ("OpenVINO/whisper-small-fp16-ov",  "models/whisper-small"),
    "4": ("OpenVINO/whisper-medium-fp16-ov", "models/whisper-medium"),
    "5": ("OpenVINO/whisper-large-v3-fp16-ov","models/whisper-large-v3"),
}

print("Which model do you want to download?")
print()
for key, (repo_id, local_dir) in MODELS.items():
    print(f"  {key}. {repo_id}")
print()

choice = input("Enter number (default 3 = small): ").strip() or "3"

if choice not in MODELS:
    print(f"Invalid choice: {choice}")
    sys.exit(1)

repo_id, local_dir = MODELS[choice]
output_dir = Path(local_dir)

if output_dir.exists():
    print(f"\n Model already exists at {output_dir}")
    overwrite = input("Re-download? (y/N): ").strip().lower()
    if overwrite != "y":
        print("Skipping download.")
        sys.exit(0)

print(f"\nDownloading {repo_id}...")
print(f"Saving to  {output_dir}\n")

snapshot_download(
    repo_id=repo_id,
    local_dir=output_dir,
)

print(f"\nModel saved to {output_dir}")
print(f"\nFiles:")
for f in sorted(output_dir.iterdir()):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name:50s} {size_mb:.1f} MB")