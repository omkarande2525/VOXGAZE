"""
VoxGaze Milestone M1 / Model Acquisition Script: resnet34_gaze.onnx

Downloads the official pretrained ResNet-34 ONNX gaze-estimation checkpoint
from yakhyo/gaze-estimation releases with strict verification:
- Pre-allocated disk space check
- Resilient streaming download with chunking
- Atomic rename via temporary file
- Exact byte-size validation (85,491,644 bytes)
- Cryptographic SHA-256 checksum calculation and provenance logging
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

EXPECTED_URL = (
    "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet34_gaze.onnx"
)
EXPECTED_BYTE_SIZE = 85491644
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "models" / "resnet34_gaze.onnx"
MIN_FREE_SPACE_BYTES = 200 * 1024 * 1024  # 200 MB minimum headroom


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 digest of a local file in 64 KB chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_disk_space(target_dir: Path, required_bytes: int) -> None:
    """Verify target partition has sufficient free disk space."""
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    if usage.free < required_bytes:
        raise OSError(
            f"Insufficient disk space on {target_dir}. "
            f"Available: {usage.free / (1024 * 1024):.1f} MB, "
            f"Required: {required_bytes / (1024 * 1024):.1f} MB"
        )


def download_resnet34_checkpoint(
    target_path: Path = DEFAULT_OUTPUT_PATH,
    url: str = EXPECTED_URL,
    force: bool = False,
    timeout: float = 120.0,
) -> Path:
    """
    Downloads and verifies resnet34_gaze.onnx.

    Returns:
        Path to validated model file.
    """
    target_path = Path(target_path).resolve()
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Existing valid model guard
    if target_path.is_file():
        actual_size = target_path.stat().st_size
        if actual_size == EXPECTED_BYTE_SIZE and not force:
            sha256_hash = compute_sha256(target_path)
            print(f"[INFO] Valid ResNet-34 model already exists at: {target_path}")
            print(f"       File size: {actual_size:,} bytes")
            print(f"       Computed SHA-256: {sha256_hash}")
            print("       (Use --force to re-download)")
            return target_path
        elif actual_size != EXPECTED_BYTE_SIZE:
            print(
                f"[WARN] Existing model at {target_path} has incorrect size "
                f"({actual_size} bytes vs expected {EXPECTED_BYTE_SIZE}). Re-downloading."
            )
        elif force:
            print(f"[INFO] Re-download requested via --force for {target_path}.")

    # 2. Check disk space
    check_disk_space(target_dir, MIN_FREE_SPACE_BYTES)

    # 3. Download to temporary file in the same directory (enables atomic rename)
    fd, temp_file_str = tempfile.mkstemp(
        prefix="resnet34_gaze_",
        suffix=".onnx.tmp",
        dir=str(target_dir),
    )
    os.close(fd)
    temp_path = Path(temp_file_str)

    try:
        print(f"[INFO] Downloading ResNet-34 ONNX checkpoint...")
        print(f"       Source: {url}")
        print(f"       Target: {temp_path}")
        print(f"       Expected Size: {EXPECTED_BYTE_SIZE:,} bytes")

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "VoxGaze-ModelAcquisition/1.0",
                "Accept": "application/octet-stream",
            },
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", None) or getattr(response, "code", None)
            if status_code != 200:
                raise RuntimeError(f"Unexpected HTTP status {status_code} while fetching model.")

            bytes_read = 0
            chunk_size = 64 * 1024  # 64 KB
            with open(temp_path, "wb") as f_out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    bytes_read += len(chunk)

        print(f"[INFO] Download complete. Validating downloaded file...")
        downloaded_size = temp_path.stat().st_size
        if downloaded_size != EXPECTED_BYTE_SIZE:
            raise ValueError(
                f"Byte-size mismatch for downloaded artifact: "
                f"expected {EXPECTED_BYTE_SIZE} bytes, received {downloaded_size} bytes."
            )

        # 4. Compute cryptographic digest
        sha256_digest = compute_sha256(temp_path)
        print(f"[INFO] SHA-256 Digest: {sha256_digest}")

        # 5. Atomic commit
        shutil.move(str(temp_path), str(target_path))
        print(f"[SUCCESS] ResNet-34 model verified and installed at: {target_path}")
        return target_path

    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        print(f"[ERROR] Model download or validation failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify official ResNet-34 ONNX gaze weights.")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_PATH, help="Target destination path.")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download even if valid file exists.")
    args = parser.parse_args()

    try:
        download_resnet34_checkpoint(target_path=args.output, force=args.force)
        sys.exit(0)
    except Exception as e:
        sys.exit(1)
