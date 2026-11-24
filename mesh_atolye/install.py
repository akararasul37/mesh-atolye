import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ENGINE, FRONTEND, INSTALL, MODELS, ROOT, RUNTIME

ENGINE_REV = "8244fbb03d5a93cd3638376d621caddd6c06d23b"
ENGINE_SHA256 = "3203b1108727faa148da2f15fe6f279d22d305e96bb02539213b4c31ddccfcd3"
WHEEL_ROOT = "https://raw.githubusercontent.com/visualbruno/ComfyUI-Hunyuan3d-2-1/a0b26fbb94c9a9d32720526c52e64f87e05d2b53/hy3dpaint"
WHEELS = [
    ("custom_rasterizer", "custom_rasterizer-0.1-cp311-cp311-win_amd64.whl",
     "630714b781b3b5bc14748818211f146bc8a7ad701c014913380af5abd9a42062"),
    ("DifferentiableRenderer", "mesh_inpaint_processor-0.0.0-cp311-cp311-win_amd64.whl",
     "92be6bf5d697bb784b65a5e404c50fc811a098a569d93afdfe37aee230a7d0f9"),
]
SEGMENT = 256 * 1024 * 1024


def matches(path: Path, size=None, sha256=None) -> bool:
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        return False
    if sha256:
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest() == sha256
    return True


def curl(url, path, *flags):
    subprocess.run(
        ["curl.exe", "-fsSL", "--retry", "3", "--connect-timeout", "20",
         "--speed-time", "35", "--speed-limit", "1024", *flags, "-o", str(path), url],
        check=True,
    )


def download(url: str, path: Path, size=None, sha256=None):
    if matches(path, size, sha256):
        return
    print(f"İndiriliyor: {path.name}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".download")
    if size is not None and size > 2 * SEGMENT and "huggingface.co/" in url:
        ranges = [(offset, min(size, offset + SEGMENT)) for offset in range(0, size, SEGMENT)]

        def segment(bounds):
            start, end = bounds
            part = partial.with_suffix(f".range-{start}")
            if not matches(part, end - start):
                temporary = part.with_suffix(part.suffix + ".tmp")
                # Include the range in the URL so cached range responses stay distinct.
                curl(f"{url}?download=1&segment={start}", temporary, "--range", f"{start}-{end - 1}")
                if not matches(temporary, end - start):
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"{path.name}: eksik veri aralığı {start}-{end - 1}")
                temporary.replace(part)
            return part

        with ThreadPoolExecutor(max_workers=5) as pool:
            parts = list(pool.map(segment, ranges))
        with partial.open("wb") as target:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if not matches(partial, size, sha256):
            partial.unlink(missing_ok=True)
            for part in parts:
                part.unlink(missing_ok=True)
            raise RuntimeError(f"İndirme doğrulanamadı: {path.name}")
        partial.replace(path)
        for part in parts:
            part.unlink()
    else:
        curl(url, partial)
        if not matches(partial, size, sha256):
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"İndirme doğrulanamadı: {path.name}")
        partial.replace(path)


def install_engine():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    marker = ENGINE / ".revision"
    if marker.is_file() and marker.read_text(encoding="ascii") == ENGINE_REV:
        return
    archive_path = RUNTIME / "hunyuan.tar.gz"
    download(f"https://codeload.github.com/YanWenKun/Hunyuan3D-2.1/tar.gz/{ENGINE_REV}", archive_path, sha256=ENGINE_SHA256)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = member.name.partition("/")[2]
            if not member.isfile() or not name:
                continue
            if name.split("/")[0] not in {"hy3dshape", "hy3dpaint", "assets", "LICENSE", "Notice.txt", "torchvision_fix.py"}:
                continue
            destination = ENGINE / name
            if not destination.resolve().is_relative_to(ENGINE.resolve()):
                raise RuntimeError(f"Geçersiz arşiv girdisi: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    marker.write_text(ENGINE_REV, encoding="ascii")
    archive_path.unlink()


def install_renderer():
    paths = []
    for folder, name, digest in WHEELS:
        path = RUNTIME / "wheels" / name
        download(f"{WHEEL_ROOT}/{folder}/dist/{name}", path, sha256=digest)
        paths.append(path)
    subprocess.run([sys.executable, "-m", "pip", "install", *map(str, paths)], check=True)


def install_three():
    vendor = FRONTEND / "vendor"
    modules = (
        "build/three.module.js", "build/three.core.js",
        "examples/jsm/controls/OrbitControls.js", "examples/jsm/loaders/GLTFLoader.js",
        "examples/jsm/utils/BufferGeometryUtils.js", "examples/jsm/utils/SkeletonUtils.js",
        "examples/jsm/environments/RoomEnvironment.js",
        "LICENSE",
    )
    if all((vendor / name).is_file() for name in modules):
        return
    package = RUNTIME / "three-0.183.2.tgz"
    download("https://registry.npmjs.org/three/-/three-0.183.2.tgz", package)
    with tarfile.open(package, "r:gz") as archive:
        for name in modules:
            destination = vendor / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(f"package/{name}") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    package.unlink()


def install_weights(geometry_only=False):
    items = json.loads((ROOT / "weights.json").read_text(encoding="utf-8-sig"))
    for item in items:
        name = item["path"]
        if geometry_only and not (name.startswith(("hunyuan/hunyuan3d-dit", "hunyuan-mv/")) or name in {
            "hunyuan/LICENSE", "hunyuan/Notice.txt", "u2net.onnx",
        }):
            continue
        download(item["url"], MODELS / name, item.get("size"), item.get("sha256"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA destekli NVIDIA ekran kartını başlatamadı.")
    print(f"{torch.cuda.get_device_name()} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)
    install_engine()
    install_three()
    install_weights(args.geometry_only)
    readiness = {"geometry": True, "pbr": False, "gpu": torch.cuda.get_device_name()}
    if not args.geometry_only:
        install_renderer()
        import custom_rasterizer
        import mesh_inpaint_processor

        positions = torch.tensor([[[-1., -1., 0., 1.], [1., -1., 0., 1.], [0., 1., 0., 1.]]], device="cuda")
        triangles = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        custom_rasterizer.rasterize(positions, triangles, (16, 16))
        torch.cuda.synchronize()
        readiness["pbr"] = True
    elif INSTALL.exists():
        readiness["pbr"] = json.loads(INSTALL.read_text(encoding="utf-8")).get("pbr", False)
    temporary = INSTALL.with_suffix(".tmp")
    temporary.write_text(json.dumps(readiness, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, INSTALL)


if __name__ == "__main__":
    main()

