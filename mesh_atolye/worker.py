import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .config import ENGINE, MODELS, MV_MODEL, RUNTIME
from .views import frontal_reference, prepare_views, read_views


def progress(stage, percent, **details):
    print("MESH_EVENT " + json.dumps({"stage": stage, "progress": percent, **details}), flush=True)


def configure():
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(RUNTIME / "hf"),
        "TORCH_HOME": str(RUNTIME / "torch"),
        "NUMBA_CACHE_DIR": str(RUNTIME / "numba"),
        "U2NET_HOME": str(MODELS),
    })
    sys.path[:0] = [str(ENGINE / "hy3dshape"), str(ENGINE / "hy3dpaint"), str(ENGINE)]


def cuda_budget(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA başlatılamadı. NVIDIA sürücüsünü ve kurulumu kontrol edin.")
    free, total = torch.cuda.mem_get_info()
    if free < 4 * 1024**3:
        raise RuntimeError("En az 4 GiB boş VRAM gerekiyor. GPU kullanan diğer uygulamaları kapatın.")
    reserve = 384 * 1024**2
    torch.cuda.set_per_process_memory_fraction(min(0.87, (free - reserve) / total))
    torch.cuda.reset_peak_memory_stats()


def load_shape(torch, multiview):
    import yaml
    from accelerate import init_empty_weights
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, instantiate_from_config

    directory = MV_MODEL if multiview else MODELS / "hunyuan" / "hunyuan3d-dit-v2-1"
    config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    if multiview:
        # The bundled 2.1 runtime retains the matching 2.0 architecture and view encoder.
        for component in config.values():
            component["target"] = component["target"].replace("hy3dgen.shapegen.", "hy3dshape.", 1)
    state = torch.load(directory / "model.fp16.ckpt", map_location="cpu", weights_only=True, mmap=True)
    with init_empty_weights(include_buffers=False):
        model = instantiate_from_config(config["model"])
        conditioner = instantiate_from_config(config["conditioner"])
    vae = instantiate_from_config(config["vae"])
    model.load_state_dict(state["model"], assign=True)
    conditioner.load_state_dict(state["conditioner"], assign=True)
    vae.load_state_dict(state["vae"], strict=False, assign=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline(
        vae, model, instantiate_from_config(config["scheduler"]), conditioner,
        instantiate_from_config(config["image_processor"]), device="cpu", dtype=torch.float16,
    )
    pipeline.device = torch.device("cuda")
    pipeline.vae.enable_flashvdm_decoder(mc_algo="mc")
    for module in (model, conditioner, vae):
        module.eval().requires_grad_(False)
    return pipeline


def generate_shape(folder, quality, seed, low_memory):
    import torch
    from PIL import Image
    from hy3dshape import DegenerateFaceRemover, FloaterRemover

    cuda_budget(torch)
    progress("shape_load", 12)
    views = read_views(folder)
    pipeline = load_shape(torch, bool(views))
    if low_memory:
        from mmgp import offload

        manager = offload.profile(
            {"model": pipeline.model},
            profile_no=5, budgets={"*": 1700}, quantizeTransformer=False,
            convertWeightsFloatTo=torch.float16, asyncTransfers=False, verboseLevel=1,
        )
    else:
        manager = None
    pipeline.conditioner.to("cuda")
    encode_conditioning = pipeline.encode_cond
    condition = None

    def encode_then_offload(*args, **kwargs):
        nonlocal condition
        if condition is None:
            condition = encode_conditioning(*args, **kwargs)
            pipeline.conditioner.to("cpu")
            torch.cuda.empty_cache()
            if not manager:
                pipeline.model.to("cuda")
        return condition

    pipeline.encode_cond = encode_then_offload

    steps, resolution, chunks = (18, 256, 3500) if quality == "fast" else (35, 384, 4500)
    studio = quality == "studio"
    if studio:
        resolution = 512
    if low_memory:
        resolution = min(resolution, 320 if studio else 256)
        chunks = 2200

    with torch.inference_mode(), ExitStack() as files:
        if views:
            image = {view: files.enter_context(Image.open(folder / f"view-{view}.png")) for view in views}
        else:
            image = files.enter_context(Image.open(frontal_reference(folder, views)))
        masks = {}
        plans = [(seed, steps)]
        if studio:
            from .shape_quality import reference_masks, silhouette_fit

            masks = reference_masks(image)
            if masks:
                plans.extend([((seed + offset) % (2**32), 56) for offset in (1, 2)])
            else:
                plans = [(seed, 56)]

        total_steps = sum(sample_steps for _, sample_steps in plans)
        completed_steps = 0
        samples = []
        for index, (sample_seed, sample_steps) in enumerate(plans, start=1):
            details = {"candidate": index, "candidate_total": len(plans)} if studio else {}

            def update(step, _timestep, _output):
                if step % max(1, sample_steps // 9) == 0:
                    percent = 16 + round(28 * (completed_steps + step + 1) / total_steps)
                    progress("shape", percent, **details)

            progress("shape", 16 + round(28 * completed_steps / total_steps), **details)
            latent = pipeline(
                image=image, num_inference_steps=sample_steps, output_type="latent", callback=update,
                callback_steps=1, generator=torch.Generator(device="cuda").manual_seed(sample_seed),
                guidance_scale=5.0, enable_pbar=False, mc_algo="mc",
            )
            samples.append(latent.cpu())
            completed_steps += sample_steps
        condition = None
        del latent
        if manager:
            manager.unload_all()
        else:
            pipeline.model.to("cpu")
        torch.cuda.empty_cache()
        pipeline.vae.to("cuda")
        best_mesh, best_score, best_index = None, -1.0, None
        results = []
        for index, latent in enumerate(samples, start=1):
            percent = 47 + round(7 * (index - 1) / len(samples))
            details = {"candidate": index, "candidate_total": len(samples)} if studio else {}
            progress("surface", percent, **details)
            mesh = pipeline._export(
                latent.to("cuda"), output_type="trimesh", num_chunks=chunks,
                octree_resolution=resolution, mc_algo="mc", enable_pbar=False,
            )[0]
            if mesh is None or len(mesh.faces) == 0:
                results.append({"candidate": index, "score": None, "valid": False})
                continue
            mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
            if len(mesh.faces) == 0:
                results.append({"candidate": index, "score": None, "valid": False})
                continue
            score, per_view = silhouette_fit(mesh, masks) if masks else (None, {})
            rank = score if score is not None else 0.0
            if rank > best_score:
                best_mesh, best_score, best_index = mesh, rank, index
            if studio:
                results.append({
                    "candidate": index, "score": round(score, 4) if score is not None else None,
                    "views": {name: round(value, 4) for name, value in per_view.items()}, "valid": True,
                })
                print(f"Studio candidate {index}: silhouette fit {score}", flush=True)
        if best_mesh is None:
            raise ValueError("Model yüzey üretemedi. Nesnenin belirgin olduğu bir görsel deneyin.")
        best_mesh.export(folder / "geometry.glb", include_normals=True)
        best_mesh.export(folder / "geometry.obj")
    details = {}
    if studio:
        details = {
            "studio_candidates": results, "studio_selected": best_index,
            "studio_shape_steps": [sample_steps for _, sample_steps in plans],
            "studio_surface_resolution": resolution,
        }
    progress("shape_done", 56, shape_vram_mib=round(torch.cuda.max_memory_reserved() / 1024**2), **details)


def export_pbr(folder):
    import numpy as np
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial

    model = folder / "model.obj"
    mesh = trimesh.load(model, force="mesh", process=False)
    if len(mesh.faces) == 0 or getattr(mesh.visual, "uv", None) is None:
        raise ValueError("Kaplamalı modelde geçerli yüzey veya UV bulunamadı.")
    with (
        Image.open(folder / "model.jpg") as diffuse,
        Image.open(folder / "model_metallic.jpg") as metallic,
        Image.open(folder / "model_roughness.jpg") as roughness,
    ):
        base_color = diffuse.convert("RGB")
        metal = np.asarray(metallic.convert("L"))
        rough = np.asarray(roughness.convert("L"))
        mr = Image.fromarray(np.stack([np.full_like(metal, 255), rough, metal], axis=-1))
    mesh.visual.material = PBRMaterial(
        name="Hunyuan Paint PBR", baseColorTexture=base_color, metallicRoughnessTexture=mr,
        metallicFactor=1.0, roughnessFactor=1.0,
    )
    mesh.export(folder / "model.glb", include_normals=True)
    with ZipFile(folder / "model-obj.zip", "w", ZIP_DEFLATED, compresslevel=3) as bundle:
        for path in sorted(folder.glob("model.*")) + sorted(folder.glob("model_*.jpg")):
            if path.suffix != ".glb":
                bundle.write(path, arcname=path.name)


def generate_pbr(folder, quality, low_memory):
    import torch
    import trimesh
    from mmgp import offload
    from PIL import Image
    from unittest.mock import patch

    cuda_budget(torch)
    progress("paint_load", 61)
    import mesh_inpaint_processor
    from torchvision_fix import apply_fix

    apply_fix()
    sys.modules["DifferentiableRenderer.mesh_inpaint_processor"] = mesh_inpaint_processor
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    studio = quality == "studio"
    reference = quality in {"reference", "studio"}
    if studio:
        image_resolution = 512 if low_memory else 768
    else:
        image_resolution = 384 if quality == "fast" or low_memory else 512
    config = Hunyuan3DPaintConfig(max_num_view=6, resolution=image_resolution)
    config.device = "cpu"
    detailed = quality != "fast" and not low_memory
    config.render_size = 1536 if detailed else 1024
    config.texture_size = 4096 if detailed else (1536 if low_memory else 2048)
    config.realesrgan_ckpt_path = str(MODELS / "RealESRGAN_x4plus.pth")
    config.dino_ckpt_path = str(MODELS / "dino")
    config.multiview_cfg_path = str(ENGINE / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    with patch("utils.multiview_utils.huggingface_hub.snapshot_download", return_value=str(MODELS / "hunyuan")):
        paint = Hunyuan3DPaintPipeline(config)
    multiview = paint.models["multiview_model"]
    core = multiview.pipeline
    core.vae.enable_slicing()
    core.vae.enable_tiling()
    weight_budget = 1200 if low_memory else (1600 if studio else 2100)
    manager = offload.profile(
        offload.extract_models(multiview), profile_no=5, budgets={"*": weight_budget},
        quantizeTransformer=False, convertWeightsFloatTo=torch.float16,
        asyncTransfers=not low_memory, verboseLevel=1,
    )
    # The upstream random generator uses the pipeline's device property.
    type(core).device = property(lambda self: torch.device("cuda"))
    type(core)._execution_device = property(lambda self: torch.device("cuda"))
    paint_steps = 26 if low_memory else 28
    if studio:
        denoise = core.denoise

        def paint_step(_pipeline, step, _timestep, tensors):
            if step % 3 == 0 or step + 1 == paint_steps:
                progress("paint", 71 + round(13 * (step + 1) / paint_steps),
                         paint_step=step + 1, paint_step_total=paint_steps)
            return tensors

        def studio_denoise(*args, **kwargs):
            kwargs.update(num_inference_steps=paint_steps, callback_on_step_end=paint_step)
            return denoise(*args, **kwargs)

        core.denoise = studio_denoise
    upsampler = paint.models["super_model"].upsampler
    upsampler.tile = 128
    upsampler.model.to("cpu")

    def paint_views(*args, **kwargs):
        progress("paint", 71)
        result = multiview(*args, **kwargs)
        manager.unload_all()
        torch.cuda.empty_cache()
        upsampler.model.to("cuda")
        progress("bake", 87)
        return result

    paint.models["multiview_model"] = paint_views
    mesh = trimesh.load(folder / "geometry.glb", force="mesh", process=False)
    face_count = (220_000 if studio else 150_000) if detailed else 100_000
    if len(mesh.faces) > face_count:
        mesh = mesh.simplify_quadric_decimation(face_count=face_count)
    mesh.export(folder / "paint-input.obj")
    progress("uv", 66)
    with Image.open(frontal_reference(folder, read_views(folder))) as image:
        paint(
            mesh_path=str(folder / "paint-input.obj"), image_path=image,
            output_mesh_path=str(folder / "model.obj"), use_remesh=False, save_glb=False,
        )
    progress("export", 88 if reference else 96)
    export_pbr(folder)
    details = {"studio_paint_steps": paint_steps, "studio_paint_resolution": image_resolution} if studio else {}
    progress("paint_done", 90 if reference else 99,
             paint_vram_mib=round(torch.cuda.max_memory_reserved() / 1024**2), **details)


def refine_texture(folder, low_memory):
    import torch
    from .reprojection import refine_albedo

    cuda_budget(torch)
    progress("reference", 92)
    image, views, coverage = refine_albedo(folder, low_memory)
    if image is not None:
        albedo = folder / "model.jpg"
        temporary = folder / "model-color.tmp"
        image.save(temporary, format="JPEG", quality=96, subsampling=0)
        temporary.replace(albedo)
        export_pbr(folder)
    progress(
        "reference_done", 99, reference_applied=image is not None, reference_views=views,
        reference_coverage=round(coverage, 3),
        reference_vram_mib=round(torch.cuda.max_memory_reserved() / 1024**2),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "shape", "paint", "refine"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--quality", choices=["fast", "quality", "reference", "studio"], default="quality")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remove-background", action="store_true")
    parser.add_argument("--layout", choices=["auto", "single", "turnaround"], default="auto")
    parser.add_argument("--profile", choices=["left", "right"], default="right")
    parser.add_argument("--low-memory", action="store_true")
    args = parser.parse_args()
    configure()
    if args.stage == "prepare":
        progress("background", 5, removing=args.remove_background)
        prepare_views(args.folder, args.remove_background, args.layout, args.profile, progress)
        return 0
    import torch

    os.environ.setdefault("OMP_NUM_THREADS", "6")
    torch.set_num_threads(6)
    try:
        if args.stage == "shape":
            generate_shape(args.folder, args.quality, args.seed, args.low_memory)
        elif args.stage == "paint":
            generate_pbr(args.folder, args.quality, args.low_memory)
        else:
            refine_texture(args.folder, args.low_memory)
    except torch.cuda.OutOfMemoryError:
        import traceback

        traceback.print_exc()
        return 42
    return 0


if __name__ == "__main__":
    sys.exit(main())
