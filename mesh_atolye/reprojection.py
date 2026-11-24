"""Transfer aligned, visible source color to the existing UV atlas."""

import cv2
import numpy as np
from PIL import Image

from .views import read_views


# MeshRender's Y-up mesh is viewed from +Z in the front image.
AZIMUTH = {"front": 0, "right": 270, "back": 180, "left": 90}


def overlap(first, second):
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def signed_distance(mask, radius):
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    return np.clip((inside - outside) / radius, -1.0, 1.0)


def align_reference(rgba, silhouette):
    """Register the reference to a rendered silhouette, keeping the same handedness."""
    height, width = silhouette.shape
    source = rgba[:, :, 3] >= 150
    sx, sy, sw, sh = cv2.boundingRect(source.astype(np.uint8))
    tx, ty, tw, th = cv2.boundingRect(silhouette.astype(np.uint8))
    if min(sw, sh, tw, th) < 24:
        return None, 0.0
    stretch_x, stretch_y = tw / sw, th / sh
    if not 0.62 <= stretch_x / stretch_y <= 1.58:
        return None, 0.0
    initial = np.float32([
        [stretch_x, 0, tx - sx * stretch_x],
        [0, stretch_y, ty - sy * stretch_y],
    ])
    initial_mask = cv2.warpAffine(source.astype(np.uint8), initial, (width, height)) > 0
    score = overlap(initial_mask, silhouette)

    # ECC works on signed-distance fields so that narrow silhouettes still have gradients.
    # The proposed correction is accepted only when it improves actual mask overlap.
    correction = np.eye(2, 3, dtype=np.float32)
    radius = max(8, round(height * 0.025))
    try:
        _, correction = cv2.findTransformECC(
            signed_distance(silhouette, radius), signed_distance(initial_mask, radius),
            correction, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 45, 1e-5),
            None, 5,
        )
    except cv2.error:
        correction = None
    if correction is not None:
        linear = correction[:, :2]
        corners = np.float32([[0, width, 0, width], [0, 0, height, height], [1, 1, 1, 1]])
        displacement = np.max(np.abs(correction @ corners - corners[:2]))
        if 0.82 < np.linalg.det(linear) < 1.22 and displacement < height * 0.11:
            inverse = cv2.invertAffineTransform(correction)
            candidate = inverse @ np.vstack([initial, [0, 0, 1]])
            fitted = cv2.warpAffine(source.astype(np.uint8), candidate, (width, height)) > 0
            fit_score = overlap(fitted, silhouette)
            if fit_score > score:
                initial, score = candidate, fit_score
    if score < 0.52:
        return None, score

    aligned = cv2.warpAffine(rgba, initial, (width, height), flags=cv2.INTER_LINEAR)
    valid = (aligned[:, :, 3] >= 185) & silhouette
    interior = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
    # Feather silhouette edges before projection to avoid tint from background pixels.
    aligned[:, :, 3] = np.round(255 * np.clip((interior - 1.2) / (height * 0.011), 0, 1)).astype(np.uint8)
    return aligned, score


def refine_albedo(folder, low_memory=False):
    import torch
    import trimesh
    from DifferentiableRenderer.MeshRender import MeshRender

    with Image.open(folder / "model.jpg") as image:
        original = np.array(image.convert("RGB"))
    mesh = trimesh.load(folder / "model.obj", force="mesh", process=False)
    if original.shape[0] != original.shape[1] or getattr(mesh.visual, "uv", None) is None:
        raise ValueError("Kaynak ayrıntıları için kare bir doku atlası ve geçerli UV koordinatları gerekiyor.")

    size = original.shape[0]
    resolution = 768 if low_memory else 1024
    renderer = MeshRender(default_resolution=resolution, texture_size=size)
    renderer.load_mesh(mesh)
    views = read_views(folder)
    references = {view: folder / f"view-{view}.png" for view in views}
    if not references:
        references = {"front": folder / "prepared.png"}
    numerator = torch.zeros((size, size, 3), device="cuda")
    denominator = torch.zeros((size, size, 1), device="cuda")
    strongest = torch.zeros_like(denominator)
    fitted_views = []
    with torch.inference_mode():
        for view, path in references.items():
            if not path.is_file():
                continue
            azimuth = AZIMUTH[view]
            visible = renderer.render_alpha(0, azimuth).squeeze().cpu().numpy() > 0
            with Image.open(path) as reference:
                rgba = np.array(reference.convert("RGBA"))
            aligned, score = align_reference(rgba, visible)
            if aligned is None:
                continue
            attributes, cosine, _ = renderer.back_project(Image.fromarray(aligned), 0, azimuth)
            confidence = ((cosine - 0.32) / 0.68).clamp(0, 1).pow(1.6) * attributes[:, :, 3:4]
            confidence *= min(1.0, max(0.0, (score - 0.48) / 0.32))
            importance = confidence.pow(3) * (1.30 if view == "front" else 0.9)
            numerator += attributes[:, :, :3] * importance
            denominator += importance
            torch.maximum(strongest, confidence, out=strongest)
            fitted_views.append({"view": view, "overlap": round(score, 3)})
        if not fitted_views or not torch.any(denominator > 0.02):
            return None, fitted_views, 0.0
        pixels = torch.as_tensor(original, device="cuda", dtype=torch.float32) / 255
        source = numerator / denominator.clamp_min(1e-7)
        blend = 0.92 * strongest
        result = torch.lerp(pixels, source, blend).clamp(0, 1)
        coverage = (blend > 0.5).count_nonzero().item() / renderer.tex_grid.shape[0]
        return Image.fromarray((result * 255).round().to(torch.uint8).cpu().numpy()), fitted_views, coverage
