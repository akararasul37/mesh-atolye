"""Compare a generated mesh's orthographic outlines to the prepared references."""

import cv2
import numpy as np


SIZE = 288
HEIGHT = 244
HORIZONTAL = {
    "front": (0, 1),
    "right": (2, 1),
    "back": (0, -1),
    "left": (2, -1),
}


def reference_masks(image):
    images = image if isinstance(image, dict) else {"front": image}
    masks = {}
    for name, source in images.items():
        mask = (np.asarray(source.getchannel("A")) >= 128).astype(np.uint8)
        x, y, width, height = cv2.boundingRect(mask)
        if min(width, height) < 16 or mask.mean() > 0.94:
            continue
        scale = min(HEIGHT / height, HEIGHT / width)
        transform = np.float32([
            [scale, 0, SIZE / 2 - (x + width / 2) * scale],
            [0, scale, SIZE / 2 - (y + height / 2) * scale],
        ])
        masks[name] = cv2.warpAffine(mask, transform, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 0
    return masks


def projected_mask(vertices, faces, view, reference):
    axis, direction = HORIZONTAL[view]
    xy = np.column_stack((vertices[:, axis] * direction, -vertices[:, 1]))
    minimum = xy.min(axis=0)
    maximum = xy.max(axis=0)
    _, _, width, height = cv2.boundingRect(reference.astype(np.uint8))
    extent = maximum - minimum
    if np.min(extent) <= 0:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    scale = height / extent[1] if height >= width else width / extent[0]
    pixels = np.rint((xy - (minimum + maximum) / 2) * scale + SIZE / 2).astype(np.int32)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for triangle in pixels[faces]:
        cv2.fillConvexPoly(mask, triangle, 1)
    return mask


def silhouette_fit(mesh, masks):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    views = {}
    for name, reference in masks.items():
        rendered = projected_mask(vertices, faces, name, reference)
        best = 0.0
        for y in (-3, 0, 3):
            for x in (-3, 0, 3):
                transform = np.float32([[1, 0, x], [0, 1, y]])
                shifted = cv2.warpAffine(rendered, transform, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 0
                union = np.count_nonzero(shifted | reference)
                score = np.count_nonzero(shifted & reference) / union if union else 0
                best = max(best, score)
        views[name] = float(best)
    if not views:
        return None, views
    scores = list(views.values())
    return 0.8 * sum(scores) / len(scores) + 0.2 * min(scores), views
