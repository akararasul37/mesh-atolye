import json

import numpy as np
from PIL import Image, ImageChops

from .config import MODELS


def mask_bounds(image):
    return image.getchannel("A").point(lambda value: 255 if value > 24 else 0).getbbox()


def turnaround_lanes(image):
    """Find three similarly sized subjects, separated by vertical gutters."""
    width, height = image.size
    if width < 1.22 * height:
        return None
    mask = np.asarray(image.getchannel("A")) > 96
    occupied = mask.sum(axis=0) >= max(4, round(height * 0.015))
    transitions = np.flatnonzero(np.diff(np.r_[False, occupied, False]))
    spans = []
    for start, end in zip(transitions[::2], transitions[1::2]):
        if spans and start - spans[-1][1] <= width * 0.018:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))

    panels = []
    for start, end in spans:
        rows, _ = np.nonzero(mask[:, start:end])
        if rows.size >= width * height * 0.018 and end - start >= width * 0.045:
            panels.append((start, end, rows.min(), rows.max() + 1, rows.size))
    if len(panels) != 3:
        return None
    heights = [bottom - top for _, _, top, bottom, _ in panels]
    overlap = min(panel[3] for panel in panels) - max(panel[2] for panel in panels)
    if min(heights) < height * 0.47 or overlap < max(heights) * 0.73:
        return None
    if any(b[0] - a[1] < width * 0.018 for a, b in zip(panels, panels[1:])):
        return None

    cuts = [0, *((a[1] + b[0]) // 2 for a, b in zip(panels, panels[1:])), width]
    return [(int(left), 0, int(right), height) for left, right in zip(cuts, cuts[1:])]


def square_subject(image, margin=0):
    bounds = mask_bounds(image)
    if not bounds:
        raise ValueError("Görselde görünür bir nesne bulunamadı. Başka bir görsel deneyin.")
    image = image.crop(bounds)
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    side = round(max(image.size) * (1 + 2 * margin))
    square = Image.new("RGBA", (side, side))
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return square


def background_remover():
    import onnxruntime
    from rembg import remove
    from rembg.sessions.u2net import U2netSession

    if not (MODELS / "u2net.onnx").is_file():
        raise RuntimeError("Arka plan modeli eksik. kurulum.bat dosyasını çalıştırın.")

    class LocalU2netSession(U2netSession):
        @classmethod
        def download_models(cls, *args, **kwargs):
            return MODELS / "u2net.onnx"

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 4
    session = LocalU2netSession("u2net", options)

    def segment(image):
        return remove(image, session=session, post_process_mask=True)

    return segment


def frontal_reference(folder, views):
    if not views:
        return folder / "prepared.png"
    return folder / "view-front.png"


def read_views(folder):
    manifest = folder / "views.json"
    return json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else []


def prepare_views(folder, cleanup, layout, profile, progress):
    with Image.open(folder / "source.png") as source:
        image = source.convert("RGBA")
    opaque = image.getchannel("A").getextrema()[0] > 24
    check_sheet = layout == "turnaround" or (layout == "auto" and image.width >= image.height * 1.22)
    segment = background_remover() if opaque and (cleanup or check_sheet) else None
    masked = segment(image) if segment else image
    lanes = turnaround_lanes(masked) if check_sheet else None
    if lanes is None:
        if layout == "turnaround":
            raise ValueError("Üç ayrı görünüş bulunamadı. Ön, yan ve arka görselleri arasında boşluk bırakın.")
        square_subject(masked if cleanup else image).save(folder / "prepared.png")
        progress("background", 9, input_layout="single")
        return

    from .config import multiview_ready

    if not multiview_ready():
        raise RuntimeError("Çok görünüş modeli eksik. kurulum.bat dosyasını çalıştırın.")
    progress("views", 8)
    views = ["front", profile, "back"]
    for view, lane in zip(views, lanes):
        crop = image.crop(lane)
        coarse = masked.crop(lane)
        if segment:
            refined = segment(crop)
            # The local pass sees details missed at the full sheet's lower model resolution.
            if mask_bounds(refined):
                refined.putalpha(ImageChops.lighter(refined.getchannel("A"), coarse.getchannel("A")))
                coarse = refined
        square_subject(coarse, margin=0.06).save(folder / f"view-{view}.png")
    (folder / "views.json").write_text(json.dumps(views), encoding="utf-8")
    progress("views", 10, input_layout="turnaround", views=views)
