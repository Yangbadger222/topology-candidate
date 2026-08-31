from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def save_contact_sheet(images: list[tuple[str, Image.Image]], output: str | Path, cell_size=(256, 256)) -> None:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    w, h = cell_size
    canvas = Image.new("RGB", (w * len(images), h + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(images):
        canvas.paste(image.convert("RGB").resize((w, h)), (i * w, 24))
        draw.text((i * w + 4, 4), label, fill="black")
    canvas.save(output, quality=92)

