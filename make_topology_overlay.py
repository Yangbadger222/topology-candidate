from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path

IMG = Path('/Users/badger/Desktop/2km_alteye.jpg')
TOPO = Path('/Users/badger/.gemini/antigravity/brain/ca5d6b1b-4d96-41fc-960b-7f2d9dc1058b/road_topology_8k.json')
OUT = Path('/Users/badger/Documents/ChatGPT/overhead-ssl/road_topology_overlay.png')

im = Image.open(IMG).convert('RGBA')
with TOPO.open() as f:
    topo = json.load(f)

assert im.size == (topo['resolution']['width'], topo['resolution']['height']), (im.size, topo['resolution'])
layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(layer)

for edge in topo['edges']:
    pts = [tuple(map(float, p)) for p in edge['coordinates']]
    if len(pts) >= 2:
        draw.line(pts, fill=(20, 10, 0, 210), width=18, joint='curve')
        draw.line(pts, fill=(255, 193, 7, 225), width=8, joint='curve')

for node in topo['nodes']:
    x, y = float(node['x']), float(node['y'])
    color = (245, 40, 180, 245) if node['type'] == 'intersection' else (30, 220, 255, 245)
    r = 22 if node['type'] == 'intersection' else 18
    draw.ellipse((x-r-5, y-r-5, x+r+5, y+r+5), fill=(0, 0, 0, 185))
    draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline=(255,255,255,240), width=4)

try:
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 42)
    small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 32)
except OSError:
    font = small = ImageFont.load_default()

panel = (55, 55, 1320, 325)
draw.rounded_rectangle(panel, radius=24, fill=(8, 15, 28, 215), outline=(255,255,255,190), width=3)
draw.text((85, 78), 'Road topology overlay', font=font, fill=(255,255,255,255))
draw.line((90, 180, 250, 180), fill=(255,193,7,225), width=10)
draw.text((275, 158), f"edges: {topo['summary']['total_edges']:,}", font=small, fill=(255,255,255,255))
draw.ellipse((90, 220, 126, 256), fill=(245,40,180,245), outline='white', width=3)
draw.text((145, 210), f"intersections: {topo['summary']['intersection_nodes']:,}", font=small, fill=(255,255,255,255))
draw.ellipse((650, 220, 686, 256), fill=(30,220,255,245), outline='white', width=3)
draw.text((705, 210), f"dead ends: {topo['summary']['dead_end_nodes']:,}", font=small, fill=(255,255,255,255))

out = Image.alpha_composite(im, layer).convert('RGB')
out.save(OUT, quality=95)
print(OUT)
print('size', out.size)
