from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import json, os

IMG = '/var/folders/w9/l2kswtfn3fs2kslkvv5vg9sh0000gn/T/codex-clipboard-996461f8-455e-480d-85d7-1ec45c3c8cd9.png'
OUT = 'outputs/rngdetpp_single'
os.makedirs(OUT, exist_ok=True)

# Approximate graph in image pixel coordinates. Coordinates follow RNGDet++'s
# (x, y) convention; this is a hand-verified fallback for the supplied image
# because the official checkpoint cannot run in this CPU-only environment.
nodes = {
    'n00': (385, 18), 'n01': (385, 125), 'n02': (390, 285),
    'n03': (392, 490), 'n04': (394, 690), 'n05': (395, 895), 'n06': (395, 1085),
    'n10': (255, 180), 'n11': (145, 285), 'n12': (70, 490), 'n13': (65, 690),
    'n14': (95, 895), 'n15': (100, 1060),
    'n20': (535, 285), 'n21': (625, 205), 'n22': (730, 165), 'n23': (820, 165),
    'n24': (1005, 260), 'n25': (790, 470), 'n26': (905, 690), 'n27': (1020, 890),
    'n28': (1005, 1060), 'n30': (540, 490), 'n31': (650, 690), 'n32': (650, 895),
}
edges = [
    ('n00','n01'), ('n01','n02'), ('n02','n03'), ('n03','n04'), ('n04','n05'), ('n05','n06'),
    ('n10','n02'), ('n11','n02'), ('n12','n03'), ('n13','n04'), ('n14','n05'), ('n15','n06'),
    ('n02','n20'), ('n20','n21'), ('n21','n22'), ('n22','n23'), ('n23','n24'),
    ('n03','n30'), ('n30','n25'), ('n25','n26'), ('n04','n31'), ('n31','n26'),
    ('n05','n32'), ('n32','n27'), ('n27','n28'), ('n04','n26'),
]

graph = {
    'model': 'RNGDet++ graph schema (single-image fallback)',
    'image': IMG, 'coordinate_system': 'pixel_xy',
    'nodes': [{'id': k, 'x': v[0], 'y': v[1], 'degree': sum(k in e for e in edges)} for k,v in nodes.items()],
    'edges': [{'source': a, 'target': b} for a,b in edges],
    'notes': [
        'The supplied image is a blurred campus/roof-top view rather than a CityScale/SpaceNet road tile.',
        'Official RNGDet++ inference was not executable here: checkpoint is not bundled and this host has no PyTorch/CUDA runtime.',
        'Nodes/edges are an approximate centerline graph of visually continuous paved corridors; inspect overlay before using metrically.'
    ]
}
with open(os.path.join(OUT,'road_graph.json'),'w') as f:
    json.dump(graph, f, ensure_ascii=False, indent=2)

im = Image.open(IMG).convert('RGB')
im = ImageEnhance.Brightness(im).enhance(0.72)
draw = ImageDraw.Draw(im, 'RGBA')
for a,b in edges:
    draw.line([nodes[a], nodes[b]], fill=(0, 235, 255, 225), width=6)
for k,(x,y) in nodes.items():
    draw.ellipse((x-9,y-9,x+9,y+9), fill=(235,40,40,245), outline=(255,255,255,255), width=2)
    draw.text((x+11,y-10), k, fill=(255,255,0,255))
im.save(os.path.join(OUT,'road_graph_overlay.png'))

# A compact edge list for reading without JSON tooling.
with open(os.path.join(OUT,'road_graph_edges.csv'),'w') as f:
    f.write('source,target,source_xy,target_xy\n')
    for a,b in edges:
        f.write(f'{a},{b},"{nodes[a][0]} {nodes[a][1]}","{nodes[b][0]} {nodes[b][1]}"\n')
print(os.path.join(OUT,'road_graph_overlay.png'))
print(os.path.join(OUT,'road_graph.json'))
