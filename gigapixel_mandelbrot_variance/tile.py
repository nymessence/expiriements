import os
import math
from PIL import Image
from tqdm import tqdm

TILE_SIZE = 256
TILE_PIXEL_DIM = 8192
GRID_SIZE = 4  # 4x4 grid
OUTPUT_DIR = "tiles"
TILES_MD = "tiles.md"

# Step 1: Read tile layout from tiles.md
def parse_tile_map():
    with open(TILES_MD, "r") as f:
        lines = [line.strip() for line in f if "|" in line and not "---" in line]
    tile_map = []
    for line in lines:
        parts = [p.strip() for p in line.split("|")[1:-1]]  # trim leading and trailing "|"
        tile_map.append([int(p) for p in parts])
    return tile_map  # 2D list [row][col] from top to bottom

# Step 2: Load all 16 tile images
def load_tiles(tile_map):
    rows = []
    for row in tile_map:
        row_imgs = []
        for tile_num in row:
            path = f"mandelbrot_tile_{tile_num}.jpg"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")
            img = Image.open(path)
            if img.size != (TILE_PIXEL_DIM, TILE_PIXEL_DIM):
                raise ValueError(f"{path} is not {TILE_PIXEL_DIM}×{TILE_PIXEL_DIM}")
            row_imgs.append(img)
        rows.append(row_imgs)
    return rows  # 4x4 grid of Image objects

# Step 3: Composite tiles on demand (tile by tile, not stitching whole image)
def get_full_image_region(x, y, zoom, tile_grid):
    scale = 2 ** (MAX_ZOOM - zoom)
    size_at_zoom = FULL_SIZE // scale
    x0 = x * TILE_SIZE * scale
    y0 = y * TILE_SIZE * scale
    x1 = x0 + TILE_SIZE * scale
    y1 = y0 + TILE_SIZE * scale

    region = Image.new("RGB", (TILE_SIZE * scale, TILE_SIZE * scale))

    # Determine which 8192×8192 tiles are needed
    tile_x_start = x0 // TILE_PIXEL_DIM
    tile_y_start = y0 // TILE_PIXEL_DIM
    tile_x_end = (x1 - 1) // TILE_PIXEL_DIM
    tile_y_end = (y1 - 1) // TILE_PIXEL_DIM

    for ty in range(tile_y_start, tile_y_end + 1):
        for tx in range(tile_x_start, tile_x_end + 1):
            tile_img = tile_grid[ty][tx]

            # Crop part of tile
            crop_x0 = max(0, x0 - tx * TILE_PIXEL_DIM)
            crop_y0 = max(0, y0 - ty * TILE_PIXEL_DIM)
            crop_x1 = min(TILE_PIXEL_DIM, x1 - tx * TILE_PIXEL_DIM)
            crop_y1 = min(TILE_PIXEL_DIM, y1 - ty * TILE_PIXEL_DIM)

            region_x = (tx * TILE_PIXEL_DIM - x0) + crop_x0
            region_y = (ty * TILE_PIXEL_DIM - y0) + crop_y0

            part = tile_img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            region.paste(part, (region_x, region_y))

    return region.resize((TILE_SIZE, TILE_SIZE), resample=Image.LANCZOS)

# Step 4: Generate tiles per zoom level
def generate_tiles(tile_grid):
    for zoom in range(MAX_ZOOM + 1):
        scale = 2 ** (MAX_ZOOM - zoom)
        size_at_zoom = FULL_SIZE // scale
        tiles_x = math.ceil(size_at_zoom / TILE_SIZE)
        tiles_y = math.ceil(size_at_zoom / TILE_SIZE)
        out_dir = os.path.join(OUTPUT_DIR, f"z{zoom}")
        os.makedirs(out_dir, exist_ok=True)
        for y in tqdm(range(tiles_y), desc=f"Zoom {zoom}"):
            for x in range(tiles_x):
                tile = get_full_image_region(x, y, zoom, tile_grid)
                tile_path = os.path.join(out_dir, f"{x}_{y}.jpg")
                tile.save(tile_path, "JPEG", quality=90)

# ---- MAIN ----
if __name__ == "__main__":
    print("Parsing tile layout...")
    tile_map = parse_tile_map()  # top-down 2D grid

    print("Loading tile images...")
    tile_grid = load_tiles(tile_map)  # top-down 2D grid of Images

    FULL_SIZE = GRID_SIZE * TILE_PIXEL_DIM  # 32768
    MAX_ZOOM = int(math.log2(FULL_SIZE // TILE_SIZE))  # 7

    print(f"Generating Deep Zoom tiles ({TILE_SIZE}x{TILE_SIZE})...")
    generate_tiles(tile_grid)

    print("✅ Done. Ready for OpenSeadragon!")
