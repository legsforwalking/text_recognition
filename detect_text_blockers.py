"""
Detect street-name text in Berlin K5 raster tiles and write blocker polygons.

The script reads street names from a GeoPackage layer, OCRs georeferenced TIFF
tiles, matches OCR text against the street-name vocabulary, and writes tight
convex-hull blocker polygons to a GeoPackage layer named "text_blockers".

Recommended setup:
    conda activate photo_ai

EasyOCR will use CUDA automatically when a CUDA-enabled torch build is present.
If EasyOCR is not installed, the script can fall back to pytesseract, but a
local Tesseract executable must be installed and on PATH.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import itertools
import math
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Keep common Windows conda geospatial installs from failing before imports.
DEFAULT_PROJ_LIB = Path(sys.prefix) / "Library" / "share" / "proj"
if not DEFAULT_PROJ_LIB.exists():
    DEFAULT_PROJ_LIB = Path.home() / ".conda" / "envs" / "autogis" / "Library" / "share" / "proj"
if "PROJ_LIB" not in os.environ and DEFAULT_PROJ_LIB.exists():
    os.environ["PROJ_LIB"] = str(DEFAULT_PROJ_LIB)
if "PROJ_DATA" not in os.environ and DEFAULT_PROJ_LIB.exists():
    os.environ["PROJ_DATA"] = str(DEFAULT_PROJ_LIB)

# Avoid an OpenMP runtime clash seen when torch and geospatial wheels coexist.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import MultiPoint, Polygon


_WORKER_ARGS = None
_WORKER_ALIASES = None
_WORKER_READER = None


STREET_SUFFIXES = (
    "strasse",
    "str",
    "weg",
    "allee",
    "platz",
    "damm",
    "ufer",
    "chaussee",
    "ring",
    "steig",
    "pfad",
    "promenade",
    "bruecke",
    "brücke",
    "gasse",
)


@dataclass(frozen=True)
class OCRBox:
    text: str
    confidence: float
    polygon_px: tuple[tuple[float, float], ...]
    tile: str

    @property
    def center(self) -> tuple[float, float]:
        xs = [p[0] for p in self.polygon_px]
        ys = [p[1] for p in self.polygon_px]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @property
    def angle(self) -> float:
        pts = list(self.polygon_px)
        if len(pts) < 2:
            return 0.0
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        angle = math.degrees(math.atan2(dy, dx))
        while angle < -90:
            angle += 180
        while angle > 90:
            angle -= 180
        return angle


def normalize_text(value: str) -> str:
    value = value.lower()
    value = (
        value.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    value = re.sub(r"\bstr\.", "strasse", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_attached_suffixes(value: str) -> str:
    words = []
    for word in normalize_text(value).split():
        split_word = False
        for suffix in STREET_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                stem = word[: -len(suffix)]
                words.extend([stem, "strasse" if suffix in {"str", "strasse"} else suffix])
                split_word = True
                break
        if not split_word:
            words.append(word)
    return " ".join(words)


def street_aliases(street_name: str) -> set[str]:
    expanded = split_attached_suffixes(street_name)
    compact = normalize_text(street_name)
    aliases = {expanded, compact}
    aliases.add(expanded.replace(" strasse", "strasse"))
    aliases.add(expanded.replace(" strasse", " str"))
    return {a for a in aliases if a}


def read_street_vocabulary(gpkg: Path, layer: str, name_column: str) -> dict[str, str]:
    streets = gpd.read_file(gpkg, layer=layer, columns=[name_column], engine="fiona")
    names = sorted({str(v).strip() for v in streets[name_column].dropna() if str(v).strip()})
    aliases: dict[str, str] = {}
    for name in names:
        for alias in street_aliases(name):
            aliases.setdefault(alias, name)
    return aliases


def iter_windows(width: int, height: int, chip_size: int, overlap: int) -> Iterable[Window]:
    step = chip_size - overlap
    if step <= 0:
        raise ValueError("--overlap must be smaller than --chip-size")
    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            win_width = min(chip_size, width - col_off)
            win_height = min(chip_size, height - row_off)
            yield Window(col_off, row_off, win_width, win_height)


def raster_window_to_image(
    src: rasterio.DatasetReader,
    window: Window,
    image_scale: float,
    remove_map_lines: bool,
    line_kernel_size: int,
) -> np.ndarray:
    arr = src.read(1, window=window)
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Slight upscaling helps OCR on small cartographic type; lower values run faster.
    if image_scale != 1.0:
        arr = cv2.resize(arr, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_CUBIC)
    arr = cv2.equalizeHist(arr)
    if remove_map_lines:
        arr = remove_long_map_lines(arr, line_kernel_size)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)


def remove_long_map_lines(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return gray
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    line_mask = cv2.bitwise_or(horizontal, vertical)
    line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    cleaned = gray.copy()
    cleaned[line_mask > 0] = 255
    return cleaned


def make_easyocr_reader(languages: list[str], gpu: bool, model_dir: Path):
    import easyocr

    model_dir.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(
        languages,
        gpu=gpu,
        model_storage_directory=str(model_dir),
        user_network_directory=str(model_dir / "user_network"),
        verbose=False,
    )


def ocr_easyocr(
    reader,
    image: np.ndarray,
    tile: str,
    window: Window,
    image_scale: float,
    batch_size: int,
    canvas_size: int,
    mag_ratio: float,
    rotation_angles: list[float],
    text_threshold: float,
    low_text: float,
    link_threshold: float,
) -> list[OCRBox]:
    boxes = []
    for angle in rotation_angles:
        rotated, inverse_matrix = rotate_image_for_ocr(image, angle)
        results = reader.readtext(
            rotated,
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=batch_size,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
        )
        for polygon, text, confidence in results:
            px = tuple(
                image_point_to_tile_point(
                    float(x),
                    float(y),
                    inverse_matrix,
                    image_scale=image_scale,
                    window=window,
                )
                for x, y in polygon
            )
            boxes.append(OCRBox(str(text), float(confidence), px, tile))
    return boxes


def rotate_image_for_ocr(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray | None]:
    if abs(angle) < 0.001:
        return image, None
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    inverse_matrix = cv2.invertAffineTransform(matrix)
    return rotated, inverse_matrix


def image_point_to_tile_point(
    x: float,
    y: float,
    inverse_matrix: np.ndarray | None,
    image_scale: float,
    window: Window,
) -> tuple[float, float]:
    if inverse_matrix is not None:
        x, y = inverse_matrix @ np.array([x, y, 1.0])
    return (float(x) / image_scale + window.col_off, float(y) / image_scale + window.row_off)


def ocr_tesseract(image: np.ndarray, tile: str, window: Window, image_scale: float) -> list[OCRBox]:
    import pytesseract

    data = pytesseract.image_to_data(
        image,
        lang="deu",
        config="--oem 1 --psm 11",
        output_type=pytesseract.Output.DICT,
    )
    boxes = []
    for i, text in enumerate(data["text"]):
        text = str(text).strip()
        try:
            confidence = float(data["conf"][i]) / 100.0
        except ValueError:
            confidence = 0.0
        if not text or confidence <= 0:
            continue
        x = float(data["left"][i]) / image_scale + window.col_off
        y = float(data["top"][i]) / image_scale + window.row_off
        w = float(data["width"][i]) / image_scale
        h = float(data["height"][i]) / image_scale
        polygon = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
        boxes.append(OCRBox(text, confidence, polygon, tile))
    return boxes


def compatible(a: OCRBox, b: OCRBox, max_angle: float, max_perp_gap: float, max_word_gap: float) -> bool:
    angle_delta = abs(a.angle - b.angle)
    angle_delta = min(angle_delta, 180 - angle_delta)
    if angle_delta > max_angle:
        return False
    angle = math.radians((a.angle + b.angle) / 2)
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux
    ax, ay = a.center
    bx, by = b.center
    parallel = abs((bx - ax) * ux + (by - ay) * uy)
    perpendicular = abs((bx - ax) * vx + (by - ay) * vy)
    return perpendicular <= max_perp_gap and parallel <= max_word_gap


def group_line_boxes(boxes: list[OCRBox], max_angle: float, max_perp_gap: float, max_word_gap: float) -> list[list[OCRBox]]:
    groups: list[list[OCRBox]] = []
    unused = set(range(len(boxes)))
    while unused:
        seed = unused.pop()
        group = [seed]
        changed = True
        while changed:
            changed = False
            for idx in list(unused):
                if any(compatible(boxes[idx], boxes[j], max_angle, max_perp_gap, max_word_gap) for j in group):
                    unused.remove(idx)
                    group.append(idx)
                    changed = True
        grouped = [boxes[i] for i in group]
        angle = math.radians(sum(b.angle for b in grouped) / len(grouped))
        ux, uy = math.cos(angle), math.sin(angle)
        grouped.sort(key=lambda b: b.center[0] * ux + b.center[1] * uy)
        groups.append(grouped)
    return groups


def phrase_candidates(line: list[OCRBox], max_words: int) -> Iterable[list[OCRBox]]:
    for start in range(len(line)):
        for end in range(start + 1, min(len(line), start + max_words) + 1):
            yield line[start:end]


def best_match(text: str, aliases: dict[str, str], min_ratio: float) -> tuple[str | None, float]:
    norm = split_attached_suffixes(text)
    tokens = norm.split()
    informative_tokens = [t for t in tokens if t not in STREET_SUFFIXES and len(t) >= 2]
    if not informative_tokens:
        return None, 0.0
    if norm in aliases:
        return aliases[norm], 1.0
    compact = norm.replace(" ", "")
    if compact in aliases:
        return aliases[compact], 1.0
    partial_name = unique_partial_token_match(informative_tokens, aliases)
    if partial_name:
        return partial_name, max(0.86, difflib.SequenceMatcher(None, norm, normalize_text(partial_name)).ratio())
    candidates = difflib.get_close_matches(norm, aliases.keys(), n=1, cutoff=min_ratio)
    if not candidates:
        return None, 0.0
    ratio = difflib.SequenceMatcher(None, norm, candidates[0]).ratio()
    return aliases[candidates[0]], ratio


def unique_partial_token_match(tokens: list[str], aliases: dict[str, str]) -> str | None:
    distinctive = {t for t in tokens if len(t) >= 5}
    if not distinctive:
        return None
    matched_names = set()
    for alias, name in aliases.items():
        alias_tokens = set(alias.split())
        if distinctive.issubset(alias_tokens):
            matched_names.add(name)
            if len(matched_names) > 1:
                return None
    if len(matched_names) == 1:
        return next(iter(matched_names))
    return None


def polygon_from_boxes(boxes: list[OCRBox], transform, geometry_mode: str, buffer: float) -> Polygon:
    pts_px = list(itertools.chain.from_iterable(b.polygon_px for b in boxes))
    pts_map = [rasterio.transform.xy(transform, y, x, offset="center") for x, y in pts_px]
    hull = MultiPoint(pts_map).convex_hull
    if geometry_mode == "rotated-rectangle":
        hull = hull.minimum_rotated_rectangle
    if buffer:
        hull = hull.buffer(buffer)
    return hull


def deduplicate(records: list[dict], distance: float) -> list[dict]:
    kept: list[dict] = []
    grid: dict[tuple[int, int, str], list[int]] = {}
    for rec in sorted(records, key=lambda r: r["confidence"], reverse=True):
        c = rec["geometry"].centroid
        key = (int(c.x // distance), int(c.y // distance), rec["matched_name"])
        nearby_keys = [
            (key[0] + dx, key[1] + dy, key[2])
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]
        duplicate = False
        for nk in nearby_keys:
            for idx in grid.get(nk, []):
                other = kept[idx]
                if rec["geometry"].centroid.distance(other["geometry"].centroid) <= distance:
                    duplicate = True
                    break
            if duplicate:
                break
        if duplicate:
            continue
        grid.setdefault(key, []).append(len(kept))
        kept.append(rec)
    return kept


def remove_layer_if_exists(gpkg: Path, layer: str) -> None:
    if gpkg.exists() and layer in fiona.listlayers(gpkg):
        fiona.remove(gpkg, layer=layer, driver="GPKG")


def records_to_gdf(records: list[dict], crs: str) -> gpd.GeoDataFrame:
    columns = ["matched_name", "ocr_text", "confidence", "match_score", "tile", "geometry"]
    return gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs=crs)


def write_records(gpkg: Path, layer: str, records: list[dict], crs: str, append: bool) -> None:
    if not records and append:
        return
    gpkg.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and gpkg.exists() and layer in fiona.listlayers(gpkg) else "w"
    gdf = records_to_gdf(records, crs)
    gdf.to_file(gpkg, layer=layer, driver="GPKG", engine="fiona", mode=mode)


def tile_gpkg_path(tile_output_dir: Path, tif_path: Path) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tif_path.stem)
    return tile_output_dir / f"{safe_name}.gpkg"


def merge_tile_gpkgs(tile_output_dir: Path, layer: str, crs: str) -> gpd.GeoDataFrame:
    parts = []
    for gpkg in sorted(tile_output_dir.glob("*.gpkg")):
        if layer not in fiona.listlayers(gpkg):
            continue
        part = gpd.read_file(gpkg, layer=layer, engine="fiona")
        if not part.empty:
            parts.append(part)
    if not parts:
        return records_to_gdf([], crs)
    return gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs=parts[0].crs or crs,
    ).to_crs(crs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpkg", type=Path, default=Path("gdi_background.gpkg"), help="GeoPackage containing the street-name layer and receiving the merged output layer.")
    parser.add_argument("--street-layer", default="streets", help="Layer in --gpkg that contains street geometries and names.")
    parser.add_argument("--street-name-column", default="nam", help="Column in --street-layer that contains the official street name.")
    parser.add_argument("--raster-dir", type=Path, default=Path("k5_sw_download") / "k5_sw", help="Folder containing input TIFF tiles to OCR.")
    parser.add_argument("--tile-pattern", default=None, help="Optional filename glob to process only matching TIFFs, for example '3695805*.tif'.")
    parser.add_argument("--output-layer", default="text_blockers", help="Final merged GeoPackage layer name for detected text blocker polygons.")
    parser.add_argument("--target-crs", default="EPSG:25833", help="CRS assigned to output polygons; K5 tile coordinates are Berlin / UTM zone 33N.")
    parser.add_argument("--backend", choices=("easyocr", "tesseract"), default="easyocr", help="OCR engine to use; EasyOCR is the main GPU-capable backend.")
    parser.add_argument("--languages", nargs="+", default=["de", "en"], help="OCR language list passed to EasyOCR.")
    parser.add_argument("--easyocr-model-dir", type=Path, default=Path(".easyocr"), help="Local folder for EasyOCR model weights.")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU use for EasyOCR even if CUDA is available.")
    parser.add_argument("--chip-size", type=int, default=2048, help="Pixel width/height of OCR windows cut from each TIFF tile.")
    parser.add_argument("--overlap", type=int, default=256, help="Pixel overlap between neighboring OCR windows to avoid cutting labels at chip edges.")
    parser.add_argument("--image-scale", type=float, default=1.5, help="Upscale factor before OCR; higher helps small text but is slower.")
    parser.add_argument("--remove-map-lines", action="store_true", help="Preprocess each OCR chip by whitening long horizontal/vertical linework.")
    parser.add_argument("--line-kernel-size", type=int, default=90, help="Pixel kernel length used by --remove-map-lines; larger removes only longer lines.")
    parser.add_argument("--easyocr-batch-size", type=int, default=16, help="Recognition batch size inside EasyOCR; higher may be faster but uses more VRAM.")
    parser.add_argument("--easyocr-canvas-size", type=int, default=2560, help="EasyOCR detector resize canvas; higher can help large chips/small text but is slower.")
    parser.add_argument("--easyocr-mag-ratio", type=float, default=1.0, help="EasyOCR detector magnification ratio; higher can improve tiny text detection at a speed cost.")
    parser.add_argument("--easyocr-text-threshold", type=float, default=0.7, help="EasyOCR text confidence threshold for detector pixels; lower detects more but adds noise.")
    parser.add_argument("--easyocr-low-text", type=float, default=0.4, help="EasyOCR low-confidence text threshold used to grow weak text regions.")
    parser.add_argument("--easyocr-link-threshold", type=float, default=0.4, help="EasyOCR threshold for linking nearby character regions into words.")
    parser.add_argument("--rotation-angles", nargs="+", type=float, default=[0.0], help="Extra OCR passes at these image rotation angles, useful for diagonal or vertical labels.")
    parser.add_argument("--max-words", type=int, default=4, help="Maximum number of neighboring OCR words to combine into one street-name candidate.")
    parser.add_argument("--min-ocr-confidence", type=float, default=0.25, help="Minimum OCR confidence for individual OCR boxes before matching.")
    parser.add_argument("--min-match-ratio", type=float, default=0.84, help="Minimum fuzzy string similarity against street names; lower increases recall and false positives.")
    parser.add_argument("--max-angle-delta", type=float, default=20.0, help="Maximum angle difference in degrees for OCR words to be grouped on the same text line.")
    parser.add_argument("--max-perp-gap", type=float, default=45.0, help="Maximum perpendicular pixel gap between OCR words grouped into the same line.")
    parser.add_argument("--max-word-gap", type=float, default=220.0, help="Maximum along-line pixel gap between OCR words grouped into the same line.")
    parser.add_argument("--dedupe-distance", type=float, default=8.0, help="Map-unit distance for dropping duplicate detections of the same street name.")
    parser.add_argument("--geometry-mode", choices=("convex-hull", "rotated-rectangle"), default="convex-hull", help="Shape written for detected text: tight convex hull or old rotated rectangle.")
    parser.add_argument("--geometry-buffer", type=float, default=0.0, help="Optional map-unit buffer around the final blocker polygon.")
    parser.add_argument("--tile-output-dir", type=Path, default=Path("tile_text_blockers"), help="Folder for per-tile GeoPackage outputs.")
    parser.add_argument("--no-tile-gpkgs", action="store_true", help="Disable per-tile GeoPackages and keep all records in memory until final write.")
    parser.add_argument("--force-tile-gpkgs", action="store_true", help="Reprocess tiles even when their per-tile GeoPackage already exists.")
    parser.add_argument("--workers", type=int, default=1, help="Number of tile-level worker processes. Start with 2 for GPU OCR to avoid VRAM contention.")
    parser.add_argument("--progress-every-windows", type=int, default=100, help="Print worker progress after this many OCR windows per tile; set 0 to disable.")
    parser.add_argument("--limit-tiles", type=int, default=None, help="Process only the first N matching TIFF tiles for testing.")
    parser.add_argument("--limit-windows", type=int, default=None, help="Process only the first N OCR windows across all tiles for quick smoke tests.")
    return parser.parse_args()


def init_worker(args: argparse.Namespace, aliases: dict[str, str]) -> None:
    global _WORKER_ARGS, _WORKER_ALIASES, _WORKER_READER
    _WORKER_ARGS = args
    _WORKER_ALIASES = aliases
    _WORKER_READER = None
    if args.backend == "easyocr":
        _WORKER_READER = make_easyocr_reader(args.languages, gpu=not args.no_gpu, model_dir=args.easyocr_model_dir)
    elif shutil.which("tesseract") is None:
        raise RuntimeError("Tesseract executable was not found on PATH.")


def worker_ocr(image: np.ndarray, tile: str, window: Window) -> list[OCRBox]:
    args = _WORKER_ARGS
    if args.backend == "easyocr":
        return ocr_easyocr(
            _WORKER_READER,
            image,
            tile,
            window,
            image_scale=args.image_scale,
            batch_size=args.easyocr_batch_size,
            canvas_size=args.easyocr_canvas_size,
            mag_ratio=args.easyocr_mag_ratio,
            rotation_angles=args.rotation_angles,
            text_threshold=args.easyocr_text_threshold,
            low_text=args.easyocr_low_text,
            link_threshold=args.easyocr_link_threshold,
        )
    return ocr_tesseract(image, tile, window, image_scale=args.image_scale)


def process_tile(payload: tuple[str, int, int]) -> dict:
    tif_path_str, tile_index, total_tiles = payload
    tif_path = Path(tif_path_str)
    args = _WORKER_ARGS
    aliases = _WORKER_ALIASES
    use_tile_gpkgs = not args.no_tile_gpkgs
    tile_gpkg = tile_gpkg_path(args.tile_output_dir, tif_path)

    if use_tile_gpkgs and tile_gpkg.exists() and not args.force_tile_gpkgs:
        return {
            "status": "skipped",
            "message": f"[{tile_index}/{total_tiles}] Skip {tif_path.name} -> {tile_gpkg}",
            "records": [],
            "count": 0,
        }
    if use_tile_gpkgs and tile_gpkg.exists() and args.force_tile_gpkgs:
        tile_gpkg.unlink()

    tile_records: list[dict] = []
    processed_windows = 0
    with rasterio.open(tif_path) as src:
        transform = src.transform
        windows = list(iter_windows(src.width, src.height, args.chip_size, args.overlap))
        for window_index, window in enumerate(windows, start=1):
            if args.limit_windows is not None and processed_windows >= args.limit_windows:
                break
            image = raster_window_to_image(
                src,
                window,
                image_scale=args.image_scale,
                remove_map_lines=args.remove_map_lines,
                line_kernel_size=args.line_kernel_size,
            )
            processed_windows += 1
            boxes = [
                b
                for b in worker_ocr(image, tif_path.name, window)
                if b.confidence >= args.min_ocr_confidence and normalize_text(b.text)
            ]
            if not boxes:
                continue
            lines = group_line_boxes(
                boxes,
                max_angle=args.max_angle_delta,
                max_perp_gap=args.max_perp_gap,
                max_word_gap=args.max_word_gap,
            )
            for line in lines:
                for candidate_boxes in phrase_candidates(line, args.max_words):
                    phrase = " ".join(b.text for b in candidate_boxes)
                    matched_name, score = best_match(phrase, aliases, args.min_match_ratio)
                    if not matched_name:
                        continue
                    confidence = min(1.0, score * float(np.mean([b.confidence for b in candidate_boxes])))
                    geometry = polygon_from_boxes(
                        candidate_boxes,
                        transform,
                        geometry_mode=args.geometry_mode,
                        buffer=args.geometry_buffer,
                    )
                    if geometry.is_empty:
                        continue
                    tile_records.append(
                        {
                            "matched_name": matched_name,
                            "ocr_text": phrase,
                            "confidence": confidence,
                            "match_score": score,
                            "tile": tif_path.name,
                            "geometry": geometry,
                        }
                    )
            if args.progress_every_windows and window_index % args.progress_every_windows == 0:
                print(
                    f"[{tile_index}/{total_tiles}] {tif_path.name}: "
                    f"{window_index}/{len(windows)} windows, matches={len(tile_records)}",
                    flush=True,
                )

    tile_records = deduplicate(tile_records, args.dedupe_distance)
    if use_tile_gpkgs:
        write_records(tile_gpkg, args.output_layer, tile_records, crs=args.target_crs, append=False)
        records_for_parent = []
    else:
        records_for_parent = tile_records
    return {
        "status": "processed",
        "message": f"[{tile_index}/{total_tiles}] {tif_path.name}: wrote {len(tile_records)} blockers",
        "records": records_for_parent,
        "count": len(tile_records),
    }


def main() -> None:
    args = parse_args()
    aliases = read_street_vocabulary(args.gpkg, args.street_layer, args.street_name_column)
    tif_paths = sorted(args.raster_dir.glob("*.tif"))
    if args.tile_pattern:
        tif_paths = [p for p in tif_paths if fnmatch.fnmatch(p.name, args.tile_pattern)]
    if args.limit_tiles:
        tif_paths = tif_paths[: args.limit_tiles]
    if not tif_paths:
        raise FileNotFoundError(f"No .tif files found in {args.raster_dir}")

    use_tile_gpkgs = not args.no_tile_gpkgs
    records: list[dict] = []
    if args.limit_windows is not None and args.workers != 1:
        print("--limit-windows is a serial smoke-test option; using --workers 1 for this run.")
        args.workers = 1

    payloads = [(str(path), index, len(tif_paths)) for index, path in enumerate(tif_paths, start=1)]
    if args.workers <= 1:
        init_worker(args, aliases)
        for payload in payloads:
            result = process_tile(payload)
            print(result["message"])
            records.extend(result["records"])
    else:
        print(f"Starting {args.workers} tile worker processes")
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args, aliases)) as pool:
            futures = [pool.submit(process_tile, payload) for payload in payloads]
            for future in as_completed(futures):
                result = future.result()
                print(result["message"])
                records.extend(result["records"])

    if use_tile_gpkgs:
        merged = merge_tile_gpkgs(args.tile_output_dir, args.output_layer, args.target_crs)
        remove_layer_if_exists(args.gpkg, args.output_layer)
        merged.to_file(args.gpkg, layer=args.output_layer, driver="GPKG", engine="fiona")
        print(
            f"Merged {len(merged)} text blockers from {args.tile_output_dir} "
            f"to {args.gpkg}:{args.output_layer}"
        )
    else:
        records = deduplicate(records, args.dedupe_distance)
        remove_layer_if_exists(args.gpkg, args.output_layer)
        write_records(args.gpkg, args.output_layer, records, crs=args.target_crs, append=False)
        print(f"Wrote {len(records_to_gdf(records, args.target_crs))} text blockers to {args.gpkg}:{args.output_layer}")


if __name__ == "__main__":
    main()
