"""Fixed-tile dense inference shared by calibration, comparison and anomaly runtime.

Schema 8 retains training tile size, normalization-before-zero-pad and weighted
overlap. It never changes a transformer's spatial graph to calibrate it.
"""

import math

import cv2
import numpy as np


def tile_contract(config):
    prep = config.get("preprocessing") or {}
    tile = prep.get("tiling") if not prep.get("whole_image") else None
    if not tile:
        return None
    size, overlap = int(tile["tile"]), int(tile["overlap"])
    scale = float(tile.get("scale", 1.0))
    expected = (
        "center_weighted_softmax"
        if config["task"] == "segment"
        else "center_weighted_map"
    )
    if (
        size < 32
        or size > 4096
        or size != config["input_height"]
        or size != config["input_width"]
        or overlap < 0
        or overlap >= size
        or not math.isfinite(scale)
        or not 0 < scale <= 1
        or tile.get("blend") != expected
        or tile.get("interpolation", "INTER_LINEAR_EXACT")
        not in {"INTER_AREA", "INTER_LINEAR_EXACT"}
    ):
        raise ValueError("Invalid dense tile contract")
    return tile


def region_pixels(config, bgr):
    if bgr is None or bgr.size == 0 or bgr.dtype != np.uint8:
        raise ValueError("Dense model input must be a nonempty uint8 image")
    crop = (config.get("preprocessing") or {}).get("center_crop")
    if crop:
        h, w = bgr.shape[:2]
        cw, ch = int(crop["width"]), int(crop["height"])
        if cw > w or ch > h:
            raise ValueError("Center crop exceeds image")
        bgr = bgr[
            (h - ch) // 2 : (h - ch) // 2 + ch, (w - cw) // 2 : (w - cw) // 2 + cw
        ]
    c = config["input_channels"]
    if c == 1:
        pixels = (
            bgr
            if bgr.ndim == 2
            else cv2.cvtColor(
                bgr, cv2.COLOR_BGR2GRAY if bgr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            )
        )
    elif c == 3:
        pixels = cv2.cvtColor(
            bgr,
            cv2.COLOR_GRAY2RGB
            if bgr.ndim == 2
            else cv2.COLOR_BGR2RGB
            if bgr.shape[2] == 3
            else cv2.COLOR_BGRA2RGB,
        )
    else:
        raise ValueError("Unsupported channel count")
    tile = tile_contract(config)
    if tile:
        scale = float(tile.get("scale", 1))
        h, w = pixels.shape[:2]
        # Same ties-to-even rounding as dataset.scaled_size and desktop inference.
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        interpolation = (
            cv2.INTER_AREA
            if tile.get("interpolation") == "INTER_AREA"
            else cv2.INTER_LINEAR_EXACT
        )
    else:
        size = (config["input_width"], config["input_height"])
        interpolation = cv2.INTER_LINEAR_EXACT
    if (pixels.shape[1], pixels.shape[0]) != size:
        pixels = cv2.resize(pixels, size, interpolation=interpolation)
    pixels = pixels.reshape(*pixels.shape[:2], c).astype(np.float32) / 255
    pixels = (pixels - np.asarray(config["normalize_mean"], np.float32)) / np.asarray(
        config["normalize_std"], np.float32
    )
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def origins(length, tile, stride):
    return sorted({*range(0, max(1, length - tile + 1), stride), max(0, length - tile)})


class PreparedWindows:
    """One normalized image, indexed tile views; never cache every tile copy."""

    def __init__(self, config, pixels):
        tile = tile_contract(config)
        self.pixels = pixels
        self.margin = 0
        _, h, w = pixels.shape
        if tile:
            size, overlap = int(tile["tile"]), int(tile["overlap"])
            self.margin = overlap // 2
            self.pixels = (
                np.pad(pixels, ((0, 0), (0, max(0, size - h)), (0, max(0, size - w))))
                if min(h, w) < size
                else pixels
            )
            h, w = self.pixels.shape[1:]
            self.shape = (size, size)
            self.positions = [
                (y, x)
                for y in origins(h, size, size - overlap)
                for x in origins(w, size, size - overlap)
            ]
        else:
            self.shape = (h, w)
            self.positions = [(0, 0)]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        y, x = self.positions[index]
        h, w = self.shape

        def axis(start, length, size):
            weight = np.ones(size, np.float32)
            if self.margin and start > 0:
                weight[: self.margin] = 1e-4
            if self.margin and start + size < length:
                weight[-self.margin :] = 1e-4
            return weight

        weight = (
            axis(y, self.pixels.shape[1], h)[:, None]
            * axis(x, self.pixels.shape[2], w)[None, :]
        )
        return (
            y,
            x,
            np.ascontiguousarray(self.pixels[None, :, y : y + h, x : x + w]),
            weight,
        )


def windows(config, pixels):
    return iter(PreparedWindows(config, pixels))


def predict_dense(config, bgr, run):
    """Return C×H×W probabilities (segment), or raw 1×H×W anomaly map.

    `run` consumes NCHW float32 and returns an output list. No GUI or torch
    dependency. Output is at the scaled ROI resolution before display resize.
    """
    pixels = region_pixels(config, bgr)
    _, h, w = pixels.shape
    tile = tile_contract(config)
    ph, pw = (max(h, tile["tile"]), max(w, tile["tile"])) if tile else (h, w)
    sums = None
    weights = np.zeros((ph, pw), np.float32)
    for y, x, tensor, weight in windows(config, pixels):
        outputs = run(tensor)
        values = np.asarray(outputs[0] if config["task"] == "segment" else outputs[1])
        channels = int(config["num_classes"]) if config["task"] == "segment" else 1
        if (
            values.shape != (1, channels, *tensor.shape[2:])
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid dense model output")
        values = values[0]
        if config["task"] == "segment":
            values = np.exp(values - values.max(0, keepdims=True))
            values /= values.sum(0, keepdims=True)
        if sums is None:
            sums = np.zeros((channels, ph, pw), np.float32)
        th, tw = weight.shape
        sums[:, y : y + th, x : x + tw] += values * weight
        weights[y : y + th, x : x + tw] += weight
    return (sums / weights)[:, :h, :w]
