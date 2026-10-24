"""Independent chessboard image generator for the refiner robustness audit.

Deliberately does NOT reuse ``Board.render`` (tools/render_uncertainty_overlays.py)
or ``optical_chessboard`` (chess-corners-testutil). Image formation:

1. Scene: an infinite chessboard on a plane, seen through a pinhole camera
   (homography ``H``: board cell units -> undistorted pixel coordinates) and a
   radial distortion. The distortion is defined image->undistorted
   (``x_u = c + (x_d - c)(1 + k1 r^2 + k2 r^4)``, ``r = |x_d - c| / R``) so the
   renderer can map every image sample to the board directly; the true corner
   positions invert it per corner by Newton's method.
2. Supersampling: ``S x S`` samples per pixel. Each sample is itself
   box-anti-aliased over its ``1/S`` footprint with a local linear ramp across
   each edge (``s~(u) = (-1)^n clamp(d / hw)``, ``hw = (|u_x| + |u_y|) / 2S``),
   so edge positions are not quantised to the sample grid (plain point
   sampling would shift an axis-aligned edge by up to 1/(2S) px).
3. Optional scene illumination (multiplicative, before the lens).
4. Lens PSF convolved at the supersampled resolution (FFT). Every PSF is
   normalised to unit sum and zero centroid, so an asymmetric PSF breaks point
   symmetry without translating the whole image.
5. Optional image-plane vignetting (multiplicative, after the lens).
6. Box downsample S x S -> pixel integration.
7. Sensor / pipeline effects in ``sensor()``: additive or Poisson-Gaussian
   noise, Bayer RGGB mosaic + bilinear demosaic, gamma, JPEG, 8-bit rounding.

Pixel ``i`` covers ``[i - 0.5, i + 0.5]``: coordinate ``i`` is its centre.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def rot(ax: float, ay: float, az: float) -> np.ndarray:
    ax, ay, az = np.deg2rad([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class Geometry:
    """Camera looking at the board plane. ``cell`` is the cell size in px at
    the image centre; ``tilt`` = (rx, ry) degrees, ``roll`` degrees; ``focal``
    in px (smaller = stronger perspective); ``phase`` = board offset (cells)."""

    w: int
    h: int
    cell: float = 10.0
    roll: float = 7.0
    tilt: tuple[float, float] = (0.0, 0.0)
    focal: float = 4000.0
    phase: tuple[float, float] = (0.0, 0.0)
    k1: float = 0.0
    k2: float = 0.0

    def homography(self) -> np.ndarray:
        r = rot(self.tilt[0], self.tilt[1], self.roll)
        z = self.focal / self.cell  # distance in cell units along the optical axis
        # Board point (u, v, 0) -> camera = R (u - u0, v - v0, 0) + (0, 0, z):
        # the board point at `phase` projects to the image centre.
        t = np.array([0.0, 0.0, z]) - r @ np.array([self.phase[0], self.phase[1], 0.0])
        k = np.array([[self.focal, 0, (self.w - 1) / 2], [0, self.focal, (self.h - 1) / 2], [0, 0, 1]])
        return k @ np.column_stack([r[:, 0], r[:, 1], t])

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.w - 1) / 2, (self.h - 1) / 2])

    @property
    def rnorm(self) -> float:
        return 0.5 * np.hypot(self.w, self.h)

    def undistort(self, xd: np.ndarray, yd: np.ndarray):
        if self.k1 == 0 and self.k2 == 0:
            return xd, yd
        c = self.centre
        dx, dy = xd - c[0], yd - c[1]
        r2 = (dx * dx + dy * dy) / self.rnorm**2
        f = 1 + self.k1 * r2 + self.k2 * r2 * r2
        return c[0] + dx * f, c[1] + dy * f

    def distort(self, xu: np.ndarray) -> np.ndarray:
        """Invert ``undistort`` per point (Newton on the radius)."""
        if self.k1 == 0 and self.k2 == 0:
            return xu.copy()
        c = self.centre
        d = xu - c
        ru = np.linalg.norm(d, axis=1) / self.rnorm
        rd = ru.copy()
        for _ in range(50):
            g = rd * (1 + self.k1 * rd**2 + self.k2 * rd**4) - ru
            dg = 1 + 3 * self.k1 * rd**2 + 5 * self.k2 * rd**4
            rd -= g / dg
        scale = np.where(ru > 0, rd / np.maximum(ru, 1e-300), 1.0)
        return c + d * scale[:, None]

    def corners(self, margin: float) -> np.ndarray:
        """True corner positions (distorted image coordinates)."""
        hm = self.homography()
        inv = np.linalg.inv(hm)
        box = np.array([[0, 0, 1], [self.w, 0, 1], [0, self.h, 1], [self.w, self.h, 1]], float)
        # Undistorted box can be larger than the image; pad generously.
        uv = box @ inv.T
        uv = uv[:, :2] / uv[:, 2:3]
        lo, hi = np.floor(uv.min(0)) - 6, np.ceil(uv.max(0)) + 6
        uu, vv = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        p = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 1) @ hm.T
        ok = p[:, 2] > 0
        xu = p[ok, :2] / p[ok, 2:3]
        xd = self.distort(xu)
        keep = (
            (xd[:, 0] >= margin)
            & (xd[:, 0] <= self.w - 1 - margin)
            & (xd[:, 1] >= margin)
            & (xd[:, 1] <= self.h - 1 - margin)
        )
        return xd[keep]


# ---------------------------------------------------------------------------
# PSFs (sampled on the supersampled grid, unit sum, zero centroid)
# ---------------------------------------------------------------------------


def _grid(rad_px: float, s: int):
    n = int(np.ceil(rad_px * s))
    t = (np.arange(-n, n + 1)) / s
    return np.meshgrid(t, t)


def _normalise(k: np.ndarray, s: int) -> np.ndarray:
    """Unit sum, and shift by the (sub-sample) centroid with bilinear
    resampling so the first moment is zero."""
    k = np.clip(k, 0, None)
    k = k / k.sum()
    n = k.shape[0]
    t = np.arange(n) - (n - 1) / 2
    cx = (k.sum(0) * t).sum()
    cy = (k.sum(1) * t).sum()
    if abs(cx) > 1e-9 or abs(cy) > 1e-9:
        m = np.float32([[1, 0, -cx], [0, 1, -cy]])
        k = cv2.warpAffine(k.astype(np.float64), m, (n, n), flags=cv2.INTER_LINEAR, borderValue=0)
        k = k / k.sum()
    return k


def psf_gaussian(sigma: float, s: int) -> np.ndarray:
    x, y = _grid(4.5 * sigma + 0.5, s)
    return _normalise(np.exp(-0.5 * (x * x + y * y) / sigma**2), s)


def psf_disk(radius: float, s: int, apod: float = 0.0, apod_angle: float = 0.0) -> np.ndarray:
    """Defocus pillbox of ``radius`` px,
    optionally with a linear intensity tilt across the
    aperture (``apod`` = relative change from centre to rim) -- an asymmetric,
    decentred-defocus-like blur."""
    x, y = _grid(radius + 1.0, s)
    r = np.hypot(x, y)
    k = np.clip((radius - r) * s + 0.5, 0, 1)  # boundary anti-aliased over one sample
    if apod:
        a = np.deg2rad(apod_angle)
        k = k * (1 + apod * (x * np.cos(a) + y * np.sin(a)) / radius)
    return _normalise(k, s)


def psf_coma(core: float, length: float, angle: float, s: int) -> np.ndarray:
    """Comet-shaped PSF: a sum of Gaussians along direction ``angle`` whose
    width grows and weight decays along the tail; recentred to zero
    centroid."""
    rad = length + 4.5 * core * 2.5 + 0.5
    x, y = _grid(rad, s)
    a = np.deg2rad(angle)
    k = np.zeros_like(x)
    for t in np.linspace(0, 1, 24):
        cx, cy = t * length * np.cos(a), t * length * np.sin(a)
        sg = core * (1 + 1.5 * t)
        wgt = (1 - 0.8 * t)
        k += wgt / sg**2 * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sg**2)
    return _normalise(k, s)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass
class Optics:
    psf: np.ndarray | None = None  # supersampled kernel, or None = no blur
    illum: str | None = None  # scene illumination model
    illum_amp: float = 0.0
    vignette: float = 0.0  # relative falloff at the image corner (0 = none)


def render(geom: Geometry, optics: Optics, s: int = 16, dark: float = 40.0, light: float = 200.0) -> np.ndarray:
    """Linear irradiance image (float64, gray levels before sensor effects)."""
    kr = 0 if optics.psf is None else (optics.psf.shape[0] // 2)
    m = int(np.ceil(kr / s)) + 1  # margin in px
    W, Hh = geom.w + 2 * m, geom.h + 2 * m
    xs = (np.arange(W * s) + 0.5) / s - 0.5 - m
    ys = (np.arange(Hh * s) + 0.5) / s - 0.5 - m
    hm_inv = np.linalg.inv(geom.homography())
    out = np.empty((Hh * s, W * s), dtype=np.float32)
    # Process in row bands to bound memory.
    band = 256
    for r0 in range(0, Hh * s, band):
        r1 = min(Hh * s, r0 + band + 2)
        yy, xx = np.meshgrid(ys[max(r0 - 1, 0):r1], xs, indexing="ij")
        xu, yu = geom.undistort(xx, yy)
        den = hm_inv[2, 0] * xu + hm_inv[2, 1] * yu + hm_inv[2, 2]
        u = (hm_inv[0, 0] * xu + hm_inv[0, 1] * yu + hm_inv[0, 2]) / den
        v = (hm_inv[1, 0] * xu + hm_inv[1, 1] * yu + hm_inv[1, 2]) / den
        # Gradients in px units (sample spacing 1/s).
        ux = np.gradient(u, axis=1) * s
        uy = np.gradient(u, axis=0) * s
        vx = np.gradient(v, axis=1) * s
        vy = np.gradient(v, axis=0) * s

        def smooth_sq(a, ax_, ay_):
            n = np.round(a)
            d = a - n
            hw = (np.abs(ax_) + np.abs(ay_)) / (2 * s)
            sgn = 1.0 - 2.0 * (np.mod(n, 2))  # (-1)^n
            return sgn * np.clip(d / hw, -1, 1)

        val = 0.5 + 0.5 * smooth_sq(u, ux, uy) * smooth_sq(v, vx, vy)
        lo = 1 if r0 > 0 else 0
        rows = min(band, Hh * s - r0)
        val = val[lo:lo + rows]
        img = dark + (light - dark) * val
        if optics.illum is not None:
            img = img * _illum(optics, xx[lo:lo + rows], yy[lo:lo + rows], geom)
        out[r0:r0 + rows] = img
    if optics.psf is not None:
        out = fftconvolve(out, optics.psf.astype(np.float32), mode="same")
    if optics.vignette:
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        c = geom.centre
        r2 = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / geom.rnorm**2
        out = out * (1 - optics.vignette * r2)
    # Box downsample and crop the margin.
    out = out.reshape(Hh, s, W, s).mean(axis=(1, 3))
    return out[m:m + geom.h, m:m + geom.w].astype(np.float64)


def _illum(optics: Optics, xx, yy, geom: Geometry):
    a = optics.illum_amp
    if optics.illum == "ramp":  # linear, +-a across the image width, oblique
        t = ((xx - geom.centre[0]) * 0.8 + (yy - geom.centre[1]) * 0.6) / (0.5 * geom.w)
        return 1 + a * t
    if optics.illum == "wave":  # smooth illumination texture, period 5 cells
        p = 5 * geom.cell
        return 1 + a * np.sin(2 * np.pi * (0.8 * xx + 0.6 * yy) / p + 0.3)
    raise ValueError(optics.illum)


# ---------------------------------------------------------------------------
# Sensor / pipeline
# ---------------------------------------------------------------------------


@dataclass
class Sensor:
    noise: float = 2.0  # additive white σ (gray levels, linear domain)
    poisson_b: float = 0.0  # signal-dependent variance slope: var = noise^2 + b * I
    bayer: bool = False
    wb: tuple[float, float, float] = (0.55, 1.0, 0.45)  # R, G, B sensitivity
    gamma: float | None = None  # encode exponent, e.g. 1/2.2
    jpeg: int | None = None
    extra: dict = field(default_factory=dict)


def sensor(lin: np.ndarray, sen: Sensor, rng: np.random.Generator | None) -> np.ndarray:
    """Apply sensor effects; ``rng=None`` -> noise-free (but every
    deterministic stage, including rounding, demosaicing and JPEG)."""
    img = lin.astype(np.float64)
    if sen.bayer:
        h, w = img.shape
        raw = np.empty_like(img)
        # RGGB: (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B
        gains = np.empty_like(img)
        gains[0::2, 0::2] = sen.wb[0]
        gains[0::2, 1::2] = sen.wb[1]
        gains[1::2, 0::2] = sen.wb[1]
        gains[1::2, 1::2] = sen.wb[2]
        raw = img * gains
        raw = _noise(raw, sen, rng)
        raw8 = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(raw8, cv2.COLOR_BayerBG2BGR).astype(np.float64)  # OpenCV's BG == RGGB at (0,0)=R
        bgr[..., 2] /= sen.wb[0]
        bgr[..., 1] /= sen.wb[1]
        bgr[..., 0] /= sen.wb[2]
        img = 0.299 * bgr[..., 2] + 0.587 * bgr[..., 1] + 0.114 * bgr[..., 0]
    else:
        img = _noise(img, sen, rng)
    if sen.gamma is not None:
        img = 255.0 * np.clip(img / 255.0, 0, 1) ** sen.gamma
    out = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if sen.jpeg is not None:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, sen.jpeg])
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out


def _noise(img, sen: Sensor, rng):
    if rng is None:
        return img
    var = sen.noise**2 + sen.poisson_b * np.clip(img, 0, None)
    return img + rng.normal(0.0, 1.0, img.shape) * np.sqrt(var)
