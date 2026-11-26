"""Blind DWT-DCT watermarking (QIM) with redundancy + keyed scrambling."""
import cv2
import numpy as np
import pywt
from scipy.fft import dctn, idctn
from skimage.metrics import structural_similarity

WM = 32            # watermark is WM x WM bits
NBITS = WM * WM
B = 8              # DCT block size
COEF = (3, 2)      # mid-frequency coefficient used for embedding


def fit_image(img, min_side=512, max_side=1280):
    """Make size workable: enough blocks to hold the watermark, not too big, multiple of 16."""
    h, w = img.shape[:2]
    s = 1.0
    if min(h, w) < min_side:
        s = min_side / min(h, w)
    elif max(h, w) > max_side:
        s = max(max_side / max(h, w), min_side / min(h, w), 0.0)
        s = min(s, 1.0)
    if s != 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    return img[: h // 16 * 16, : w // 16 * 16]


def text_to_bits(text):
    text = (text or "WM").strip()[:6] or "WM"
    canvas = np.full((128, 128), 255, np.uint8)
    scale, th = 3.0, 6
    while True:
        (tw, thh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        if tw <= 116 or scale < 0.3:
            break
        scale -= 0.1
    x, y = (128 - tw) // 2, (128 + thh) // 2
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, th, cv2.LINE_AA)
    small = cv2.resize(canvas, (WM, WM), interpolation=cv2.INTER_AREA)
    return (small < 160).astype(np.uint8)


def image_to_bits(gray):
    small = cv2.resize(gray, (WM, WM), interpolation=cv2.INTER_AREA)
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bits = (th == 0).astype(np.uint8)  # dark = ink = 1
    if bits.mean() > 0.6:              # inverted logo, flip
        bits = 1 - bits
    return bits


def _transform(img):
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    LL, rest = pywt.dwt2(y, "haar")
    bh, bw = LL.shape[0] // B, LL.shape[1] // B
    blocks = LL.reshape(bh, B, bw, B).transpose(0, 2, 1, 3)
    return LL, rest, dctn(blocks, axes=(2, 3), norm="ortho"), (bh, bw)


def _bit_index(n, key):
    return np.random.default_rng(key).permutation(n) % NBITS


def embed(img, bits, delta=60.0, key=1234):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    LL, rest, D, (bh, bw) = _transform(img)
    b = bits.flatten()[_bit_index(bh * bw, key)].reshape(bh, bw).astype(np.float32)
    c = D[:, :, COEF[0], COEF[1]]
    D[:, :, COEF[0], COEF[1]] = np.round((c - b * delta / 2) / delta) * delta + b * delta / 2
    LL2 = idctn(D, axes=(2, 3), norm="ortho").transpose(0, 2, 1, 3).reshape(LL.shape)
    y2 = pywt.idwt2((LL2, rest), "haar")
    ycc[:, :, 0] = np.clip(np.round(y2), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def extract(img, delta=60.0, key=1234):
    _, _, D, (bh, bw) = _transform(img)
    c = D[:, :, COEF[0], COEF[1]]
    d0 = np.abs(c - np.round(c / delta) * delta)
    d1 = np.abs(c - (np.round((c - delta / 2) / delta) * delta + delta / 2))
    soft = (d0 - d1).flatten()  # >0 means bit 1
    votes = np.zeros(NBITS)
    np.add.at(votes, _bit_index(bh * bw, key), soft)
    return (votes > 0).astype(np.uint8).reshape(WM, WM)


# ---------- attacks ----------
def attack(img, kind, p):
    if kind == "none":
        return img.copy()
    if kind == "jpeg":
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(p)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "noise":
        rng = np.random.default_rng(0)
        return np.clip(img + rng.normal(0, p, img.shape), 0, 255).astype(np.uint8)
    if kind == "blur":
        k = int(p) | 1
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "resize":
        h, w = img.shape[:2]
        s = p / 100.0
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "cut":
        out = img.copy()
        h, w = out.shape[:2]
        f = np.sqrt(p / 100.0)
        out[int(h * (1 - f)):, int(w * (1 - f)):] = 0
        return out
    raise ValueError(kind)


# ---------- metrics ----------
def psnr(a, b):
    return float(cv2.PSNR(a, b))


def ssim(a, b):
    return float(structural_similarity(a, b, channel_axis=2))


def nc(w, w2):
    w, w2 = w.astype(float).flatten(), w2.astype(float).flatten()
    d = np.sqrt((w * w).sum()) * np.sqrt((w2 * w2).sum())
    return float((w * w2).sum() / d) if d else 0.0


def ber(w, w2):
    return float((w != w2).mean())
