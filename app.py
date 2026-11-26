import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import watermark as wm

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

PRESETS = [
    ("No attack", "none", 0),
    ("JPEG q90", "jpeg", 90), ("JPEG q70", "jpeg", 70), ("JPEG q50", "jpeg", 50), ("JPEG q30", "jpeg", 30),
    ("Noise s5", "noise", 5), ("Noise s10", "noise", 10), ("Noise s20", "noise", 20),
    ("Blur k3", "blur", 3), ("Blur k7", "blur", 7),
    ("Resize 50%", "resize", 50), ("Resize 25%", "resize", 25),
    ("Cut 10% area", "cut", 10), ("Cut 25% area", "cut", 25),
]


def png64(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def bits_img(bits, scale=8):
    img = (255 * (1 - bits)).astype(np.uint8)
    return png64(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))


def read(file, flag):
    data = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(data, flag)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    f = request.files.get("image")
    if not f:
        return jsonify(error="Upload an image."), 400
    img = read(f, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Could not read image."), 400
    img = wm.fit_image(img)

    wf = request.files.get("wm_file")
    if wf and wf.filename:
        g = read(wf, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return jsonify(error="Could not read watermark image."), 400
        bits = wm.image_to_bits(g)
    else:
        bits = wm.text_to_bits(request.form.get("wm_text", "WM"))

    delta = float(request.form.get("delta", 60))
    kind = request.form.get("attack", "jpeg")
    p = float(request.form.get("strength", 50))

    marked = wm.embed(img, bits, delta)
    attacked = wm.attack(marked, kind, p)
    got = wm.extract(attacked, delta)

    table = []
    for name, k, v in PRESETS:
        r = wm.extract(wm.attack(marked, k, v), delta)
        table.append({"name": name, "nc": round(wm.nc(bits, r), 3), "ber": round(wm.ber(bits, r), 3)})

    return jsonify(
        size=[img.shape[1], img.shape[0]],
        psnr=round(wm.psnr(img, marked), 2),
        ssim=round(wm.ssim(img, marked), 4),
        nc=round(wm.nc(bits, got), 3),
        ber=round(wm.ber(bits, got), 3),
        orig=png64(img), marked=png64(marked), attacked=png64(attacked),
        wm_orig=bits_img(bits), wm_ext=bits_img(got),
        table=table,
    )


if __name__ == "__main__":
    app.run(debug=True)
