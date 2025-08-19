import base64, io, os, requests
import numpy as np
from PIL import Image
import cv2
import runpod

def load_image(input_):
    # Accept: URL, base64 image, or path
    if isinstance(input_, str) and input_.startswith("http"):
        img_bytes = requests.get(input_, timeout=20).content
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if isinstance(input_, str) and input_.startswith("data:image/"):
        b64 = input_.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return Image.open(input_).convert("RGB")

def saliency_map(pil_img):
    # OpenCV expects BGR uint8
    img = np.array(pil_img)[:, :, ::-1]
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, sal = saliency.computeSaliency(img)
    if not success:
        raise RuntimeError("Saliency computation failed")
    sal = (sal * 255).astype(np.uint8)
    # normalize & colorize (optional)
    sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX)
    return sal

def encode_png(np_img):
    # Return base64 PNG
    ok, buf = cv2.imencode(".png", np_img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

def handler(job):
    inp = job["input"] or {}
    src = inp.get("image") or inp.get("image_url") or inp.get("image_path")
    if not src:
        return {"error": "Provide 'image' (base64), 'image_url', or 'image_path'."}

    try:
        pil = load_image(src)
        sal = saliency_map(pil)
        sal_b64 = encode_png(sal)
        return {"ok": True, "saliency": sal_b64}
    except Exception as e:
        return {"ok": False, "error": str(e)}

runpod.serverless.start({"handler": handler})
