# RunPod Serverless entrypoint
# CPU-friendly saliency via OpenCV Spectral Residual (fast baseline)

import base64, io, json, cv2, numpy as np, requests
from runpod import serverless

def load_image_from_input(inp):
    if "image_url" in inp and inp["image_url"]:
        r = requests.get(inp["image_url"], timeout=15)
        r.raise_for_status()
        data = np.frombuffer(r.content, np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if "image_b64" in inp and inp["image_b64"]:
        data = base64.b64decode(inp["image_b64"])
        data = np.frombuffer(data, np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    raise ValueError("Provide image_url or image_b64")

def spectral_residual_saliency(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # FFT
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag, angle = cv2.cartToPolar(dft[:,:,0], dft[:,:,1])
    log_mag = np.log(mag + 1e-8)
    avg = cv2.blur(log_mag, (3,3))
    spectral_residual = log_mag - avg
    # back to spatial
    real, imag = cv2.polarToCart(np.exp(spectral_residual), angle)
    dft[:,:,0], dft[:,:,1] = real, imag
    idft = cv2.idft(dft)
    sal = cv2.magnitude(idft[:,:,0], idft[:,:,1])
    sal = cv2.GaussianBlur(sal, (9,9), 2.5)
    sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX)
    sal = sal.astype(np.uint8)
    return sal

def make_mask_and_bbox(saliency, thresh_ratio=0.6):
    thr = int(np.percentile(saliency, 100*thresh_ratio))
    mask = (saliency >= thr).astype(np.uint8) * 255
    # largest contour bbox
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return mask, None
    c = max(cs, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return mask, (int(x),int(y),int(w),int(h))

def encode_png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def handler(job):
    """
    input:
      image_url: string (https) OR
      image_b64: base64 png/jpg
      thresh_ratio: float 0..1 (optional, default 0.6)
    output:
      {
        "mask_b64": "...",  // PNG of saliency mask
        "bbox": [x,y,w,h] or null,
        "saliency_stats": {"min":..,"max":..,"mean":..}
      }
    """
    try:
        inp = job.get("input", {})
        img = load_image_from_input(inp)
        sal = spectral_residual_saliency(img)
        mask, bbox = make_mask_and_bbox(sal, inp.get("thresh_ratio", 0.6))
        out = {
            "mask_b64": encode_png_b64(mask),
            "bbox": bbox if bbox else None,
            "saliency_stats": {
                "min": int(sal.min()),
                "max": int(sal.max()),
                "mean": float(sal.mean())
            }
        }
        return {"ok": True, "result": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    serverless.start({"handler": handler})
