Saliency Service

A serverless saliency detection API deployed on RunPod.
This service takes an image URL as input and returns a binary saliency mask (regions of interest highlighted).


---

🔧 Setup

This endpoint is based on:

Python 3.10

RunPod worker runtime

Dependencies in requirements.txt:

runpod

opencv-python-headless

numpy

requests




---

🚀 Usage

Once deployed, you can call the endpoint with:

curl -X POST https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -d '{
    "input": {
      "image_url": "https://upload.wikimedia.org/wikipedia/commons/1/17/Google-flutter-logo.png"
    }
  }'


---

📦 Response

The API will return JSON like:

{
  "status": "COMPLETED",
  "output": {
    "mask_url": "https://runpod-bucket/.../saliency-mask.png"
  }
}

mask_url points to the saliency mask stored in temporary object storage.

White = salient region, Black = background.



---

🖥️ Local Testing

To run locally:

pip install -r requirements.txt
python handler.py

Then send requests to your local Flask/FastAPI test server (if you add one), or directly test the function in handler.py.

