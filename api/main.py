from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ THIS WORKS IN KERAS 3
# Model loading (INFERENCE ONLY)
MODEL = tf.keras.models.load_model(
    "potatoes.h5",
    compile=False
)


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return image

@app.get("/ping")
def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    preds = MODEL.predict(img_batch)
    idx = int(np.argmax(preds[0]))

    return {
        "class": CLASS_NAMES[idx],
        "confidence": float(preds[0][idx])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
