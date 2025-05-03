from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf

try:
    model = tf.keras.models.load_model('reclosetmodel.h5')
except Exception as e:
    import traceback
    print("failed model loading:", e)
    traceback.print_exc()
    model = None

class_names = ['Large tear', 'Wear / Small tear', 'Shrinkage / Stretching / Wrinkling', 'Buckle / Button / Zipper damage',
'Oil / Food / Chemical stain', 'Ink', 'Mold']


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

     
        image = image.resize((416, 416))
        image_array = np.array(image) / 255.0  
        image_array = np.expand_dims(image_array, axis=0)


        predictions = model.predict(image_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": class_names[class_idx],
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
