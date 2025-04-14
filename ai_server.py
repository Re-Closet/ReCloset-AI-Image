from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# 모델 로드 (처음에 1번만 로드)
try:
    model = tf.keras.models.load_model('reclosetmodel.h5')
except Exception as e:
    import traceback
    print("모델로딩실패:", e)
    traceback.print_exc()
    model = None

# 클래스 이름 (예시)
class_names = ['Large tear', 'Wear / Small tear', 'Shrinkage / Stretching / Wrinkling', 'Buckle / Button / Zipper damage',
'Oil / Food / Chemical stain', 'Ink', 'Mold']

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정 (프론트 연결 시 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인 지정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 예측 엔드포인트
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 전처리 (예: 224x224, 모델에 맞춰 수정 필요)
        image = image.resize((416, 416))
        image_array = np.array(image) / 255.0  # 정규화
        image_array = np.expand_dims(image_array, axis=0)

        # 예측
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

# 로컬 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
