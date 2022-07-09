from fastapi import APIRouter
from fastapi import UploadFile, File
from pydantic import BaseModel

import numpy as np
from PIL import Image
import io

from FaceRecognition.face_verify import verify
from FaceRecognition.face_detection import face_detection

router = APIRouter(
    prefix="/verify",
    tags=["Face Recognition"]
)


class ResultsVerify(BaseModel):
    verified: float
    distance: float
    max_threshold_to_verify: float = 0.4


@router.post("/")
async def verify(
        image1: UploadFile = File(...),
        image2: UploadFile = File(...)
):
    img1 = image1.file.read()
    img1_encoded = np.array(Image.open(io.BytesIO(img1)))
    img2 = image2.file.read()
    img2_encoded = np.array(Image.open(io.BytesIO(img2)))
    img = [face_detection(img1_encoded), face_detection(img2_encoded)]
    score = verify(img)
    return {"score": score}
