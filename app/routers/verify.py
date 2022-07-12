# fastapi
import PIL.ImageShow
from fastapi import APIRouter
from fastapi import UploadFile, File
from pydantic import BaseModel
# Python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import logging
# Face Recognition Module
from FaceRecognition import face_verify
from FaceRecognition.face_detection import face_detection
from FaceRecognition.face_detection import read_bytes_img

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/verify",
    tags=["Face Recognition"]
)


class ResultsVerify(BaseModel):
    """
    Result class to returnn
    """
    verified: bool
    score: float
    max_threshold_to_verify: float = 0.5


@router.post("/", response_model=ResultsVerify)
async def verify(
        image1: UploadFile = File(...),
        image2: UploadFile = File(...)
):
    """
    This path operation compare two face images and return if is verified and the distance between embeddings vectors
    :param image1: First person image
    :param image2: Second person image
    :return: ResultsVerify: Model response
    """
    img1 = read_bytes_img(image1.file.read())
    logger.info(f"Image read: {image1.filename}")
    img2 = read_bytes_img(image2.file.read())
    logger.info(f"Image read: {image2.filename}")
    img = [face_detection(img1), face_detection(img2)]
    logger.info("Faces detected")
    score = face_verify.verify(img)
    logger.info("Face Verify")
    return {
        "score": score,
        "verified": True if score >= 0.5 else False
    }
