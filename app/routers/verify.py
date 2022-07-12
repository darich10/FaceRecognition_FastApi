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
# Face Recognition Module
from FaceRecognition import face_verify
from FaceRecognition.face_detection import face_detection
from FaceRecognition.face_detection import read_bytes_img


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
    this path operation compare two face images and return if is verified and the distance between embeddings vectors
    :param image1: First person image
    :param image2: Second person image
    :return: ResultsVerify
    """
    img1 = read_bytes_img(image1.file.read())
    img2 = read_bytes_img(image2.file.read())

    img = [face_detection(img1), face_detection(img2)]
    score = face_verify.verify(img)
    return {
        "score": score,
        "verified": True if score >= 0.5 else False
    }
