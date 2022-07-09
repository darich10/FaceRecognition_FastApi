# python

# Fastapi
from fastapi import FastAPI
from .routers import verify


app = FastAPI()
app.include_router(verify.router)


@app.get("/")
async def root():
    return {"Title": "Face Recognition by DeepFace", "By": "Darío Rosas", "Date": "08/07/2022"}


# @app.post("/verify", deprecated=True)
# async def verify(
#         image1: UploadFile = File(...),
#         image2: UploadFile = File(...)
# ):
#     img1 = await image1.read()
#     img1_encoded = np.array(Image.open(io.BytesIO(img1)))
#     img2 = await image2.read()
#     img2_encoded = np.array(Image.open(io.BytesIO(img2)))
#     result = DeepFace.verify(
#         img1_path=img1_encoded,
#         img2_path=img2_encoded,
#         model_name="Facenet",
#         model=model,
#         detector_backend="mtcnn",
#     )
#     return result
