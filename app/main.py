# python
import logging

# Fastapi
from fastapi import FastAPI
from .routers import verify

# TODO: Error Handling
# TODO: Docs
# TODO: Dockerfile
# TODO: Post Heroku

app = FastAPI()
app.include_router(verify.router)


@app.get(path="/", tags=["Home"])
async def root():
    """
    Operation path home
    :return:
    """
    return {"Title": "Face Recognition by DeepFace", "By": "Darío Rosas", "Date": "08/07/2022"}

