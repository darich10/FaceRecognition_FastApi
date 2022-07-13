# python
import logging

# Fastapi
from fastapi import FastAPI
from .routers import verify

# TODO: Post Heroku

FORMAT = "%(levelname)s:%(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(verify.router)


@app.get(path="/", tags=["Home"])
async def root():
    """
    Operation path home
    :return:
    """
    logger.info("Home Face Verification")
    return {"Title": "Face Recognition", "By": "Darío Rosas", "Date": "08/07/2022"}

