FROM python:3.8
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY ./start.sh /start.sh
RUN chmod +x /start.sh
COPY ./app /app
COPY ./FaceRecognition /FaceRecognition
CMD ["./start.sh"]