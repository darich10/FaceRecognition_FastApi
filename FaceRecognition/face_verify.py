from typing import List

from FaceRecognition.face_detection import face_detection
from tensorflow.keras.models import load_model
import numpy as np


model = load_model("FaceRecognition/keras_models/ArcFace")  # Load model


def predict(face_image: np) -> np:
    """
    Obtain the embeddings from a face image
    :param face_image: An image of  a face
    :return: nompy array embedding
    """
    face_image = np.expand_dims(face_image, axis=0)
    embedding = model.predict(face_image)
    return embedding


def cosine_similarity(list_embeddings: List) -> np:
    """
    Compute cosine similarity between two vectors
    :param list_embeddings: A list of embedding vector of each person
    :return: Return the cosine similarity with values between -1 and 1
    """
    emb1 = list_embeddings[0][0]
    emb2 = list_embeddings[1][0]
    cos_sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return cos_sim


def verify(list_image: List) -> float:
    """
    From a list of images, compute their embeddings and compute the cosine similarity
    :param list_image: A list of face images to compare
    :return: The cosine similarity between faces images
    """
    embedding_list = []
    for im in list_image:
        embedding_list.append(predict(im))
    score = cosine_similarity(embedding_list)
    print(score)
    return float(score)


if __name__ == "__main__":
    img = [face_detection("../images/0000001.jpg"), face_detection("../images/0001_0000255_script.jpg")]
    print(verify(img))
    # print(cosine_similarity([np.array( [4, 47, 8, 3]), np.array( [3, 52, 12, 16])]))