from skimage import io
from skimage.transform import resize
from mtcnn.mtcnn import MTCNN
import numpy as np
from matplotlib import pyplot as plt


def read_image(path: str) -> np:
    """
    Read an image by PIL
    :param path: image file path
    :return: image numpy
    """
    image = io.imread(path)
    return image


def find_face(image: np) -> np:
    """
    Find a face in the image by the mtcnn method
    :param image: Original image
    :return: Face image if exist
    """
    detector = MTCNN()
    # create the detector, using default weights
    # detect faces in the image
    results = detector.detect_faces(image)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    return face


def face_detection(path: str):
    image = read_image(path)
    image = find_face(image)
    image = resize(image, (112, 112))
    return image


if __name__ == "__main__":
    img = face_detection("../images/0000001.jpg")
    print(img.shape)
    plt.imshow(img, interpolation='nearest')
    plt.show()
