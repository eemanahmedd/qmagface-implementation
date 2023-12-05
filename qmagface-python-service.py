import os
import sys
import cv2
import numpy as np
from flask import Flask, request, jsonify
import torch
import time
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import euclidean
import torchvision.transforms
from torch.utils import data
from collections import namedtuple
from utils.files import list_all_files
from preprocessing.magface.network_inf import builder_inf
#from preprocessing.insightface.src.mtcnn_detector import MtcnnDetector
from preprocessing.insightface.src import face_preprocess
from similarity.base import Similarity
from scipy.spatial import distance
from dotenv import load_dotenv

known_faces = None
model = None
model_loaded = False

def crop_image(img):
    _, width, _ = img.shape
    crop_width = int(width * 0.7)
    cropped_image = img[:, :crop_width, :] # crop the left side of the image
    return cropped_image

def get_face_from_image(image_path, detector):
    image = cv2.imread(image_path)
    cropped_image = crop_image(image)
    faces = detector.detect_faces(np.array(cropped_image))

    if len(faces) > 0:
        face = faces[0]
        x1, y1, width1, height1 = face['box']
        extracted_face = cropped_image[y1:y1 + height1, x1:x1 + width1]
        resized_face = cv2.resize(np.array(extracted_face), (112,112))
        return resized_face
    else:
        return None
    

def get_known_embeddings(dir, detector, model, trans):
    local_known_faces = {}

    for people in os.listdir(dir):
        people_dir = os.path.join(dir, people)
        encoding_list = []

        for filename in os.listdir(people_dir):
            encoding_dict = {}
            image_path = os.path.join(people_dir, filename)
            face = get_face_from_image(image_path, detector)

            if face is not None:
                torch_face = torch.unsqueeze(trans(face),0)
                face_embedding = model(torch_face).squeeze().cpu().detach().numpy()
                encoding_dict['filename'] = filename
                encoding_dict['embedding'] = face_embedding
                encoding_list.append(encoding_dict)
            else:
                print("no face found :( ")
        local_known_faces[people] = encoding_list
    return local_known_faces


# Load environment variables from .env file
load_dotenv()

# Access the variables
model_path = os.getenv("model_path")
upload_folder = os.getenv("upload_folder")

if model is None:

    detector = MTCNN()

    Args = namedtuple('Args', ['arch', 'resume', 'embedding_size', 'cpu_mode'])
    args = Args('iresnet100', model_path, 512, True)

    start_time = time.time()

    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    print('model load time: ', time.time()-start_time)

    model.eval()
    trans = torchvision.transforms.ToTensor()
    model_loaded =  True

if known_faces is None:
    known_faces = get_known_embeddings('./people', detector, model, trans)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    global model_loaded
    if model_loaded:
        return jsonify(status='ok', message='Model is ready'), 200
    else:
        return jsonify(status='not ready', message='Model is still loading'), 503
    
@app.route('/process_frame', methods=['POST'])
def process_frame():
    if request.method == 'POST':
        start_time = time.time()
        global model_loaded

        if not model_loaded:
            return jsonify(status='not ready', message='Model is still loading'), 503

        
        data = request.files['image']
        data.save('img.jpg')

        face = get_face_from_image('img.jpg', detector)
        face = torch.unsqueeze(trans(face),0)
        face_embedding = model(face).squeeze().cpu().detach().numpy()
        # person_list = []
        # filename_list = []
        # similarity_list = []
        result_list = []

        # Compare with known embeddings
        for person, file_and_embs_list in known_faces.items():
            for file_dict in file_and_embs_list:
                filename = file_dict['filename']
                embedding = file_dict['embedding']
                similarity = distance.euclidean(face_embedding, embedding)
                # print(f'similarity: {similarity}')

                # person_list.append(person)
                # filename_list.append(filename)
                # similarity_list.append(float(similarity))
                temp_dict = {'person': person, 'filename': filename, 'similarity': float(similarity)}
                result_list.append(temp_dict)
            

        # temp_dict = {'person': person_list, 'filename': filename_list, 'similarity': similarity_list}
        end_time = time.time()
        print(f'Time taken: {end_time - start_time}')
        return jsonify(result_list)

if __name__ == '__main__':
    size_of_function = sys.getsizeof(process_frame)
    print(f"Size of function: {size_of_function} bytes")
    app.run(debug=True)
