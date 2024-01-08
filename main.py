import os
import cv2
import argparse
import warnings
import time
import sys 
from typing import Union
from io import BytesIO
from PIL import Image
import numpy as np
import config
warnings.filterwarnings('ignore')
import service
from datetime import datetime

# Import Log
from log.logger import Logger
app_logger = Logger(log_file='app.log')


# Import Fastapi framework
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import AI model
from recognize_predict import RecognizeModel

from face_anti_model import FaceAntiModel

from resources import strings

import utils

# Init model 
face_anti_model = FaceAntiModel(config.ANTI_MODEL_DIR)


face_recognize_mode = RecognizeModel(config.DATASET_DIR,
                                     weight_dir=config.MODEL_WEIGHT_DIR,
                                     model_name=config.MODEL_NAME['facenet'])

import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


import subprocess
@app.post("/train_model")
def train_model():
    try:
        # Construct the command to run the train.py script with the provided data_dir
        command = f"./ai_env/bin/python train_face_recognize.py --data_dir=datasets"
        
        # Run the script using subprocess
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Assuming face_recognize_model is the global RecognizeModel instance
        global face_recognize_model
        face_recognize_model = RecognizeModel(config.DATASET_DIR,
                                                weight_dir=config.MODEL_WEIGHT_DIR,
                                                model_name=config.MODEL_NAME['facenet'])

        # command_service = f"systemctl restart ai.service"
        
        if process.returncode == 0:
            # Run the script using subprocess
            # process_service = subprocess.Popen(command_service, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # process_service.communicate()
            return {"message": "Training completed successfully!", "stdout": stdout.decode(), "stderr": stderr.decode()}, 200
        else:
            return {"message": "Training failed!", "stdout": stdout.decode(), "stderr": stderr.decode()}, 500
    except Exception as e:
        return {"message": f"Error during training: {str(e)}"}, 500
    
@app.post("/deploy_new_model")
def deploy_new_model():
    try:
    # Construct the command to run the train.py script with the provided data_dir
        command = f"systemctl restart ai.service"
        
        # Run the script using subprocess
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        return {"message": "Deploy successfully!", "stdout": stdout.decode(), "stderr": stderr.decode()}, 200
    except Exception as e:
        return {"message": f"Error during training: {str(e)}"}, 500


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...), folder_name: str = None):
    save_directory = config.DATASET_DIR 

    folder_name_normalized = utils.normalize_folder_name(folder_name)
    
    img_path = os.path.join(save_directory, folder_name_normalized)

    if not os.path.exists(img_path):
        os.makedirs(img_path)  

    for file in files:
        contents = await file.read()
        file_path = os.path.join(img_path, file.filename)  

        with open(file_path, "wb") as f:
            f.write(contents) 

    return {"message": f"{len(files)} images uploaded and saved successfully in the datasets folder"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    global face_recognize_model
    if face_recognize_model is None:
        app_logger.log_infor(message=strings.MODEL_NOT_TRAIN)
        return {"message": strings.MODEL_NOT_TRAIN}, 404
    
    contents = await file.read()

    image_bytes = BytesIO(contents)
    image = Image.open(image_bytes)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    np_image = np.array(image)

    result_anti = await face_anti_model.predict_image(image=np_image)

    if result_anti['label'] == 0:
        app_logger.log_infor(result_anti)

        return JSONResponse(content={"data": result_anti}, status_code=200)
    
    else :
        result_recognize = face_recognize_mode.recognize_predict(np_image)
        result_recognize['image_box'] = result_anti['image_box']
        # print('confidence', result_recognize['confidence'])
        app_logger.log_infor(result_recognize['confidence'])
        if result_recognize['confidence'] < 75: 
            # print("Low confidence")
            result_recognize['name'] = strings.PEOPLE_UNKNOWN
        # else :
        #     current_time = datetime.now()

        #     # Attendance Student 
        #     connection = await service.connect_to_db()
        #     query_update = """
        #     UPDATE attendance
        #     SET check_in = $1, status = $2
        #     WHERE student_id = $3
        #     AND Date(create_at) = CURRENT_DATE
        #     RETURNING student_id, check_in
        #     """
        #     query_search= """
        #     SELECT name 
        #     FROM student 
        #     WHERE id = $1
        #     """
        #     student_update = await connection.fetchrow(query_update,
        #                                           current_time,
        #                                           1,
        #                                           int(result_recognize['name']))
            
        #     student = await connection.fetchrow(query_search,
        #                                             student_update['student_id'])
            
        #     result_recognize['name'] = student['name']
        
        app_logger.log_infor(result_recognize)

        return JSONResponse(content={"data": result_recognize}, status_code=200)


