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

# Init model 
face_anti_model = FaceAntiModel(config.ANTI_MODEL_DIR)


face_recognize_mode = RecognizeModel(config.DATASET_DIR,
                                     weight_dir=config.MODEL_WEIGHT_DIR,
                                     model_name=config.MODEL_NAME['facenet'])

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...), folder_name: str = None):
    save_directory = config.DATASET_DIR 
    
    img_path = os.path.join(save_directory, folder_name)

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

        if result_recognize['confidence'] < 85: 
            print("Low confidence")
            result_recognize['name'] = 'UNKNOWN'

        app_logger.log_infor(result_recognize)

        return JSONResponse(content={"data": result_recognize}, status_code=200)


