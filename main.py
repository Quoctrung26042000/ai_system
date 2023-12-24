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

# Import AI model
from recognize_predict import RecognizeModel

from face_anti_model import FaceAntiModel

# Init model 
face_anti_model = FaceAntiModel(config.ANTI_MODEL_DIR)


face_recognize_mode = RecognizeModel(config.DATASET_DIR,
                                     weight_dir=config.MODEL_WEIGHT_DIR,
                                     model_name=config.MODEL_NAME['facenet'])

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    image_bytes = BytesIO(contents)
    image = Image.open(image_bytes)
    np_image = np.array(image)

    result_anti = face_anti_model.predict_image(image=np_image)

    if result_anti['label'] == 0:
        app_logger.log_infor(result_anti)

        return JSONResponse(content={"message": result_anti}, status_code=200)
    
    else :
        result_recognize = face_recognize_mode.recognize_predict(np_image)

        if result_recognize['confidence'] < 87: 
            print("Low confidence")
            result_recognize['name'] = 'UNKNOWN'

        app_logger.log_infor(result_recognize)

        return JSONResponse(content={"message": result_recognize}, status_code=200)


