
import os
import cv2
import numpy as np
import argparse
import warnings
import time
import sys 
import config 
warnings.filterwarnings('ignore')
sys.path.append(config.FACE_ANTI_DIR)

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name



class FaceAntiModel():
    """
    Building a Class for Image Spoof Detection
    
    This class serves as a model for detecting fake or spoof images using machine learning techniques.

    Attributes:
    - None (for now): This class might contain attributes related to model parameters, paths to pre-trained models, etc.

    Methods:
    - __init__: Initializes the FaceAntiModel class.
    - predict_image: Accepts an image as input and predicts whether it's genuine or spoof.
    """
    def __init__(self, model_dir):
        self.anti_spoof_model  = AntiSpoofPredict(device_id=0) # Device 0 is use gpu
        self.croper = CropImage()
        self.model_dir = model_dir
        self.dirs    = os.listdir(self.model_dir)

    async def check_image(self, image):
        height, width, channel = image.shape
        if width/height != 3/4:
            return False
        else:
            return True
        
    async def predict_image(self, image):
        # result = await self.check_image(image)
        # if result is False:
        #     return
        image_bbox = self.anti_spoof_model.get_bbox(image)
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in self.dirs:
            h_input, w_input, model_type, scale =  parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = self.croper.crop(**param)
            start = time.time()
            prediction += self.anti_spoof_model.predict(img, os.path.join(self.model_dir, model_name))
            test_speed += time.time()-start

        label = np.argmax(prediction)
        value = prediction[0][label]/2

        if label == 1:
            result_text = {'name':'RealFace : {:.2f}'.format(value), 'label': 1, 'image_box':image_bbox}
        else:
            result_text = {'name':'FakeFace : {:.2f}'.format(value), 'label': 0, 'image_box':image_bbox}
        
        return result_text











