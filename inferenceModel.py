import cv2
from mltu.configs import BaseModelConfigs
from model import ImageToWordModel
import os
import csv
from pathConfigs import *
from Levenshtein import distance
import pandas as pd
from tqdm import tqdm
import numpy as np

test_path = os.path.join(TEST_DATASET_PATH)

configs = BaseModelConfigs.load(MODEL_PATH + 'configs.yaml')

model = ImageToWordModel(
    model_path=configs.model_path, char_list=configs.vocab)

''' Test with single image '''

# image_path = 'path_to_img'

# image = cv2.imread(image_path)
# prediction_text = model.predict(image)

# print("Image: ", image_path)
# print("Prediction: ", prediction_text)

''' Test with test dataset '''

# with open('submission.csv', 'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['id', 'answer'])
#     accum_score = []
#     for foldername in os.listdir(test_path):
#         for filename in os.listdir(os.path.join(test_path, foldername)):
#             image = cv2.imread(os.path.join(
#                 test_path, foldername, filename))

#             file_path = foldername + '/' + filename
#             prediction_text = model.predict(image)

#             accum_score.append(distance(prediction_text, file_path))

#             writer.writerow([file_path, prediction_text])

#             print("Image: ", file_path)
#             print("Prediction: ", prediction_text)

''' Evaluate with validation dataset '''

df = pd.read_csv(MODEL_PATH + 'val.csv').values.tolist()
accum_score = []

for image_path, label in tqdm(df):
    image = cv2.imread(image_path)
    prediction_text = model.predict(image)
    d = distance(prediction_text, label)
    score = 1 if prediction_text == label else max(
        0, 1 - pow(1.5, d) / len(label))
    accum_score.append(score)

print(f"Average score: {np.average(accum_score)}")
