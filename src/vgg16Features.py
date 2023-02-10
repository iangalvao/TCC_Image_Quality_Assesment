# example of using the vgg16 model as a feature extraction model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from pickle import dump

import pandas as pd

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

print("reading data")
images = pd.read_csv("dados/db1_scores.csv")

data = images[["image_name", "MOS_zscore"]]
n = 0
print("starting loop")

for index, row in data.iterrows():
    name = row["image_name"]
    print(name, n)
    score = row["MOS_zscore"]
    path = "dados/1024x768/" + name
    # load an image from file
    image = load_img(path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    features = model.predict(image)
    out = [name]
    out.extend(features)
    out.append(score)
    print(out)
    n += 1
    # save to file
    dump(out, open("vgg16.pkl", "ab"))
