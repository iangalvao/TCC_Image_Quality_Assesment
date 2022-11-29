import datetime
import pandas as pd
import numpy as np
from features import *
import csv


names = [
    "nome",
    "edge_simpl2"
    # "edge_simpl",
    # "hue_simpl",
    # "average_lum",
    # "contrast_ratio",
    # "hist_width",
    # "blur",
    # "score",
]
print("reading data")
images = pd.read_csv("dados/db1_scores.csv")

data = images[["image_name", "MOS_zscore"]]
n = 0
print("starting loop")
with open("dataExp.csv", "w") as fp:
    wr = csv.writer(fp, dialect="excel")
    wr.writerow(names)
    for index, row in data.iterrows():
        print(n)
        name = row["image_name"]
        # score = row["MOS_zscore"]
        path = "dados/1024x768/" + name
        im = Image.open(path)
        imArray = np.array(im)
        out = [name]
        out.extend(extractFeatures(imArray))
        # out.append(score)
        wr.writerow(out)
        im.close()
        n += 1
    fp.close()
