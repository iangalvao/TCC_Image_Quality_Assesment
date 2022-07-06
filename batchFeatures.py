import datetime
import pandas as pd
import numpy as np
from features import *
import csv


names = ["edge_simpl", "hue_simpl", "average_lum",
         "contrast_ratio", "hist_width", "blur", "score"]

df = pd.read_csv("dados/db1_scores.csv")
df = df.sort_values(by=["MOS_zscore"])
imagesData = df.iloc[np.arange(0, 11000, 1000)]
bestImages = df.iloc[9000:]
print("bestImages:", len(bestImages))
data = df[["image_name", "MOS_zscore"]]
targetHour = 17
targetMin = 55
n = 0
with open("dataBest.csv", "w") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(names)
    for index, row in data.iterrows():
        now = datetime.datetime.now()
        hour = now.hour
        min = now.minute
        if ((hour == targetHour) and (min >= targetMin)) or (hour > targetHour):
            fp.close()
            break
        else:
            name = row["image_name"]
            score = row["MOS_zscore"]
            path = "dados/1024x768/" + name
            print(path)
            print(n)
            im = Image.open(path)
            imArray = np.array(im)
            out = [name]
            out.extend(extractFeatures(imArray))
            out.append(score)
            wr.writerow(out)
            im.close()
            n += 1
    fp.close()
