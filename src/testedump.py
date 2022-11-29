from pickle import dump

for i in range(4):
    features = [i, 2 * i, 3 * i, 4 * i]
    dump(features, open("dump3.pkl", "ab"))
