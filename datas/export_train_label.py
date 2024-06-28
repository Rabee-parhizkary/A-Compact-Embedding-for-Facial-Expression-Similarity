import pandas as pd

from datas.utils import extract

dataset = pd.read_csv('faceexp-comparison-data-train-public.csv', header=None, on_bad_lines="skip")

base = "train/trainData/train/"
itersize = 63000

new_data = extract(dataset=dataset, base=base, itersize=itersize)
print("DataFrame's cols is {}".format(new_data.shape[0]))
new_data.to_csv('pd_triplet_data.csv')