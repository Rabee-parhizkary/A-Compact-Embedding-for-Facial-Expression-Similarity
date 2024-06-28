import os
from pprint import pprint

import pandas as pd
from tqdm import tqdm


def grade_mode(list):
    list_set = set(list)
    frequency_dict = {}
    for i in list_set:
        frequency_dict[i] = list.count(i)
    grade_mode = []
    for key, value in frequency_dict.items():
        if value == max(frequency_dict.values()):
            grade_mode.append(key)
    return grade_mode


def extract(itersize, base, dataset):
    names1 = []
    names2 = []
    names3 = []
    types = []
    modes = []
    for i in tqdm(range(0, itersize)):
        name1 = base + dataset.iloc[i, 0].split('/')[-1]
        sizes1 = tuple(dataset.iloc[i, 1:5])
        name2 = base + dataset.iloc[i, 5].split('/')[-1]
        sizes2 = tuple(dataset.iloc[i, 6:10])
        name3 = base + dataset.iloc[i, 10].split('/')[-1]
        size3 = tuple(dataset.iloc[i, 11:15])
        if not (os.path.exists(name1) and os.path.exists(name2) and os.path.exists(name3)):
            continue
        # print(name1)
        # print(name2)
        # print(name3)
        the_type = dataset.iloc[i, 15]
        modes = grade_mode(
            [dataset.iloc[i, 17], dataset.iloc[i, 19], dataset.iloc[i, 21], dataset.iloc[i, 23], dataset.iloc[i, 25],
             dataset.iloc[i, 27]])
        mode = modes[0]
        # print(mode)

        name1 = ("datas/" + name1, sizes1)
        name2 = ("datas/" + name2, sizes2)
        name3 = ("datas/" + name3, size3)

        if mode == 1:
            names1.append(name2)
            names2.append(name3)
            names3.append(name1)
        elif mode == 2:
            names1.append(name3)
            names2.append(name1)
            names3.append(name2)
        elif mode == 3:
            names1.append(name1)
            names2.append(name2)
            names3.append(name3)

    new_dataset = {}
    new_dataset["anchor"] = names1
    new_dataset["postive"] = names2
    new_dataset["negative"] = names3
    # new_dataset[3]=modes
    # new_dataset[4]=types
    new_data = pd.DataFrame(new_dataset)
    pprint(new_data)
    return new_data
