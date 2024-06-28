'''FEC Dataset class'''

import pprint
import pandas as pd
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

path_pd = "datas/pd_triplet_data.csv"
path_test = "datas/pd_triplet_data_test.csv"


class FecData(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """

    def __init__(self, path, transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(path)
        self.data = self.pd_data.to_dict("list")
        self.data_anc = self.data['anchor']
        self.data_pos = self.data["postive"]
        self.data_neg = self.data["negative"]

    def __len__(self):
        return len(self.data["anchor"])

    def __getitem__(self, index):
        anc_list = eval(self.data_anc[index])
        anc_img = self.get_image(anc_list)

        pos_list = eval(self.data_pos[index])
        pos_img = self.get_image(pos_list)

        neg_list = eval(self.data_neg[index])
        neg_img = self.get_image(neg_list)

        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        return anc_img, pos_img, neg_img

    @staticmethod
    def get_image(inp_list):
        inp_img = Image.open(inp_list[0])
        wid, hei = inp_img.size
        inp_img = inp_img.crop((inp_list[1][0] * wid, inp_list[1][2] * hei, inp_list[1][1] * wid, inp_list[1][3] * hei))
        if inp_img.getbands()[0] != 'R':
            inp_img = inp_img.convert('RGB')
        return inp_img
