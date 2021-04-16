import torch
from skimage.io import imread
from torch.utils import data
import pandas as pd
import numpy as np

df = pd.read_csv("understanding_cloud_organization/train_320.csv")

def rle_decode(mask_rle: str = '', shape: tuple = (320, 480)):
    """
    Decode rle encoded mask.
    Args:
        mask_rle: encoded mask
        shape: final shape
    Returns:
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (320, 480)):
    """
    Create mask based on df, image name and shape.
    Args:
        df: dataframe with cloud dataset
        image_name: image name
        shape: final shape
    Returns:
    """

    encoded_masks = df.loc[df['Image'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def mask2rle(img):
    """
    Convert mask to rle.
    Args:
        img:
    Returns:
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        
        shape = (320, 480)

        # Load input and target
        x, y = self.inputs[index], self.targets[index]

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)


        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        
        return x, y
    
