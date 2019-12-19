import json
import numpy as np
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset_helpers import crop_from_center, get_nine_crops


class GetDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_paths, labels, transform=None):
        'Initialization'
        self.imgs = [(img_path, label) for img_path, label in zip(file_paths, labels)]
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_paths)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        file_path = self.file_paths[index]
        label = self.labels[index]
        pil_image = Image.open(file_path)

        # Check if image has only single channel. If True, then swap with 0th image
        # Assumption 0th image has got 3 number of channels
        if len(pil_image.getbands()) != 3:
            file_path = self.file_paths[0]
            label = self.labels[0]
            pil_image = Image.open(file_path)

        # Convert image to torch tensor
        tr_image = self.transform(pil_image)

        return tr_image, label


class GetJigsawPuzzleDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_paths, avail_permuts_file_path, range_permut_indices=None, transform=None):
        'Initialization'
        self.file_paths = file_paths
        self.transform = transform
        self.permuts_avail = np.load(avail_permuts_file_path)
        self.range_permut_indices = range_permut_indices

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_paths)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        file_path = self.file_paths[index]
        pil_image = Image.open(file_path)

        # Check if image has only single channel. If True, then swap with 0th image
        # Assumption 0th image has got 3 number of channels
        if len(pil_image.getbands()) != 3:
            file_path = self.file_paths[0]
            pil_image = Image.open(file_path)

        # Convert image to torch tensor
        pil_image = pil_image.resize((256, 256))
        pil_image = crop_from_center(pil_image, 225, 225)

        # Get nine crops for the image
        nine_crops = get_nine_crops(pil_image)

        # Permut the 9 patches obtained from the image
        if self.range_permut_indices:
            permut_ind = random.randint(self.range_permut_indices[0], self.range_permut_indices[1])
        else:
            permut_ind = random.randint(0, len(self.permuts_avail) - 1)

        permutation_config = self.permuts_avail[permut_ind]

        permuted_patches_arr = [None] * 9
        for crop_new_pos, crop in zip(permutation_config, nine_crops):
            permuted_patches_arr[crop_new_pos] = crop

        # Apply data transforms
        # TODO: Remove hard coded values from here
        tensor_patches = torch.zeros(9, 3, 64, 64)
        for ind, jigsaw_patch in enumerate(permuted_patches_arr):
            jigsaw_patch_tr = self.transform(jigsaw_patch)
            tensor_patches[ind] = jigsaw_patch_tr

        return tensor_patches, permut_ind
