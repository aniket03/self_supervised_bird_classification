import argparse
import os

import numpy as np
import pandas as pd

import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch import nn
from torchvision import models
from torch.autograd import Variable
8
from dataset_helpers import get_train_test_file_paths_n_labels, def_data_transform, split_train_into_train_val
from get_dataset import GetJigsawPuzzleDataset, GetDataset
from resnet_file import resnet18
from train_test_helper import JigsawModelTrainTest, ModelTrainTest


def pil_loader(path):
    pil_img = Image.open(path)
    if pil_img.mode == "L":
        return None
    else:
        return pil_img



if __name__ == '__main__':

    # Eval arguments
    parser = argparse.ArgumentParser(description='Eval script')
    parser.add_argument('--model-name', type=str, default='resnet_trained_ssl_e8_last_b_b_ft.pt')
    parser.add_argument('--test-compact-bilinear', type=bool, default=False)
    parser.add_argument('--test-imagenet-based', type=bool, default=True)
    parser.add_argument('--test-on', type=str, default='test')  # Whether test on train, val or test set
    args = parser.parse_args()

    # Set device to use to gpu if available and declare model_file_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    par_weights_dir = 'weights/'
    model_file_path = os.path.join(par_weights_dir, args.model_name)

    # Data loading and data generators set up
    train_image_ids, test_image_ids, train_file_paths, test_file_paths, train_labels, test_labels = \
        get_train_test_file_paths_n_labels()

    train_image_ids, val_image_ids, train_file_paths, val_file_paths, train_labels, val_labels = \
        split_train_into_train_val(train_image_ids, train_file_paths, train_labels, test_size=0.1)

    if args.test_imagenet_based:
        model_to_train = models.resnet18(pretrained=True)
        model_to_train.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        model_to_train.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 200),
            nn.LogSoftmax()
        )
    else:
        model_to_train = resnet18(num_classes=200, siamese_deg=None)

    # Check if saved model exists, and load if it does.
    if os.path.exists(model_file_path):
        model_to_train.load_state_dict(torch.load(model_file_path))
    model_to_train.to(device)

    # Setup on which set evaluation is to be carried out
    if args.test_on == 'train':
        eval_file_paths, eval_labels = train_file_paths, train_labels
    elif args.test_on == 'val':
        eval_file_paths, eval_labels = val_file_paths, val_labels
    else:
        eval_file_paths, eval_labels = test_file_paths, test_labels

    # Start evaluation
    model_to_train.eval()
    correct = 0
    preds = []
    for f, label in zip(eval_file_paths, eval_labels):
        pil_img = pil_loader(f)
        if pil_img is None:
            preds.append(0)
            continue
        data = def_data_transform(pil_img)
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        data = Variable(data, volatile=True).to(device)
        output = model_to_train(data)
        pred = output.data.max(1, keepdim=True)[1]

        x = pred.data
        preds.append(x)

        if x == label:
            correct += 1

    print (correct, len(eval_file_paths), correct * 100 / len(eval_file_paths))
    conf_mat = np.array(confusion_matrix(eval_labels, preds))
    conf_df = pd.DataFrame(conf_mat)
    conf_df.columns = np.arange(1,201)
    conf_df.to_csv('confusion_matrix.csv')
