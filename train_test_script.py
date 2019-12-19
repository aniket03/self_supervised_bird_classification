import argparse
import os

import torch
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, utils, models

from dataset_helpers import get_train_test_file_paths_n_labels, split_train_into_train_val, def_data_transform, \
    hflip_data_transform, darkness_jitter_transform, lightness_jitter_transform, rotations_transform, all_in_transform
from get_dataset import GetDataset
from resnet_file import resnet18
from train_test_helper import ModelTrainTest


torch.manual_seed(3)


def visualize(sample_data_loader):

    def imshow(img, mean=0.0, std=1.0):
        """
        Parameters passed:
        img: Image to display
        mean: Mean that was subtracted while normalizing the images
        std: Standard deviation that was used for division while normalizing the image
        """
        img = img * std + mean  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Visualize some MNIST images
    print("Visualization for sample images present in MNIST")
    dataiter = iter(sample_data_loader)
    images, labels = dataiter.__next__()
    imshow(utils.make_grid(images))


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Train test script')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay constant (default: 5e-4)')
    parser.add_argument('--jigsaw-task-weights', type=str, default=None)
    parser.add_argument('--model-file-name', type=str, default='resnet_trained_for_classification.pt')
    parser.add_argument('--experiment-name', type=str, default='e1')
    parser.add_argument('--train-imagenet-based', type=bool, default=False)
    parser.add_argument('--train-ssl-block-4-ft', type=bool, default=False)
    parser.add_argument('--train-ssl-block-3-ft', type=bool, default=False)
    parser.add_argument('--train-ssl-full-ft', type=bool, default=False)
    parser.add_argument('--train-wo-ssl', type=bool, default=False)
    args = parser.parse_args()

    # Set device to use to gpu if available and declare model_file_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    par_weights_dir = 'weights/'
    jigsaw_pre_trained_weights_path =  os.path.join(par_weights_dir, args.jigsaw_task_weights)

    # Data loading and data generators set up
    train_image_ids, test_image_ids, train_file_paths, test_file_paths, train_labels, test_labels = \
        get_train_test_file_paths_n_labels()

    # Get validation files and validation labels separate
    train_image_ids, val_image_ids, train_file_paths, val_file_paths, train_labels, val_labels = \
        split_train_into_train_val(train_image_ids, train_file_paths, train_labels, test_size=0.1)

    # Compute channel means
    channel_means = np.array([124.09, 127.67, 110.50]) / 256.0

    # Define data loaders
    batch_size = args.batch_size
    train_data_loader = DataLoader(
        ConcatDataset(
            [GetDataset(train_file_paths, train_labels, def_data_transform),
             GetDataset(train_file_paths, train_labels, hflip_data_transform),
             GetDataset(train_file_paths, train_labels, darkness_jitter_transform),
             GetDataset(train_file_paths, train_labels, lightness_jitter_transform),
             GetDataset(train_file_paths, train_labels, rotations_transform),
             GetDataset(train_file_paths, train_labels, all_in_transform)]
        ),
        batch_size = batch_size, shuffle = True, num_workers = 8
    )
    val_data_gen = GetDataset(val_file_paths, val_labels, def_data_transform)
    val_data_loader = DataLoader(
        val_data_gen, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_data_gen = GetDataset(test_file_paths, test_labels, def_data_transform)
    test_data_loader = DataLoader(
        test_data_gen, batch_size=batch_size, shuffle=True, num_workers=8
    )

    # Visualize a batch of images
    # visualize(train_data_loader)

    # Train required model defined above on CUB200 data
    num_classes = 200
    epochs = args.epochs
    lr = args.lr
    weight_decay_const = args.weight_decay

    if args.train_imagenet_based:
        model_to_train = models.resnet18(pretrained=True)
        model_to_train.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        model_to_train.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 200),
            nn.LogSoftmax()
        )
        model_file_path = os.path.join(par_weights_dir, 'resnet_imagenet_based.pt')

    elif args.train_wo_ssl:
        model_to_train = resnet18(num_classes=num_classes, siamese_deg=None)
        model_file_path = os.path.join(par_weights_dir, 'resnet_trained_from_scratch.pt')

    else:
        model_to_train = resnet18(num_classes=num_classes, siamese_deg=None)
        model_to_train.fc = nn.Linear(2048 * 9, 200)  # 2048 is the last resnet layer output length which gets
        # multiplied with degree of siamese net, which for jigsaw puzzle solving was 9

        # Load state dict for pre trained model weights
        model_to_train.load_state_dict(torch.load(jigsaw_pre_trained_weights_path))

        # Redefine the last linear layer
        model_to_train.fc = nn.Linear(2048, 200)

        if args.train_ssl_block_4_ft:
            model_file_path = os.path.join(par_weights_dir,
                                           'resnet_trained_ssl_{}_last_a_ft.pt'.format(args.experiment_name))
            for name, param in model_to_train.named_parameters():
                if name[:6] == 'layer4' or name in ['fc.0.weight', 'fc.0.bias']:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif args.train_ssl_block_3_ft:
            model_file_path = os.path.join(par_weights_dir,
                                           'resnet_trained_ssl_{}_last_b_ft.pt'.format(args.experiment_name))
            for name, param in model_to_train.named_parameters():
                if name[:6] == 'layer4' or name[:6] == 'layer3' or name in ['fc.0.weight', 'fc.0.bias']:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        else:
            model_to_train.fc = nn.Linear(2048, 200)
            model_file_path = os.path.join(par_weights_dir,
                                           'resnet_trained_ssl_{}_full_ft.pt'.format(args.experiment_name))


    # Set device on which training is done. Plus optimizer to use.
    model_to_train.to(device)
    sgd_optimizer = optim.SGD(model_to_train.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_const)
    adam_optimizer = optim.Adam(model_to_train.parameters(), lr=lr, weight_decay=weight_decay_const)

    if args.optim == 'sgd':
        optimizer = sgd_optimizer
    else:
        optimizer = adam_optimizer

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, min_lr=1e-5)

    # Start training
    model_train_test_obj = ModelTrainTest(model_to_train, device, model_file_path)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch_no in range(epochs):
        train_loss, train_acc, val_loss, val_acc = model_train_test_obj.train(
            optimizer, epoch_no, params_max_norm=4,
            train_data_loader=train_data_loader, val_data_loader=val_data_loader
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_loss)

    observations_df = pd.DataFrame()
    observations_df['epoch count'] = [i for i in range(1, args.epochs + 1)]
    observations_df['train loss'] = train_losses
    observations_df['val loss'] = val_losses
    observations_df['train acc'] = train_accs
    observations_df['val acc'] = val_accs
    observations_file_path = args.experiment_name + '_observations.csv'
    observations_df.to_csv(observations_file_path)
