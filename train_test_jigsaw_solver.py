import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torchvision.transforms import transforms

from dataset_helpers import get_train_test_file_paths_n_labels, get_inat_birds_file_paths, \
    get_na_birds_file_paths
from get_dataset import GetJigsawPuzzleDataset
from resnet_file import resnet18
from train_test_helper import JigsawModelTrainTest


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Jigsaw Train test script')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay constant (default: 5e-4)')
    parser.add_argument('--experiment-name', type=str, default='e1_js')
    parser.add_argument('--dataset-config', type=str, default='js_d1')
    args = parser.parse_args()

    # Data files which will get referred
    permuts_file_path = 'selected_permuts.npy'

    # Set device to use to gpu if available and declare model_file_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    par_weights_dir = 'weights/'
    model_file_path = os.path.join(par_weights_dir, 'resnet_jigsaw_solver_{}_trained.pt'.format(args.experiment_name))

    # Data loading and data generators set up
    # Get image file paths, ids and labels from CUB-200 dataset
    cub_tr_image_ids, cub_te_image_ids, cub_train_file_paths, cub_test_file_paths, cub_tr_labels, cub_te_labels = \
        get_train_test_file_paths_n_labels()

    # Get image file paths from INAT dataset
    inat_file_paths = get_inat_birds_file_paths()

    # Get image file paths from NA Birds dataset
    na_file_paths = get_na_birds_file_paths()

    # Add the file paths from cub, inat and na together
    if args.dataset_config == 'js_d1':
        all_file_paths = cub_train_file_paths
    else:
        all_file_paths = cub_train_file_paths + inat_file_paths + na_file_paths

    # Get validation files separate
    train_file_paths, val_file_paths = train_test_split(all_file_paths, test_size=0.1, shuffle=True, random_state=3)

    # Compute channel means
    channel_means = np.array([124.09, 127.67, 110.50]) / 256.0

    # Define data transforms
    data_transform = transforms.Compose([
        transforms.RandomCrop((64, 64)),
        transforms.ColorJitter(brightness=[0.5, 1.5]),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Define data loaders
    batch_size = args.batch_size

    if args.dataset_config == 'js_d1':
        train_data_loader = DataLoader(
            ConcatDataset(
                [GetJigsawPuzzleDataset(train_file_paths, permuts_file_path,
                                        range_permut_indices=[st_perm_ind, st_perm_ind+9], transform=data_transform)
                 for st_perm_ind in range(0, 200, 10)
                ]
            ),
            batch_size=batch_size, shuffle=True, num_workers=8
        )
        val_data_loader = DataLoader(
            ConcatDataset(
                [GetJigsawPuzzleDataset(val_file_paths, permuts_file_path,
                                        range_permut_indices=[st_perm_ind, st_perm_ind + 9], transform=data_transform)
                 for st_perm_ind in range(0, 200, 10)
                 ]
            ),
            batch_size=batch_size, shuffle=True, num_workers=8
        )
    else:
        train_data_loader = DataLoader(
            GetJigsawPuzzleDataset(train_file_paths, permuts_file_path, transform=data_transform),
            batch_size=batch_size, shuffle=True, num_workers=8
        )
        val_data_loader = DataLoader(
            GetJigsawPuzzleDataset(val_file_paths, permuts_file_path, transform=data_transform),
            batch_size=batch_size, shuffle=True, num_workers=8
        )

    # Print sample batches that would be returned by the train_data_loader
    dataiter = iter(train_data_loader)
    X, y = dataiter.__next__()
    print (X.size())
    print (y.size())

    # Train required model defined above on CUB200 data
    num_outputs = 200
    epochs = args.epochs
    lr = args.lr
    weight_decay_const = args.weight_decay

    # If using Resnet18
    model_to_train = resnet18(num_classes=num_outputs, siamese_deg=9)

    # Set device on which training is done. Plus optimizer to use.
    model_to_train.to(device)
    optimizer = optim.SGD(model_to_train.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_const)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, min_lr=1e-5)

    # Start training
    model_train_test_obj = JigsawModelTrainTest(model_to_train, device, model_file_path)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch_no in range(epochs):
        train_loss, train_acc, val_loss, val_acc = model_train_test_obj.train(
            optimizer, epoch_no, params_max_norm=4,
            train_data_loader = train_data_loader, val_data_loader = val_data_loader
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
