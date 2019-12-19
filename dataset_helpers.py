import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms


def_data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

hflip_data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

darkness_jitter_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ColorJitter(brightness=[0.5, 0.9]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

lightness_jitter_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ColorJitter(brightness=[1.1, 1.5]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

rotations_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=15),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

all_in_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=[0.5, 1.5]),
    transforms.RandomRotation(degrees=15),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def crop_from_center(pil_image, new_h, new_w):

    width, height = pil_image.size  # Get dimensions

    left = (width - new_w) / 2
    top = (height - new_h) / 2
    right = (width + new_w) / 2
    bottom = (height + new_h) / 2

    # Crop the center of the image
    pil_image = pil_image.crop((left, top, right, bottom))

    return pil_image


def get_nine_crops(pil_image):
    """
    Get nine crops for a square pillow image. That is height and width of the image should be same.
    :param pil_image: pillow image
    :return: List of pillow images. The nine crops
    """
    w, h = pil_image.size
    diff = int(w/3)

    r_vals = [0, diff, 2 * diff]
    c_vals = [0, diff, 2 * diff]

    list_patches = []

    for r in r_vals:
        for c in c_vals:

            left = c
            top = r
            right = c + diff
            bottom = r + diff

            patch = pil_image.crop((left, top, right, bottom))
            list_patches.append(patch)

    return list_patches


def split_train_into_train_val(train_file_ids, train_file_paths, train_labels, test_size=0.1):
    """
    Split train_file_paths and train_labels to train_file_paths, val_file_paths and
    train_labels, val_labels
    """

    # Create a mapping between image_id and file_path
    image_id_name_map = dict(zip(train_file_ids, train_file_paths))

    # Get validation files and validation labels separate
    train_file_ids, val_file_ids, train_labels, val_labels = train_test_split(
        train_file_ids, train_labels, test_size=test_size, random_state=5, shuffle=True
    )
    train_file_paths = [image_id_name_map[image_id] for image_id in train_file_ids]
    val_file_paths = [image_id_name_map[image_id] for image_id in val_file_ids]

    print ("Length of train files list", len(train_file_paths))
    print ("Length of train labels", len(train_labels))
    print ("Length of val files list", len(val_file_paths))
    print ("Length of val labels", len(val_labels))

    return train_file_ids, val_file_ids, train_file_paths, val_file_paths, train_labels, val_labels


def get_inat_birds_file_paths():
    par_data_dir = '../inat_birds/'
    species_dirs = os.listdir(par_data_dir)
    file_paths_to_return = []

    for species_dir in species_dirs:

        if species_dir.isdigit():  # Since species directories only have digits in their names
            species_dir_path = os.path.join(par_data_dir, species_dir)
            all_files = os.listdir(species_dir_path)

            for file_name in all_files:
                if file_name[-3:] == 'jpg':
                    file_paths_to_return.append(os.path.join(species_dir_path, file_name))

    return file_paths_to_return


def get_na_birds_file_paths():
    par_data_dir = '../nabirds/images'
    species_dirs = os.listdir(par_data_dir)
    file_paths_to_return = []

    for species_dir in species_dirs:

        if species_dir.isdigit():  # Since species directories only have digits in their names
            species_dir_path = os.path.join(par_data_dir, species_dir)
            all_files = os.listdir(species_dir_path)

            for file_name in all_files:
                if file_name[-3:] == 'jpg':
                    file_paths_to_return.append(os.path.join(species_dir_path, file_name))

    return file_paths_to_return


def get_train_test_file_paths_n_labels():
    """
    Get array train_file_paths, train_labels, test_file_paths and test_labels
    """

    # Data loading and data generators set up
    par_data_dir = 'CUB_200_2011/CUB_200_2011'
    images_data_dir = 'CUB_200_2011/CUB_200_2011/images'
    train_test_split_file = os.path.join(par_data_dir, 'train_test_split.txt')
    images_file = os.path.join(par_data_dir, 'images.txt')
    labels_file = os.path.join(par_data_dir, 'image_class_labels.txt')

    # Read the images_file which stores image-id and image-name mapping
    image_file_id_df = pd.read_csv(images_file, sep=' ', header=None)
    image_file_id_mat = image_file_id_df.as_matrix()
    image_id_name_map = dict(zip(image_file_id_mat[:, 0], image_file_id_mat[:, 1]))

    # Read the train_test_split file which stores image-id and train-test split mapping
    image_id_train_test_split_df = pd.read_csv(train_test_split_file, sep=' ', header=None)
    image_id_train_test_split_mat = image_id_train_test_split_df.as_matrix()
    image_id_train_test_split_map = dict(zip(image_id_train_test_split_mat[:, 0],
                                             image_id_train_test_split_mat[:, 1]))

    # Read the image class labels file
    image_id_label_df = pd.read_csv(labels_file, sep=' ', header=None)
    image_id_label_mat = image_id_label_df.as_matrix()
    image_id_label_map = dict(zip(image_id_label_mat[:, 0], image_id_label_mat[:, 1]))

    # Put together train_files train_labels test_files and test_labels lists
    train_image_ids, test_image_ids = [], []
    train_file_paths, test_file_paths = [], []
    train_labels, test_labels = [], []
    for file_id in image_id_name_map.keys():
        file_name = image_id_name_map[file_id]
        is_train = image_id_train_test_split_map[file_id]
        label = image_id_label_map[file_id] - 1  # To ensure labels start from 0

        if is_train:
            train_image_ids.append(file_id)
            train_file_paths.append(os.path.join(images_data_dir, file_name))
            train_labels.append(label)
        else:
            test_image_ids.append(file_id)
            test_file_paths.append(os.path.join(images_data_dir, file_name))
            test_labels.append(label)

    print ("Length of train files list", len(train_file_paths))
    print ("Length of train labels list", len(train_labels))
    print ("Length of test files list", len(test_file_paths))
    print ("Length of test labels list", len(test_labels))

    return train_image_ids, test_image_ids, train_file_paths, test_file_paths, train_labels, test_labels

