"""
File: label_categorization.py
Name:Alex
----------------------------------------
TODO:
Categorize the files into one-label, two-label, and three-label.
"""
import json
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import nibabel as nib


def main():
    with open('my_file.json', 'r') as file:
        # Load the JSON data
        description = json.load(file)
    print(description)

    in_dir = 'data/'
    path_train_volumes = sorted(glob(os.path.join(in_dir, "train_vols", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "train_labels", "*.nii.gz")))
    path_test_volumes = sorted(glob(os.path.join(in_dir, "val_vols", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "val_labels", "*.nii.gz")))

    one_label_slice = 0
    two_label_slice = 0
    three_label_slice = 0
    one_label_slices = []
    two_label_slices = []
    three_label_slices = []

    for i in tqdm(range(len(path_train_volumes))):
        file_name = path_train_volumes[i]
        label_file_name = path_train_segmentation[i]
        # Load liver volume data
        volume = nib.load(file_name).get_fdata()
        label = nib.load(label_file_name).get_fdata()
        num_of_labels = len(np.unique(label))
        if num_of_labels == 1:
            one_label_slice += 1
            one_label_slices.append(file_name[16:])
        elif num_of_labels == 2:
            two_label_slice += 1
            two_label_slices.append(file_name[16:])
        elif num_of_labels == 3:
            three_label_slice += 1
            three_label_slices.append(file_name[16:])
        else:
            print('error')

    print(one_label_slice)
    print(two_label_slice)
    print(three_label_slice)

    num_label_files = {'one_label_slices': one_label_slices,
                   'two_label_slices': two_label_slices,
                   'three_label_slices': three_label_slices}

    # open the file in write mode
    with open("my_file.json", "w") as f:
        # write the dictionary to the file
        json.dump(num_label_files, f)

    print()


if __name__ == '__main__':
    main()
