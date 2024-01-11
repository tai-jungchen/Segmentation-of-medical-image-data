"""
File: preprocess.py
Name:Alex
----------------------------------------
TODO:
preprocess the medical image data
"""
import json
import os
from glob import glob
import shutil
from tqdm import tqdm
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism


def main():
    train_volumes = sorted(glob(os.path.join("data/Task03_Liver/imagesTr", "*.nii.gz")))
    train_labels = sorted(glob(os.path.join("data/Task03_Liver/labelsTr", "*.nii.gz")))
    path_train_volumes = train_volumes[:-10]
    path_train_labels = train_labels[:-10]
    path_val_volumes = train_volumes[-10:]
    path_val_labels = train_labels[-10:]

    counter = 0
    for i in tqdm(range(len(path_train_volumes))):
        print(f'processing {path_train_volumes[i]} and {path_train_labels[i]}')
        train_vol = path_train_volumes[i]
        train_label = path_train_labels[i]
        extract_images(train_vol, train_label, 'train')
        counter += 1
    print(f'\nnumber of train volumes / labels: {counter}\n')

    counter = 0
    for i in tqdm(range(len(path_val_volumes))):
        print(f'processing {path_val_volumes[i]} and {path_val_labels[i]}')
        val_vol = path_val_volumes[i]
        val_label = path_val_labels[i]
        extract_images(val_vol, val_label, 'test')
        counter += 1
    print(f'\nnumber of testing volumes / labels: {counter}')


def extract_images(vol_file_name, label_file_name, data_type):
    if data_type == 'train':
        vol_prefix = 'data/train_vols/'
        label_prefix = 'data/train_labels/'
    else:
        vol_prefix = 'data/val_vols/'
        label_prefix = 'data/val_labels/'

    file_name = vol_file_name[27:-7]
    img = nib.load(vol_file_name)
    label = nib.load(label_file_name)
    img_data = img.get_fdata()
    label_data = label.get_fdata()

    for slice_idx in range(img_data.shape[2]):
        processed_file_name = file_name + '_' + str(slice_idx) + '.nii.gz'
        slice_img = img_data[:, :, slice_idx]  # Extract a 2D slice along the third dimension
        slice_label = label_data[:, :, slice_idx]

        slice_img_file = nib.Nifti1Image(slice_img, np.eye(4))
        slice_label_file = nib.Nifti1Image(slice_label, np.eye(4))

        nib.save(slice_img_file, vol_prefix+processed_file_name)
        nib.save(slice_label_file, label_prefix+processed_file_name)
    print()


def create_groups(in_dir, out_dir, Number_slices):
    '''
    This function is to get the last part of the path so that we can use it to name the folder.
    `in_dir`: the path to your folders that contain dicom files
    `out_dir`: the path where you want to put the converted nifti files
    `Number_slices`: here you put the number of slices that you need for your project and it will
    create groups with this number.
    '''

    for patient in glob(in_dir + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))

        # Here we need to calculate the number of folders which mean into how many groups we will divide the number of slices
        number_folders = int(len(glob(patient + '/*')) / Number_slices)

        for i in range(number_folders):
            output_path = os.path.join(out_dir, patient_name + '_' + str(i))
            os.mkdir(output_path)

            # Move the slices into a specific folder so that you will save memory in your desk
            for i, file in enumerate(glob(patient + '/*')):
                if i == Number_slices + 1:
                    break

                shutil.move(file, output_path)


def find_empy(in_dir):
    '''
    This function will help you to find the empty volumes that you may not need for your training
    so instead of opening all the files and search for the empty ones, them use this function to make it quick.
    '''

    list_patients = []
    for patient in glob(os.path.join(in_dir, '*')):
        img = nib.load(patient)

        if len(np.unique(img.get_fdata())) > 2:
            print(os.path.basename(os.path.normpath(patient)))
            list_patients.append(os.path.basename(os.path.normpath(patient)))

    return list_patients


def prepare(in_dir, batch_size, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128, 128, 64], cache=False):
    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you
    find in the Monai documentation.
    https://monai.io/docs.html
    """
    with open('my_file.json', 'r') as f:
        train_data = json.load(f)
    with open('my_file_val.json', 'r') as f:
        test_data = json.load(f)

    set_determinism(seed=0)
    # path_train_volumes = sorted(glob(os.path.join(in_dir, "train_vols", "*.nii.gz")))
    # path_train_segmentation = sorted(glob(os.path.join(in_dir, "train_labels", "*.nii.gz")))
    # path_test_volumes = sorted(glob(os.path.join(in_dir, "val_vols", "*.nii.gz")))
    # path_test_segmentation = sorted(glob(os.path.join(in_dir, "val_labels", "*.nii.gz")))

    path_train_volumes = []
    path_train_segmentation = []
    path_test_volumes = []
    path_test_segmentation = []

    for file in train_data['three_label_slices']:
        path_train_volumes.append(os.path.join(in_dir, "train_vols", file))
        path_train_segmentation.append(os.path.join(in_dir, "train_labels", file))
    for file in test_data['three_label_slices']:
        path_test_volumes.append(os.path.join(in_dir, "val_vols", file))
        path_test_segmentation.append(os.path.join(in_dir, "val_labels", file))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(path_test_volumes, path_test_segmentation)]

    train_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=batch_size)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=batch_size)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        return train_loader, test_loader


if __name__ == '__main__':
    # main()
    data_in = prepare('data/', 32)
