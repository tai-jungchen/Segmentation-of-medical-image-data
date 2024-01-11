"""
File: em.py
Name:Alex
----------------------------------------
TODO:
Implement EM algorithm.
"""
import json
import math

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import nibabel as nib
from sklearn.mixture import GaussianMixture
from utils import dice_coefficient

train_volume_dir = 'data/train_vols/'
train_label_dir = 'data/train_labels/'
test_volume_dir = 'data/val_vols/'
test_label_dir = 'data/val_labels/'

# Define the number of clusters and the EM algorithm parameters
n_components = 3
max_iter = 100
tol = 1e-5


def main():
    # with open('my_file.json', 'r') as f:
    #     train_data = json.load(f)
    #
    # train_vols = train_data['three_label_slices']
    # train_labels = train_data['three_label_slices']

    with open('my_file_val.json', 'r') as f:
        test_data = json.load(f)
    test_vols = test_data['three_label_slices']
    test_labels = test_data['three_label_slices']

    dices = []
    converge = []
    for i in tqdm(range(len(test_vols))):
        volume = nib.load(test_volume_dir + test_vols[i]).get_fdata()
        label = nib.load(test_label_dir + test_labels[i]).get_fdata()
        data = volume.reshape(-1, 1)       # Reshape volume data to a 2D array

        # Initialize the model with the number of clusters and fit the data
        model = GaussianMixture(n_components=n_components, max_iter=max_iter, tol=tol)
        model.fit(data)
        converge.append(model.converged_)

        # Obtain the probability of each voxel belonging to each cluster
        probs = model.predict_proba(data)

        # Generate a segmentation mask by assigning each voxel to the cluster with the highest probability
        seg = np.argmax(probs, axis=1).reshape(volume.shape)

        # Save the segmentation mask
        # nib.save(nib.Nifti1Image(seg, volume.affine), 'em_seg.nii.gz')

        dice = dice_coefficient(label, seg)
        print(f'dice: {dice}')
        dices.append(dice)

        if dice > 0.55:
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(volume, cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(label)
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(seg)
            plt.show()

    print(f'mean dice: {np.mean(dices)}')
    print(f'se: {np.std(dices) / math.sqrt(len(dices))}')
    print(f'max dice: {max(dices)}')
    print(f'min dice: {min(dices)}')
    print(converge.count(True))


if __name__ == '__main__':
    main()
