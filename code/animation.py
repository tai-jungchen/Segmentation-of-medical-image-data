"""
File: animation.py
Name:Alex
----------------------------------------
TODO:
Read in the images by slices and animate the label slices
"""
import json
import nibabel as nib
import cv2
from matplotlib import pyplot as plt
import numpy as np


def main():
    # Open the JSON file in read mode
    with open('data/Task03_Liver/dataset.json', 'r') as file:
        # Load the JSON data
        description = json.load(file)
    print(description)

    img = nib.load('data/Task03_Liver/imagesTr/liver_0.nii.gz')
    label = nib.load('data/Task03_Liver/labelsTr/liver_0.nii.gz')
    # nifti_img = nib.load('data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz')
    img_data = img.get_fdata()
    label_data = label.get_fdata()

    # Plot a slice of the image
    for slice_idx in range(img_data.shape[2]):
        slice_data = img_data[:, :, slice_idx]  # Extract a 2D slice along the third dimension
        plt.imshow(slice_data, cmap='gray')
        plt.axis('off')
        plt.title('Image Slice')
        plt.show()

    images = []
    counter = 0
    # Plot a slice of the image
    for slice_idx in range(label_data.shape[2]):
        print(counter)
        slice_data = label_data[:, :, slice_idx]  # Extract a 2D slice along the third dimension
        plt.imshow(slice_data, cmap='gray')
        plt.axis('off')
        plt.title('Image Slice')
        plt.show()
        images.append(slice_data)
        counter += 1

    # Set up display window
    cv2.namedWindow("Image Animation", cv2.WINDOW_NORMAL)

    # Set initial image
    current_image = 0

    # Animation loop
    while True:
        cv2.imshow("Image Animation", images[current_image])
        if cv2.waitKey(40) & 0xFF == 27:  # adjust the delay value to control animation speed
            break
        current_image = (current_image + 1) % len(images)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
