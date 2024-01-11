"""
File: testing.py
Name:Alex
----------------------------------------
TODO:
Test the models.
"""
import os
import torch
from matplotlib import pyplot as plt
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import json
import numpy as np
from monai.utils import first
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
	Activations,
)
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

model_dir = 'results/temp_5'
train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
test_loss = np.load(os.path.join(model_dir, 'loss_test.npy'))
test_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))


def main():
	# plot_loss_epoch()

	device = torch.device("cpu")
	model = UNet(
		dimensions=3,
		in_channels=1,
		out_channels=2,
		channels=(16, 32, 64, 128, 256),
		strides=(2, 2, 2, 2),
		num_res_units=2,
		norm=Norm.BATCH,
	).to(device)

	model.load_state_dict(torch.load(
		os.path.join(model_dir, "best_metric_model.pth"), map_location=torch.device('cpu')))
	model.eval()

	test_loader = prepare('data/', 1)

	sw_batch_size = 4
	roi_size = (128, 128, 64)
	with torch.no_grad():
		test_patient = first(test_loader)
		t_volume = test_patient['vol']
		# t_segmentation = test_patient['seg']

		test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
		sigmoid_activation = Activations(sigmoid=True)
		test_outputs = sigmoid_activation(test_outputs)
		test_outputs = test_outputs > 0.53

		for i in range(32):
			# plot the slice [:, :, 80]
			plt.figure("check", (18, 6))
			plt.subplot(1, 3, 1)
			plt.title(f"image {i}")
			plt.imshow(test_patient["vol"][0, 0, :, :, i], cmap="gray")
			plt.subplot(1, 3, 2)
			plt.title(f"label {i}")
			plt.imshow(test_patient["seg"][0, 0, :, :, i] != 0)
			plt.subplot(1, 3, 3)
			plt.title(f"output {i}")
			plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
			plt.show()


def prepare(in_dir, batch_size, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128, 128, 64]):
	"""
	This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you
	find in the Monai documentation.
	https://monai.io/docs.html
	"""
	with open('my_file_val.json', 'r') as f:
		test_data = json.load(f)

	path_test_volumes = []
	path_test_segmentation = []

	for file in test_data['three_label_slices'][14:16]:
		path_test_volumes.append(os.path.join(in_dir, "val_vols", file))
		path_test_segmentation.append(os.path.join(in_dir, "val_labels", file))

	test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
				  zip(path_test_volumes, path_test_segmentation)]

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

	test_ds = Dataset(data=test_files, transform=test_transforms)
	test_loader = DataLoader(test_ds, batch_size=batch_size)

	return test_loader


def plot_loss_epoch():
	plt.figure("Results 25 june", (12, 6))
	plt.subplot(2, 2, 1)
	plt.title("Train dice loss")
	x = [i + 1 for i in range(len(train_loss))]
	y = train_loss
	plt.xlabel("epoch")
	plt.plot(x, y)

	plt.subplot(2, 2, 2)
	plt.title("Train metric DICE")
	x = [i + 1 for i in range(len(train_metric))]
	y = train_metric
	plt.xlabel("epoch")
	plt.plot(x, y)

	plt.subplot(2, 2, 3)
	plt.title("Test dice loss")
	x = [i + 1 for i in range(len(test_loss))]
	y = test_loss
	plt.xlabel("epoch")
	plt.plot(x, y)

	plt.subplot(2, 2, 4)
	plt.title("Test metric DICE")
	x = [i + 1 for i in range(len(test_metric))]
	y = test_metric
	plt.xlabel("epoch")
	plt.plot(x, y)

	plt.show()


if __name__ == '__main__':
	main()
