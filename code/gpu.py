"""
File: gpu.py
Name:Alex
----------------------------------------
TODO:
Test gpu
"""
import torch


def main():
	if torch.cuda.is_available():
		device = torch.device('cuda')
		print('Using GPU:', torch.cuda.get_device_name())
	else:
		device = torch.device('cpu')
		print('Using CPU')


if __name__ == '__main__':
	main()
