import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.list_files = os.listdir(self.root_dir)

	def __len__(self):
		return len(self.list_files)

	def __getitem__(self, index):
		img_file = self.list_files[index]
		img_path = os.path.join(self.root_dir, img_file)
		image = np.array(Image.open(img_path)) # open image for the particula index
		input_image = image[:,:600,:] # divide the image into x and y
		target_image = image[:,600:,:] # y

		# apply augmenatations to both input and target image
		augmentations = config.both_transform(image=input_image, image0=target_image)
		input_image = augmentations["image"]
		target_image = augmentations["image0"]

		# augmrntations only to the input image (horizontal flip, color jitter, normalize)
		input_image = config.transform_only_input(image=input_image)["image"]
		# augmrntations only to the target image (only normalization)
		target_image = config.transform_only_mask(image=target_image)["image"]

		return input_image, target_image


if __name__ == "__main__":
	dataset = MapDataset("/ssd_scratch/cvit/debtanu.gupta/pix2pix/maps/maps/train")
	loader = DataLoader(dataset, batch_size=5)
	for x, y in loader:
		print(x.shape)
		save_image(x, "x.png")
		save_image(y, "y.png")

		import sys
		sys.exit()

