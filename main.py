from torch import nn
from torchvision import transforms
from utils import get_device, create_image_dataloaders
from vit import EmbeddingLayer

def main():
	device = get_device()

	train_transform = transforms.Compose([
		transforms.Resize([224, 224]),
		transforms.ToTensor(),
	])
	
	test_transform = transforms.Compose([
		transforms.Resize([224, 224]),
		transforms.ToTensor(),
	])

	train_dataloader, test_dataloader, class_names = create_image_dataloaders(
		dir="data/pss_10",
		train_transform=train_transform,
		test_transform=test_transform,
		batch_size=32,
		num_workers=4
	) 

	print(f"Class names: {class_names}")

	image_batch, label_batch = next(iter(test_dataloader))
	image, label = image_batch[0], label_batch[0]

	print(f"X: {image.shape}, y: {label}")

	embed = EmbeddingLayer()
	patch_embedded_image = embed(image_batch)

	print(f"{patch_embedded_image.shape}")

if __name__ == "__main__":
		main()
