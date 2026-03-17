import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count() or 1

def create_image_dataloaders(
    *,
    dir: str,  
    train_transform: transforms.Compose,
    test_transform: transforms.Compose | None = None, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
) -> tuple[DataLoader, DataLoader, list[str]]:
  if test_transform is None:
    test_transform = train_transform
  
  train_dataloader, classes = create_image_dataloader(
    dir=dir + "/train",
    transform=train_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True
  )
  test_dataloader, _ = create_image_dataloader(
    dir=dir + "/test",
    transform=test_transform,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False
  )

  return train_dataloader, test_dataloader, classes

def create_image_dataloader(
    *,
    dir: str, 
    transform: transforms.Compose,
    batch_size: int, 
    num_workers: int,
    shuffle: bool
) -> tuple[DataLoader, list[str]]:
  data = datasets.ImageFolder(dir, transform=transform)

  # Turn images into data loaders
  dataloader = DataLoader(
      data,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_workers,
      pin_memory=True,
  )

  return dataloader, data.classes