# generate_data.py
import torch
from torchvision import datasets, transforms

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    images = torch.stack([dataset[i][0] for i in range(2000)])
    labels = torch.tensor([dataset[i][1] for i in range(2000)])

    torch.save(images, "images.pt")
    torch.save(labels, "labels.pt")

if __name__ == "__main__":
    load_data()
