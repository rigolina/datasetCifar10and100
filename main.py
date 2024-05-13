import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# Funktionen zur Anzeige von Bildern aus den Datensätzen
def display_cifar(dataset, num_images):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    for i in range(num_images):
        image, _ = dataset[i]
        image = image.permute(1, 2, 0)  # Ändere die Reihenfolge der Dimensionen (C, H, W) zu (H, W, C)
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.show()

def display_cifar_side_by_side(dataset1, dataset2, num_images):
    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))

    for i in range(num_images):
        # Bild aus CIFAR-10 anzeigen
        image1, _ = dataset1[i]
        image1 = image1.permute(1, 2, 0)  # Ändere die Reihenfolge der Dimensionen (C, H, W) zu (H, W, C)
        axes[0, i].imshow(image1)
        axes[0, i].axis('off')

        # Bild aus CIFAR-100 anzeigen
        image2, _ = dataset2[i]
        image2 = image2.permute(1, 2, 0)  # Ändere die Reihenfolge der Dimensionen (C, H, W) zu (H, W, C)
        axes[1, i].imshow(image2)
        axes[1, i].axis('off')

    plt.show()

# Transformationen definieren
transform = transforms.Compose([
    transforms.ToTensor()
])

# CIFAR-10-Datensatz laden
def load_cifar10():
    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("CIFAR-10 train dataset size:", len(cifar10_train))
    print("CIFAR-10 test dataset size:", len(cifar10_test))
    return cifar10_train, cifar10_test

# CIFAR-100-Datensatz laden
def load_cifar100():
    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    print("CIFAR-100 train dataset size:", len(cifar100_train))
    print("CIFAR-100 test dataset size:", len(cifar100_test))
    return cifar100_train, cifar100_test

# Definiere dein neuronales Netzwerk
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(train_loader, test_loader, num_epochs, num_classes):
    model = SimpleCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    print(f"Training on dataset with {num_classes} classes...")
    for epoch in range(num_epochs):
        print("Epoch %d/%d" % (epoch + 1, num_epochs))
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print("Training duration:", duration_minutes, "minutes")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images: %d %%' % (
        100 * correct / total))

# Main program
def main():
    choice = input("Which dataset do you want to train on? Type 'cifar10' or 'cifar100': ")
    if choice == 'cifar10':
        cifar10_train, cifar10_test = load_cifar10()
        display_cifar(cifar10_train, 5)
        train_model(DataLoader(cifar10_train, batch_size=64, shuffle=True),
                    DataLoader(cifar10_test, batch_size=64, shuffle=False),
                    num_epochs=5, num_classes=10)
    elif choice == 'cifar100':
        cifar100_train, cifar100_test = load_cifar100()
        display_cifar(cifar100_train, 5)
        train_model(DataLoader(cifar100_train, batch_size=64, shuffle=True),
                    DataLoader(cifar100_test, batch_size=64, shuffle=False),
                    num_epochs=5, num_classes=100)
    else:
        print("Invalid choice. Please type 'cifar10' or 'cifar100'.")

if __name__ == "__main__":
    main()
