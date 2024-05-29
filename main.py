import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

# Transformationen definieren mit Datenaugmentation und Normalisierung
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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

# Definiere erweitertes neuronales Netzwerk mit Dropout
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(train_loader, val_loader, test_loader, num_classes, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_time = time.time()

    print(f"Training on dataset with {num_classes} classes...")
    best_val_acc = 0
    patience_counter = 0
    max_epochs = 100  # Maximal mögliche Epochen

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f'Validation loss after epoch {epoch + 1}: {val_loss / len(val_loader):.4f}')
        print(f'Validation accuracy after epoch {epoch + 1}: {val_acc:.2f} %')

        scheduler.step()

        # Early stopping based on validation accuracy improvement
        if val_acc - best_val_acc > 3:  # Minimum 3% improvement
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to less than 3% improvement in accuracy!")
                break

    print('Finished Training')

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print(f"Training duration: {duration_minutes:.2f} minutes")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct / total} %')

# Main program
def main():
    choice = input("Which dataset do you want to train on? Type 'cifar10' or 'cifar100': ")
    if choice == 'cifar10':
        cifar10_train, cifar10_test = load_cifar10()
        train_size = int(0.8 * len(cifar10_train))
        val_size = len(cifar10_train) - train_size
        cifar10_train, cifar10_val = random_split(cifar10_train, [train_size, val_size])
        display_cifar(cifar10_train, 5)
        train_model(DataLoader(cifar10_train, batch_size=64, shuffle=True),
                    DataLoader(cifar10_val, batch_size=64, shuffle=False),
                    DataLoader(cifar10_test, batch_size=64, shuffle=False),
                    num_classes=10)
    elif choice == 'cifar100':
        cifar100_train, cifar100_test = load_cifar100()
        train_size = int(0.8 * len(cifar100_train))
        val_size = len(cifar100_train) - train_size
        cifar100_train, cifar100_val = random_split(cifar100_train, [train_size, val_size])
        display_cifar(cifar100_train, 5)
        train_model(DataLoader(cifar100_train, batch_size=64, shuffle=True),
                    DataLoader(cifar100_val, batch_size=64, shuffle=False),
                    DataLoader(cifar100_test, batch_size=64, shuffle=False),
                    num_classes=100)
    else:
        print("Invalid choice. Please type 'cifar10' or 'cifar100'.")

if __name__ == "__main__":
    main()
