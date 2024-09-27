import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import time
import os
import wandb
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler



wandb.login()

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

def mc_dropout_predict(model, inputs, n_samples=10):
    model.train()  # Dropout während der Inferenz aktivieren
    outputs = [model(inputs) for _ in range(n_samples)]  # Mehrere Vorhersagen generieren
    outputs = torch.stack(outputs)  # (n_samples, batch_size, num_classes)
    mean_outputs = outputs.mean(dim=0)  # Mittelwert der Vorhersagen
    var_outputs = outputs.var(dim=0)  # Varianz der Vorhersagen
    return mean_outputs, var_outputs

def predictive_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

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

# Funktion zur Visualisierung der Entropie und Varianz
def plot_uncertainty(entropies, variances):
    plt.figure(figsize=(12, 6))

    # Entropie plotten
    plt.subplot(1, 2, 1)
    plt.hist(entropies, bins=50, color='blue', alpha=0.7)
    plt.title('Entropy Distribution')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.savefig("entropy_distribution.png")  # Speichere das Bild
    wandb.log({"entropy_distribution": wandb.Image("entropy_distribution.png")})

    # Varianz plotten
    plt.subplot(1, 2, 2)
    plt.hist(variances, bins=50, color='orange', alpha=0.7)
    plt.title('Variance Distribution')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def evaluate_model(loader, model, criterion, log=True):
    model.eval()  # Evaluierungsmodus
    running_loss = 0.0
    correct = 0
    total = 0
    images = []
    predictions = []
    labels_list = []
    entropies = []
    variances = []

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            mean_outputs, var_outputs = mc_dropout_predict(model, inputs, n_samples=10)
            probs = F.softmax(mean_outputs, dim=1)
            entropy = predictive_entropy(probs)

            # Speichern der Entropie und Varianz
            entropies.extend(entropy.cpu().numpy())
            variances.extend(var_outputs.mean(dim=1).cpu().numpy())

            loss = criterion(mean_outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(mean_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Speichern von Bildern und Vorhersagen
            images.extend(inputs.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

            # Loggen der Entropie und der Varianz in wandb
            if log:
                wandb.log({
                    "entropy": entropy.mean().item(),
                    "variance": var_outputs.mean().item()
                })

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)

    # Visualisierung der Entropie und Varianz
    if log:
        plot_uncertainty(entropies, variances)

    return avg_loss, accuracy, images, predictions, labels_list

def compute_class_weights(dataset, num_classes):
    class_counts = np.zeros(num_classes)  # Erstelle ein Array für die Anzahl der Klassen
    for _, label in dataset:
        class_counts[label] += 1

    total_count = len(dataset)
    class_weights = total_count / (num_classes * class_counts)
    return class_weights


def create_weighted_sampler(dataset, num_classes):
    class_counts = np.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1

    total_count = len(dataset)
    class_weights = total_count / (num_classes * class_counts)
    sample_weights = np.array([class_weights[label] for _, label in dataset])

    # Erstelle einen WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def train_model(train_loader, val_loader, test_loader, num_classes, patience=3):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedCNN(num_classes).to(device)
    # Berechne die Klassengewichte
    class_weights = compute_class_weights(train_loader.dataset, num_classes)
    weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    wandb.init(project="cifar-training", config={
        "learning_rate": 0.001,
        "epochs": 500,
        "batch_size": 64,
        "dataset": "CIFAR-10" if num_classes == 10 else "CIFAR-100",
        "model": "AdvancedCNN"
    })

    start_time = time.time()

    print(f"Training on dataset with {num_classes} classes...")

    best_val_loss = float('inf')  # Setze den besten Validierungsverlust initial auf unendlich
    patience_counter = 0  # Zähler für die Geduld

    epoch = 0
    while True:  # Endlosschleife für das Training
        epoch += 1
        print(f"Epoch {epoch}")
        running_loss = 0.0
        entropies = []
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

            # Berechnung der Entropie
            probs = F.softmax(outputs, dim=1)
            entropy = predictive_entropy(probs)  # Berechne die Entropie für die aktuelle Batch
            entropies.extend(entropy.detach().cpu().numpy())  # Speichere die Entropiewerte

            if i % 100 == 99:
                print(f'[{epoch}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

                # Validierung des Modells
                val_loss, val_acc, val_images, val_predictions, val_labels = evaluate_model(val_loader, model,
                                                                                            criterion)

                print(f'Validation loss after epoch {epoch}: {val_loss:.4f}')
                print(f'Validation accuracy after epoch {epoch}: {val_acc:.2f} %')

                # Logge Metriken und Bilder in WandB
                wandb.log({
                    "train_loss": running_loss / len(train_loader),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "examples": [
                        wandb.Image(
                            np.transpose(val_images[j], (1, 2, 0)),
                            caption=f"Predicted: {val_predictions[j]}, Actual: {val_labels[j]}"
                        )
                        for j in range(min(5, len(val_images)))
                    ]
                })

            # Durchschnittliche Entropie berechnen und loggen
        avg_entropy = np.mean(entropies)  # Durchschnittliche Entropie für die Epoche
        wandb.log({"average_entropy": avg_entropy, "epoch": epoch})  # Logge die Entropie

        min_delta = 0.001  # Mindestschwelle für den Rückgang des Validierungsverlustes

        # Early Stopping Logik
        if val_loss < best_val_loss - min_delta:  # Überprüfe, ob der Verlust signifikant gesunken ist
                print("Validation loss improved, resetting patience counter.")
                best_val_loss = val_loss
                patience_counter = 0
        else:
                patience_counter += 1
                print(f"No significant improvement. Patience counter: {patience_counter}/{patience}")

        # Abbruch, wenn Geduld erschöpft ist
        if patience_counter >= patience:
            print("Early stopping triggered due to lack of improvement!")
            break

        scheduler.step()

    print('Finished Training')

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print(f"Training duration: {duration_minutes:.2f} minutes")

    # Test the model
    test_loss, test_acc, test_images, test_predictions, test_labels = evaluate_model(test_loader, model, criterion, log=False)
    print(f'Accuracy on test images: {test_acc} %')

    # Log test results
    wandb.log({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "examples": [
            wandb.Image(
                np.transpose(test_images[j], (1, 2, 0)),  # Umwandlung von (C, H, W) nach (H, W, C)
                caption=f"Predicted: {test_predictions[j]}, Actual: {test_labels[j]}"
            )
            for j in range(min(5, len(test_images)))  # Die ersten 5 Bilder loggen
        ]
    })

# Main program
def main():
    choice = input("Which dataset do you want to train on? Type 'cifar10' or 'cifar100': ")
    if choice == 'cifar10':
        cifar10_train, cifar10_test = load_cifar10()
        train_size = int(0.8 * len(cifar10_train))
        val_size = len(cifar10_train) - train_size
        cifar10_train, cifar10_val = random_split(cifar10_train, [train_size, val_size])
        display_cifar(cifar10_train, 5)

        # Erstelle den WeightedRandomSampler
        sampler = create_weighted_sampler(cifar10_train, num_classes=10)

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

        # Erstelle den WeightedRandomSampler
        sampler = create_weighted_sampler(cifar100_train, num_classes=100)

        train_model(DataLoader(cifar100_train, batch_size=64, shuffle=True),
                    DataLoader(cifar100_val, batch_size=64, shuffle=False),
                    DataLoader(cifar100_test, batch_size=64, shuffle=False),
                    num_classes=100)
    else:
        print("Invalid choice. Please type 'cifar10' or 'cifar100'.")

if __name__ == "__main__":
    main()
