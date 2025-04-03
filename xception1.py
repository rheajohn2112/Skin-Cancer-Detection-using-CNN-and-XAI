import torch
import timm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(cm, class_names, filename="best_confusion_matrix_xception_3.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Best Model Confusion Matrix")
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved as {filename}", flush=True)

def save_checkpoint(model, optimizer, epoch, best_val_accuracy, train_accuracies, val_accuracies, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with validation accuracy {best_val_accuracy:.2f}%", flush=True)

def plot_accuracy_graph(epochs, train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o', linestyle='-')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='s', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("train_vs_test_accuracy_xception_3.png")
    plt.show()
    print("Train vs Test Accuracy graph saved as train_vs_test_accuracy_xception_3.png", flush=True)

def train_model():
    model = timm.create_model('xception', pretrained=True)
    model.last_linear = nn.Linear(2048, 2)  # replace the classification head

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.ImageFolder(root="/projects/u2103179cse/Project/dataset 3/Train", transform=transform)
    val_dataset = datasets.ImageFolder(root="/projects/u2103179cse/Project/dataset 3/Test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    best_val_accuracy = 0.0
    checkpoint_path = "xception_checkpoint3.pth"
    best_model_path = "best_xception_model3.pth"

    train_accuracies = []
    val_accuracies = []
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint['best_val_accuracy']
        train_accuracies = checkpoint.get('train_accuracies', [])[:start_epoch]
        val_accuracies = checkpoint.get('val_accuracies', [])[:start_epoch]
        print(f"Resuming training from epoch {start_epoch}...", flush=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        correct_train, total_train = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        correct_val, total_val = 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        save_checkpoint(model, optimizer, epoch, val_accuracy, train_accuracies, val_accuracies, checkpoint_path)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best Model Saved at epoch {epoch}! Validation Accuracy: {best_val_accuracy:.2f}%", flush=True)
            conf_matrix = confusion_matrix(all_labels, all_preds)
            save_confusion_matrix(conf_matrix, class_names=train_dataset.classes, filename="best_confusion_matrix_xception_3.png")

        print(f"Epoch {epoch} complete: Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%", flush=True)

    print("Training complete.", flush=True)
    epochs = list(range(start_epoch, start_epoch + len(train_accuracies)))
    plot_accuracy_graph(epochs, train_accuracies, val_accuracies)

if __name__ == '__main__':
    train_model()
