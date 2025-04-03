import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

def save_confusion_matrix(cm, class_names, filename="best_confusion_matrix_3.png"):
    """Save the best confusion matrix as an image."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def train_model():
    num_classes = 2
    model = models.inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, num_classes)
    model.AuxLogits.fc = nn.Linear(768, num_classes)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.ImageFolder(root="/projects/u2103179cse/Project/dataset 3/Train", transform=transform)
    test_dataset = datasets.ImageFolder(root="/projects/u2103179cse/Project/dataset 3/Test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    best_test_accuracy = 0.0
    best_conf_matrix = None
    checkpoint_path = "inception_checkpoint3.pth"
    best_model_path = "best_inception_model3.pth"

    start_epoch = 0
    train_accuracies, test_accuracies, test_f1_scores = [], [], []

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_test_accuracy = checkpoint['best_test_accuracy']
        train_accuracies = checkpoint['train_accuracies']
        test_accuracies = checkpoint['test_accuracies']
        test_f1_scores = checkpoint.get('test_f1_scores', [])
        print(f" Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if hasattr(model, 'AuxLogits'):
                loss1 = criterion(outputs.logits, labels)
                loss2 = criterion(outputs.aux_logits, labels)
                loss = loss1 + 0.4 * loss2
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.logits if hasattr(model, 'AuxLogits') else outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        correct_test, total_test = 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)

        test_f1 = f1_score(all_labels, all_preds, average='weighted')
        test_f1_scores.append(test_f1)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_conf_matrix = conf_matrix
            torch.save(model.state_dict(), best_model_path)
            print(f" New Best Model Saved! Test Accuracy: {best_test_accuracy:.2f}%, F1 Score: {test_f1:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_test_accuracy': best_test_accuracy,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'test_f1_scores': test_f1_scores
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, F1 Score: {test_f1:.4f}")

    if best_conf_matrix is not None:
        save_confusion_matrix(best_conf_matrix, train_dataset.classes, filename="best_confusion_matrix_inception_3.png")
        print("Best confusion matrix saved as 'best_confusion_matrix.png'.")

    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('accuracy_plot_inception_3.png')
    plt.show()

    plt.plot(range(1, len(test_f1_scores) + 1), test_f1_scores, label='Test F1 Score', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig('f1_score_plot_inception_3.png')
    plt.show()

    print("Training complete. Accuracy, F1 score, and best confusion matrix images saved.")

if __name__ == '__main__':
    train_model()
