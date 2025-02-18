import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn

def train_model():
    # Load pre-trained Inception-v3 model
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    model.eval()  # Set model to evaluation mode

    # Define Image Transformations
    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Load Train and Test Datasets
    train_dataset = datasets.ImageFolder(root="C:/Users/rohan/Visual Studios/Skin-Cancer-Detection-using-CNN-and-XAI-main/Skin-Cancer-Detection-using-CNN-and-XAI-main/dataset/Train", transform=transform)
    val_dataset = datasets.ImageFolder(root="C:/Users/rohan/Visual Studios/Skin-Cancer-Detection-using-CNN-and-XAI-main/Skin-Cancer-Detection-using-CNN-and-XAI-main/dataset/Test", transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)  # Set num_workers=0 to avoid multiprocessing issues
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=0)

    # Modify Pretrained Inception-v3 Model
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(2048, num_classes)
    model.AuxLogits.fc = nn.Linear(768, num_classes)

    # Move Model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Parameters
    num_epochs = 10
    best_val_accuracy = 0.0

    # Training Loop
    for epoch in range(num_epochs):
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

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation
        model.eval()
        correct_val, total_val = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_inception_model.pth")
            print(f"🔥 New Best Model Saved! Validation Accuracy: {best_val_accuracy:.2f}%")

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%\n")

    print("Training Complete!!")

if __name__ == '__main__':
    train_model()
