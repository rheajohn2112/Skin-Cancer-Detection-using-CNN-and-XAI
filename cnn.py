import os
import torch
import torch.nn as nn
import torchvision
import pathlib
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from sklearn.metrics import precision_score, recall_score

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMS:
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# DataLoader
train_path = '/projects/u2103179cse/Project/dataset/Train'  # Adjust paths accordingly
test_path = '/projects/u2103179cse/Project/dataset/Train'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=256, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=256, shuffle=True
)

# CATEGORIES:
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
num_classes = len(classes)

# NETWORK CLASS MODEL:
class ConvNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=512)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=512)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.pool2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.pool3(output)

        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)
        output = self.pool4(output)

        output = self.conv5(output)
        output = self.bn5(output)
        output = self.relu5(output)
        output = self.pool5(output)
        
        output = self.conv6(output)
        output = self.bn6(output)
        output = self.relu6(output)
        output = self.pool6(output)

        output = self.conv7(output)
        output = self.bn7(output)
        output = self.relu7(output)
        output = self.pool7(output)
        
        output = self.dropout(output)
        output = self.global_average_pooling(output)
        output = output.view(output.size(0), -1)  # Flatten for the fully connected layer
        output = self.fc(output)
        return output

# Initialize the model
model = ConvNet(num_classes=num_classes).to(device)

# Checkpoint Paths
checkpoint_path = "model_checkpoint.pth"
best_model_path = "best_model.pth"

# Load Checkpoint if Exists
start_epoch = 0
best_accuracy = 0.0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    print(f"Resuming training from epoch {start_epoch+1} with best accuracy {best_accuracy:.2f}%")

# Function to Calculate Accuracy, Precision, and Recall
def calculate_metrics(loader, model):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total * 100
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    
    return accuracy, precision, recall

# TRAINING FUNCTION WITH CHECKPOINTING & GRAPH PLOTTING
def train_cnn(model, train_loader, num_epochs=50, learning_rate=0.01, momentum=0.9, weight_decay=1e-05):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    global best_accuracy
    
    train_acc_list = []
    test_acc_list = []
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_accuracy, train_precision, train_recall = calculate_metrics(train_loader, model)
        test_accuracy, test_precision, test_recall = calculate_metrics(test_loader, model)
        
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
        print(f"Train Acc: {train_accuracy:.2f}%, Precision: {train_precision:.2f}, Recall: {train_recall:.2f}")
        print(f"Test Acc: {test_accuracy:.2f}%, Precision: {test_precision:.2f}, Recall: {test_recall:.2f}")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_accuracy': best_accuracy
        }, checkpoint_path)
        
        # Save Best Model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
    
    # Plot Accuracy Graph
    epochs_range = range(start_epoch+1, num_epochs+1)
    plt.figure()
    plt.plot(epochs_range, train_acc_list, label='Train Accuracy')
    plt.plot(epochs_range, test_acc_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()
    print("Training complete. Accuracy plot saved.")

train_cnn(model, train_loader, num_epochs=50, learning_rate=0.01, momentum=0.9, weight_decay=1e-05)