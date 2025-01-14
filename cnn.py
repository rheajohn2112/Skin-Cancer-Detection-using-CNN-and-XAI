import os
import numpy as np
import glob
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
import pathlib
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import SGD

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMS:
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #transforms.RandomRotation(15),
    #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=10),
    #transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])

# DataLoader
train_path = '/projects/u2103179/testcase/Dataset/Train/'  # Train path
test_path = '/projects/u2103179/testcase/Dataset/Test/'    # Test path

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

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100

# TRAINING THE CNN MODEL WITH ADAMS OPTIMIZER
'''def train_cnn(model, train_loader, num_epochs=50, learning_rate=0.0001, weight_decay=1e-05):
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate train and test accuracy
        train_accuracy = calculate_accuracy(train_loader, model)
        test_accuracy = calculate_accuracy(test_loader, model)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the trained model
    torch.save(model.state_dict(), "trained_cnn_model.pth")
    print("CNN model trained and saved.")

# Train the CNN
train_cnn(model, train_loader, num_epochs=50)'''
# TRAINING THE CNN MODEL WITH SGD MOMENTUM
def train_cnn(model, train_loader, num_epochs=50, learning_rate=0.01, momentum=0.9, weight_decay=1e-05, use_sgd=True):
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate train and test accuracy
        train_accuracy = calculate_accuracy(train_loader, model)
        test_accuracy = calculate_accuracy(test_loader, model)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the trained model
    torch.save(model.state_dict(), "trained_cnn_model_sgd.pth")
    print("CNN model trained and saved using SGD with momentum.")

# Train the CNN with SGD momentum
train_cnn(
    model,
    train_loader,
    num_epochs=50,
    learning_rate=0.01,  # Typical learning rate for SGD
    momentum=0.9,  # Common momentum value
    weight_decay=1e-05,
    use_sgd=True  # Toggle between SGD and Adam
)


# LOAD THE TRAINED CNN MODEL FOR FEATURE EXTRACTION
model.load_state_dict(torch.load("/projects/u2103179/CNN/trained_cnn_model.pth"))
model.eval()

# FEATURE EXTRACTION FUNCTION (Modified to Extract Flattened Features for XGBoost)
def extract_features(loader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label in loader:
            images = images.to(device)
            # Get features before the final fully connected layer
            output = model.global_average_pooling(model.pool7(model.relu7(model.bn7(model.conv7(images)))))
            feature = output.view(output.size(0), -1)  # Flatten features for XGBoost
            features.append(feature.cpu().numpy())
            labels.append(label.numpy())
    return np.vstack(features), np.concatenate(labels)

# Extract features using the trained CNN model
train_features, train_labels = extract_features(train_loader, model)
test_features, test_labels = extract_features(test_loader, model)

# Split the extracted features for training and validation (optional)
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Train XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.01,
    objective='multi:softmax',
    num_class=num_classes,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(test_features)
accuracy = accuracy_score(test_labels, y_pred)
print(f'XGBoost Accuracy: {accuracy:.4f}')
