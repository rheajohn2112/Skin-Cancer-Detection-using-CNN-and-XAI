import shap
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the ConvNet model architecture (same as during training)
class ConvNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input):
        x = self.pool1(self.relu1(self.bn1(self.conv1(input))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.pool7(self.relu7(self.bn7(self.conv7(x))))
        x = self.dropout(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2

# Load the trained CNN model
model = ConvNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(
    "C:/Users/rohan/Visual Studios/Skin-Cancer-Detection-using-CNN-and-XAI-main/Skin-Cancer-Detection-using-CNN-and-XAI-main/trained_cnn_model_sgd.pth",
    map_location=device))
model.eval()

# Define the transformation for input images
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# SHAP explanation for a single image with fusion and filtering
def explain_image(image_path, model):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transformer(image).unsqueeze(0).to(device)

    # Define a SHAP explainer
    background = torch.randn((10, 3, 150, 150)).to(device)  # Random background data
    explainer = shap.DeepExplainer(model, background)

    # Generate SHAP values
    shap_values = explainer.shap_values(input_tensor)

    # Convert SHAP values to numpy arrays
    shap_values = np.array(shap_values[0])  # Use the SHAP values for the first class
    input_image = input_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0]  # Convert to NHWC format
    #print("SHAP values shape:", shap_values.shape)
    shap_image=shap_values[0]
    # Sum across channels to match image dimensions
    shap_sum =np.sum(shap_image, axis=-1)

    # Filter out low SHAP values below threshold
    threshold = 0.0005
    shap_sum[np.abs(shap_sum) < threshold] = 0

    # Lighten the input image for better visibility
    lightened_image = np.clip(input_image * 0.7 + 0.3, 0, 1)  # Increase brightness

    # Plot fused image
    plt.figure(figsize=(8, 6))

    # Show lightened input image
    plt.imshow(lightened_image)

    # Overlay SHAP values where they meet the threshold
    plt.imshow(shap_sum, cmap='bwr', alpha=0.4)

    plt.colorbar(label="SHAP value")
    plt.title("SHAP Explanation (Filtered and Fused)")
    plt.axis('off')
    plt.show()

# Example usage
example_image_path = "C:/Users/rohan/Downloads/Lesion-1.jpg"
explain_image(example_image_path, model)
