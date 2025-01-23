import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
import torch
import torch.nn as nn

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

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
model = ConvNet(num_classes=num_classes).to(device)

# Load the trained model
model.load_state_dict(torch.load("C:/Users/rohan/Visual Studios/Skin-Cancer-Detection-using-CNN-and-XAI-main/Skin-Cancer-Detection-using-CNN-and-XAI-main/trained_cnn_model_sgd.pth", map_location=device))
model.eval()

print("Model loaded successfully!")

# Define image transformation (same as used during training)
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load a sample image for explanation
img_path = "C:/Users/rohan/Downloads/Lesion-1.jpg"
img = Image.open(img_path).convert('RGB')
input_tensor = transformer(img).unsqueeze(0).to(device)

# Define a prediction function for LIME
def predict_fn(images):
    batch = torch.stack([transformer(Image.fromarray(image)).to(device) for image in images])
    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy()

# Initialize the LIME explainer
explainer = lime_image.LimeImageExplainer()

# Perform explanation on the sample image
explanation = explainer.explain_instance(
    np.array(img), 
    predict_fn, 
    top_labels=1, 
    hide_color=0, 
    num_samples=1000  # Number of perturbations
)

# Visualize the segmentation (before perturbation)
segments = explanation.segments
plt.figure(figsize=(10, 5))

# Original image with segmentation boundaries
plt.subplot(1, 2, 1)
plt.imshow(mark_boundaries(np.array(img), segments))
plt.title("Segmented Image (Before Perturbation)")

# Visualize perturbed image with LIME explanation
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME Perturbation (After Perturbation)")

plt.show()
