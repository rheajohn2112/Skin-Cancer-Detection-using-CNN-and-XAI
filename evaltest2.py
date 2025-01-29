import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lime.lime_image import LimeImageExplainer
from torchvision import transforms

# Load an image and preprocess it
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  
    return image

# Apply K-Means clustering for segmentation
def segment_image(image, num_segments=49):
    reshaped_image = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init=10)
    kmeans.fit(reshaped_image)
    labels = kmeans.labels_.reshape(image.shape[:2])
    return labels

# Compute IOU between two bounding boxes
def compute_iou(boxA, boxB):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

# Normalize heatmap to 0-255
def normalize_heatmap(heatmap):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return np.uint8(255 * heatmap)

# Extract top 25% most important pixels as bounding box
def get_bounding_box_from_heatmap(heatmap, threshold=0.75):
    heatmap = normalize_heatmap(heatmap)
    mask = heatmap > np.percentile(heatmap, threshold * 100)
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return (0,0,10,10)
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    return (x_min, y_min, x_max, y_max)

# Generate LIME explanation
def generate_lime_explanation(image, model, preprocess):
    explainer = LimeImageExplainer()
    def predict_fn(images):
        images = torch.stack([preprocess(img) for img in images])
        return model(images).detach().cpu().numpy()
    
    explanation = explainer.explain_instance(image, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    return normalize_heatmap(explanation.get_image_and_mask(explanation.top_labels[0])[1])

# Generate SHAP explanation
def generate_shap_explanation(image, model, preprocess):
    background = torch.randn((10, 3, 224, 224))
    image_tensor = preprocess(image).unsqueeze(0)
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(image_tensor)
    shap_heatmap = np.abs(shap_values[0]).sum(axis=0)

    if len(shap_heatmap.shape) == 3:
        shap_heatmap = shap_heatmap.mean(axis=-1)
    
    return normalize_heatmap(shap_heatmap)

# Generate Grad-CAM explanation
def generate_gradcam_explanation(image, model, target_layer, preprocess):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    activations, gradients = {}, {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(image_tensor)
    target_class = torch.argmax(output).item()
    output[:, target_class].backward()
    
    handle1.remove()
    handle2.remove()
    
    grad = gradients['value'].mean(dim=[2, 3], keepdim=True)
    cam = F.relu(grad * activations['value']).sum(dim=1).squeeze().detach().cpu().numpy()
    return normalize_heatmap(cv2.resize(cam, (224, 224)))

# Visualize results
def visualize_results(image, heatmap, method_name, ground_truth_box):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    x_min, y_min, x_max, y_max = ground_truth_box
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='red', linewidth=2, fill=False))
    plt.title(f"{method_name} Explanation")
    plt.show()

# CNN Model
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

# Evaluation Function
def evaluate_xai_methods(image_path, ground_truth_box):
    image = load_image(image_path)
    model = ConvNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("C:/Users/rohan/Visual Studios/Skin-Cancer-Detection-using-CNN-and-XAI-main/Skin-Cancer-Detection-using-CNN-and-XAI-main/trained_cnn_model_sgd.pth", map_location=device))
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Generate explanations
    lime_heatmap = generate_lime_explanation(image, model, preprocess)
    shap_heatmap = generate_shap_explanation(image, model, preprocess)
    gradcam_heatmap = generate_gradcam_explanation(image, model, model.conv7, preprocess)

    # Get bounding boxes
    lime_box = get_bounding_box_from_heatmap(lime_heatmap)
    shap_box = get_bounding_box_from_heatmap(shap_heatmap)
    gradcam_box = get_bounding_box_from_heatmap(gradcam_heatmap)

    # Compute IOU
    print(f"LIME IOU: {compute_iou(lime_box, ground_truth_box):.3f}")
    print(f"SHAP IOU: {compute_iou(shap_box, ground_truth_box):.3f}")
    print(f"Grad-CAM IOU: {compute_iou(gradcam_box, ground_truth_box):.3f}")

    # Visualize
    visualize_results(image, lime_heatmap, "LIME", ground_truth_box)
    visualize_results(image, shap_heatmap, "SHAP", ground_truth_box)
    visualize_results(image, gradcam_heatmap, "Grad-CAM", ground_truth_box)

evaluate_xai_methods("C:/Users/rohan/Downloads/Cancer-3.jpg", ground_truth_box=(50, 50, 175, 175))
