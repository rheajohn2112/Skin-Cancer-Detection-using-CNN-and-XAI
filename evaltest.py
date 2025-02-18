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

# Load and preprocess image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  
    return image

# Improved segmentation with spatial features
def segment_image(image, num_segments=49):
    height, width = image.shape[:2]
    y_coords, x_coords = np.indices((height, width))
    y_coords = y_coords.astype(np.float32) / height
    x_coords = x_coords.astype(np.float32) / width
    features = np.concatenate([image, y_coords[..., None], x_coords[..., None]], axis=-1)
    reshaped_features = features.reshape(-1, 5)
    
    kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init=10)
    kmeans.fit(reshaped_features)
    return kmeans.labels_.reshape(height, width)

# Normalize heatmap to 0-255
def normalize_heatmap(heatmap):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return (255 * heatmap).astype(np.uint8)

# Corrected DHIS implementation
def generate_dhis_explanation(image, heatmap, num_segments=49, top_percent=25):
    labels = segment_image(image, num_segments)
    unique_segments = np.unique(labels)
    normalized_heatmap = heatmap.astype(np.float32) / 255.0

    segment_scores = []
    for seg in unique_segments:
        mask = labels == seg
        segment_scores.append((seg, np.mean(normalized_heatmap[mask])))

    segment_scores.sort(key=lambda x: x[1], reverse=True)
    num_top = int(len(unique_segments) * top_percent / 100)
    top_segments = [seg for seg, _ in segment_scores[:num_top]]
    
    return np.isin(labels, top_segments).astype(np.float32)

# Enhanced LIME explanation
def generate_lime_explanation(image, model, preprocess):
    explainer = LimeImageExplainer()
    def predict_fn(images):
        processed = torch.stack([preprocess(img) for img in images])
        with torch.no_grad():
            return model(processed).cpu().numpy()
    
    explanation = explainer.explain_instance(
        image, predict_fn, top_labels=1, hide_color=0, num_samples=1000
    )
    _, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True)
    return normalize_heatmap(mask)

# Improved SHAP explanation
def generate_shap_explanation(image, model, preprocess):
    background = torch.zeros((10, 3, 224, 224)).to(device)  # Better background
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    target_class = torch.argmax(output).item()

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(image_tensor)
    
    if isinstance(shap_values, list):
        shap_heatmap = np.abs(shap_values[target_class][0]).sum(axis=0)
    else:
        shap_heatmap = np.abs(shap_values[0]).sum(axis=0)
        
    if len(shap_heatmap.shape) == 3:
        shap_heatmap = shap_heatmap.mean(axis=-1)
    
    return normalize_heatmap(shap_heatmap)

# Grad-CAM implementation
def generate_gradcam_explanation(image, model, target_layer, preprocess):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(image_tensor)
    target_class = torch.argmax(output)
    model.zero_grad()
    output[0, target_class].backward()
    
    handle_forward.remove()
    handle_backward.remove()
    
    activations = activations[0].squeeze()
    gradients = gradients[0].squeeze()
    weights = gradients.mean(dim=(1, 2), keepdim=True)
    cam = (weights * activations).sum(0).clamp(min=0)
    cam = cv2.resize(cam.cpu().numpy(), (224, 224))
    return normalize_heatmap(cam)

# Visualization function
def visualize_results(image, heatmap, method_name):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    overlay = np.zeros((*heatmap.shape, 4))  # RGBA array
    overlay[..., 0] = 1.0  # Red channel
    overlay[..., 1] = 1.0  # Green channel
    overlay[..., 3] = heatmap * 0.5  # Alpha channel (30% opacity where mask is 1)
    
    plt.imshow(overlay)
    #plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.title(f"{method_name} Explanation")
    plt.axis('off')
    plt.show()

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

def evaluate_xai_methods(image_path):
    image = load_image(image_path)
    model = ConvNet().to(device)
    model.load_state_dict(torch.load("C:/Users/rohan/Visual Studios/Skin-Cancer-Detection-using-CNN-and-XAI-main/Skin-Cancer-Detection-using-CNN-and-XAI-main/trained_cnn_model_sgd.pth", map_location=device))
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Generate base explanations
    lime_base = generate_lime_explanation(image, model, preprocess)
    shap_base = generate_shap_explanation(image, model, preprocess)
    gradcam_base = generate_gradcam_explanation(image, model, model.conv7, preprocess)

    # Generate DHIS explanations
    dhis_lime = generate_dhis_explanation(image, lime_base)
    dhis_shap = generate_dhis_explanation(image, shap_base)
    dhis_gradcam = generate_dhis_explanation(image, gradcam_base)

    # Visualize results
    visualize_results(image, dhis_lime, "DHIS-LIME")
    visualize_results(image, dhis_shap, "DHIS-SHAP")
    visualize_results(image, dhis_gradcam, "DHIS-GradCAM")

# Execute evaluation
evaluate_xai_methods("C:/Users/rohan/Downloads/Cancer-3.jpg")