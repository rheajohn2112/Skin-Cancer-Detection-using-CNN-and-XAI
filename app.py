from flask import Flask, request, redirect, flash, send_from_directory, render_template
import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from lime.lime_image import LimeImageExplainer
from sklearn.cluster import KMeans
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UPLOAD_FOLDER = './uploads'
MODEL_PATH = './best_model.pth'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key="Hello World"

# Define the CNN Model Architecture
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

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
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

# Load the Model
num_classes = 2  # No. of classes in the dataset
model = ConvNet(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# Define Image Transformations
transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def get_bounding_box_from_heatmap(heatmap, threshold=0.75):
    heatmap = normalize_heatmap(heatmap)
    mask = heatmap > np.percentile(heatmap, threshold * 100)
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return (0,0,10,10)
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    return (x_min, y_min, x_max, y_max)

def compute_iou(boxA, boxB=(50, 50, 175, 175)):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

def normalize_heatmap(heatmap):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return np.uint8(255 * heatmap)

def lime(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  
    
    explainer = LimeImageExplainer()
    def predict_fn(images):
        images = torch.stack([transformer(img) for img in images])
        return model(images).detach().cpu().numpy()
    
    explanation = explainer.explain_instance(image, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    return normalize_heatmap(explanation.get_image_and_mask(explanation.top_labels[0])[1])

def shap_gen(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)) 
     
    background = torch.randn((10, 3, 224, 224))
    image_tensor = transformer(image).unsqueeze(0)
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(image_tensor)
    
    shap_values = np.array(shap_values[0])  # Use the SHAP values for the first class
    input_image = image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0]  # Convert to NHWC format
    shap_image = shap_values[0]
    
    # Sum across channels to match image dimensions
    shap_sum = np.sum(shap_image, axis=-1)
    
    # Normalize SHAP values to enhance contrast
    shap_sum = (shap_sum - np.min(shap_sum)) / (np.max(shap_sum) - np.min(shap_sum))  # Normalize to [0, 1]
    shap_sum = shap_sum * 2 - 1  # Scale to [-1, 1] for a balanced red-blue colormap
    shap_sum[np.abs(shap_sum) < 0.05] = 0  # Filter out low SHAP values below threshold

    # Lighten the input image for better visibility
    lightened_image = np.clip(input_image * 0.7 + 0.3, 0, 1)

    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"shap_explanation_{name}.png"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    # Plot and save the SHAP explanation
    plt.figure(figsize=(8, 6))
    plt.imshow(lightened_image)
    plt.imshow(shap_sum, cmap='bwr', alpha=0.6)  # Increased alpha for better visibility
    plt.colorbar(label="Normalized SHAP value")
    plt.title("SHAP Explanation (Enhanced Contrast)")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path

def gradcam(file_path, target_layer):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  
    
    image_tensor = transformer(image).unsqueeze(0)
    activations, gradients = {}, {}

    #Hooks for analyzing the thought process
    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(image_tensor)
    target_class = torch.argmax(output, dim=1).item()
    output[:, target_class].backward()
    
    handle1.remove()
    handle2.remove()
    
    grad = gradients['value'].mean(dim=[2, 3], keepdim=True)
    cam = F.relu(grad * activations['value']).sum(dim=1).squeeze().detach().cpu().numpy()
    return target_class, normalize_heatmap(cv2.resize(cam, (224, 224)))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file1' not in request.files or request.files['file1'].filename == '':
        flash("Please upload a file!", "warning")
        return redirect('/')

    file = request.files['file1']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file_path = file_path.replace("\\", "/")
    file.save(file_path)
    print(file_path)

    try:
        target_layer = model.conv7  # Use the final layer
        predicted_class, gradcam_heatmap = gradcam(file_path, target_layer)     
        lime_heatmap = lime(file_path)
        shap_path = shap_gen(file_path)
        
        lime_box = get_bounding_box_from_heatmap(lime_heatmap)
        #shap_box = get_bounding_box_from_heatmap(shap_heatmap)
        gradcam_box = get_bounding_box_from_heatmap(gradcam_heatmap)
        l_iou = compute_iou(lime_box)
        g_iou = compute_iou(gradcam_box)
        print("LIME Box:", lime_box)
        print("Grad-CAM Box:", gradcam_box)
        print(f"{g_iou:.3f}") 
        print(f"{l_iou:.3f}") 
        
        # Open and resize the original image
        img = Image.open(file_path).resize((224, 224))  
        img_np = np.array(img)      

        # Ensure heatmap has 3 channels and matches image size
        gradcam_heatmap = cv2.applyColorMap(gradcam_heatmap, cv2.COLORMAP_JET)
        gradcam_heatmap = cv2.cvtColor(gradcam_heatmap, cv2.COLOR_BGR2RGB) 
        gradcam_heatmap = cv2.resize(gradcam_heatmap, img_np.shape[:2][::-1])  

        lime_heatmap = cv2.applyColorMap(lime_heatmap, cv2.COLORMAP_JET)
        lime_heatmap = cv2.cvtColor(lime_heatmap, cv2.COLOR_BGR2RGB) 
        lime_heatmap = cv2.resize(lime_heatmap, img_np.shape[:2][::-1]) 
         
        # Create overlay
        overlay = cv2.addWeighted(img_np, 0.5, gradcam_heatmap, 0.5, 0)
        overlay2 = cv2.addWeighted(img_np, 0.5, lime_heatmap, 0.5, 0)

        # Save and render the Grad-CAM result
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{file.filename}")
        lime_path = os.path.join(app.config['UPLOAD_FOLDER'], f"lime_{file.filename}")

        Image.fromarray(overlay).save(gradcam_path)
        Image.fromarray(overlay2).save(lime_path)
        
        return render_template('index.html', 
                               prediction=f'Predicted class: {predicted_class}', 
                               gradcam_image=gradcam_path,
                               lime_image=lime_path,
                               shap_image=shap_path,
                               g_iou=g_iou,
                               l_iou=l_iou,
                               file_path=file_path)

    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
