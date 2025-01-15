from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from flask_cors import CORS

UPLOAD_FOLDER = './uploads'
MODEL_PATH = './trained_cnn_model_sgd.pth'

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Define Image Transformations
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transformer(image).unsqueeze(0)

# Grad-CAM Implementation
def generate_gradcam(model, input_tensor, target_layer, target_class=None):
    model.eval()
    
    gradients = []
    activations = []

    #Hooks are for analyzing the thought process
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_backward = target_layer.register_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)

    output = model(input_tensor)
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()       #Use class with highest probability
    loss = output[0, target_class]

    #Clear gradients
    model.zero_grad()
    loss.backward()

    handle_backward.remove()
    handle_forward.remove()

    gradients = gradients[0].detach()
    activations = activations[0].detach()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    gradcam = (weights * activations).sum(dim=1, keepdim=True)
    gradcam = F.relu(gradcam)       #Ensures highlighting of contributing regions
    gradcam = gradcam.squeeze().cpu().numpy()       #Conversion to array
    #Normalizing values to 0, 1
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()
    gradcam = cv2.resize(gradcam, (input_tensor.shape[2], input_tensor.shape[3]))

    return gradcam, target_class

#@app.route('/gradcam', methods=['POST'])
#def gradcam():  
    if 'file1' not in request.files:
        return 'No file uploaded'
    file = request.files['file1']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    input_tensor = preprocess_image(file_path)
    target_layer = model.conv7  # Replace with the layer you want to use
    gradcam, predicted_class = generate_gradcam(model, input_tensor, target_layer)

    gradcam_img = np.uint8(255 * gradcam)
    heatmap = cv2.applyColorMap(gradcam_img, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img = Image.open(file_path).resize((150, 150))
    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{file.filename}")
    Image.fromarray(overlay).save(result_path)

    return send_file(result_path, mimetype='image/png')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file1' not in request.files:
        return 'No file uploaded'
    
    file = request.files['file1']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Preprocess the image and predict
        input_tensor = preprocess_image(file_path)
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

        # Generate Grad-CAM visualization
        target_layer = model.conv7  # Use the final layer
        gradcam, _ = generate_gradcam(model, input_tensor, target_layer)

        # Save the Grad-CAM image
        gradcam_img = np.uint8(255 * gradcam)   #Expanding ranges of values
        heatmap = cv2.applyColorMap(gradcam_img, cv2.COLORMAP_JET)      #Apply blur, green, yellow, red
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)      #Convert to RGB

        img = Image.open(file_path).resize((150, 150))  
        img_np = np.array(img)      #convert image to 2D array for computation
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{file.filename}")
        Image.fromarray(overlay).save(result_path)
        gradcam_url = result_path.replace("\\", "/")

        return render_template('index.html', 
                               prediction=f'Predicted class: {predicted_class}', 
                               gradcam_image=gradcam_url,
                               file_path=file_path)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"



@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
