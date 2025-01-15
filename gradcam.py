import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def generate_gradcam(model, input_tensor, target_layer, target_class=None):
    model.eval()
    
    # Hook for gradients
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Attach hooks to the target layer
    handle_backward = target_layer.register_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    output = model(input_tensor)
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()
    loss = output[0, target_class]

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Remove hooks
    handle_backward.remove()
    handle_forward.remove()

    # Get gradients and activations
    gradients = gradients[0].detach()
    activations = activations[0].detach()

    # Global average pooling of gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    # Weighted sum of activations
    gradcam = (weights * activations).sum(dim=1, keepdim=True)
    gradcam = F.relu(gradcam)  # Apply ReLU
    gradcam = gradcam.squeeze().cpu().numpy()

    # Normalize the Grad-CAM
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()
    gradcam = cv2.resize(gradcam, (input_tensor.shape[2], input_tensor.shape[3]))

    return gradcam, target_class
