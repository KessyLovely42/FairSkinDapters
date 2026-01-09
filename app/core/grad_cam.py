import numpy as np
import torch
import cv2
from PIL import Image
import torch.nn.functional as F

class ViTGradCAM:
        """
        Grad-CAM implementation specifically for Vision Transformers (ViT) and DINOv2.
        """
        
        def __init__(self, model, use_cuda=True):
            """
            Args:
                model: The ViT/DINOv2 model
                target_layer: The layer to compute gradients from. 
                            For DINOv2: model.dinov2.encoder.layer[-1].output (last layer)
                            If None, will try to auto-detect
                use_cuda: Whether to use CUDA if available
            """
            self.model = model
            self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            self.target_layer = self.model.dinov2.encoder.layer[-1]
            self.gradients = None
            self.activations = None
            
            # Register hooks
            self._register_hooks()

        def _register_hooks(self):
            """Register forward and backward hooks."""
            def forward_hook(module, input, output):
                # For ViT, output is usually a tuple (hidden_states, attentions)
                if isinstance(output, tuple):
                    self.activations = output[0]
                else:
                    self.activations = output
            
            def backward_hook(module, grad_input, grad_output):
                if isinstance(grad_output, tuple):
                    self.gradients = grad_output[0]
                else:
                    self.gradients = grad_output
            
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)
        
        def generate_cam(self, input_image, target_class=None, eigen_smooth=False):
            """
            Generate Grad-CAM heatmap.
            
            Args:
                input_image: Input tensor (1, C, H, W) or (C, H, W)
                target_class: Target class index. If None, use predicted class
                eigen_smooth: Whether to apply eigen-smooth for better visualization
                
            Returns:
                cam: Normalized CAM heatmap
                prediction: Model prediction
            """
            # Ensure input has batch dimension
            if input_image.dim() == 3:
                input_image = input_image.unsqueeze(0)
            
            input_image = input_image.to(self.device)
            input_image.requires_grad = True
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(pixel_values=input_image)
            
            # Get logits
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            # Compute softmax probabilities
            probs = torch.softmax(logits, dim=1)

            # Predicted class + probability
            prediction = torch.argmax(probs, dim=1).item()
            prediction_prob = probs[0, prediction].item()
            
            # Use target class or predicted class
            target_class = prediction
            
            # Backward pass
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)
            
            # Get gradients and activations
            gradients = self.gradients.detach().cpu()
            activations = self.activations.detach().cpu()
            
            # For ViT: [batch, num_patches + 1, hidden_dim]
            # Remove CLS token (first token)
            if gradients.shape[1] > 1:
                gradients = gradients[:, 1:, :]  # Remove CLS token
                activations = activations[:, 1:, :]
            
            # Compute weights (global average pooling of gradients)
            weights = torch.mean(gradients, dim=1, keepdim=True)
            
            # Weighted combination
            cam = torch.sum(weights * activations, dim=-1)
            cam = cam.squeeze(0)  # Remove batch dimension
            
            # Apply ReLU
            cam = F.relu(cam)
            
            # Reshape to 2D (assuming square patches)
            patch_size = int(np.sqrt(cam.shape[0]))
            cam = cam.reshape(patch_size, patch_size)
            
            # Normalize
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam.numpy(), prediction, prediction_prob
        
        def visualize(self, input_image, original_image, 
                    alpha=0.5, colormap=cv2.COLORMAP_JET):
            """
            Visualize Grad-CAM overlay on original image.
            
            Args:
                input_image: Preprocessed input tensor
                original_image: Original PIL Image or numpy array
                target_class: Target class for CAM generation
                alpha: Transparency of overlay (0-1)
                colormap: OpenCV colormap for heatmap
                save_path: Path to save visualization
                
            Returns:
                visualization: Combined image with CAM overlay
            """
            # Generate CAM
            cam, prediction, prediction_prob = self.generate_cam(input_image)
            
            # Convert original image to numpy if needed
            if isinstance(original_image, Image.Image):
                original_image = np.array(original_image)
            
            # Resize CAM to match image size
            img_h, img_w = original_image.shape[:2]
            cam_resized = cv2.resize(cam, (img_w, img_h))
            
            # Apply colormap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Ensure original image is RGB
            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            
            # Overlay
            visualization = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
            
            return visualization, cam_resized, prediction, prediction_prob
