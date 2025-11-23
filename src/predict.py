import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models


class MosquitoClassifier:
    def __init__(self, model_path, mean_std_path):
        """
        Initialize the mosquito classifier
        
        Args:
            model_path: Path to the trained model (.pt file)
            mean_std_path: Path to the mean and std normalization values
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load mean and std
        mean_std = torch.load(mean_std_path, map_location=self.device)
        self.mean = mean_std['mean']
        self.std = mean_std['std']
        
        # Define transforms (same as test transforms)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Class names (0: non-dengue, 1: dengue)
        self.class_names = ['Non-Dengue Mosquito', 'Dengue Mosquito (Aedes)']
    
    def _load_model(self, model_path):
        """Load the DenseNet161 model with custom classifier"""
        # Create model architecture (same as training)
        model = models.densenet161(pretrained=False)
        
        # Get number of features (DenseNet uses 'classifier', but training code uses 'fc')
        # Check which attribute exists and use it
        if hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            # Replace fc layer (same architecture as training)
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2),  # 2 classes
            )
        else:
            num_ftrs = model.classifier.in_features
            # Replace classifier (same architecture as training)
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2),  # 2 classes
            )
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        
        return model
    
    def predict(self, image_path):
        """
        Predict whether the mosquito is dengue or not
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Contains prediction, confidence, and probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        return {
            'prediction': self.class_names[predicted_class],
            'is_dengue': predicted_class == 1,
            'confidence': confidence_score,
            'probabilities': {
                'Non-Dengue': probabilities[0][0].item(),
                'Dengue': probabilities[0][1].item()
            }
        }
