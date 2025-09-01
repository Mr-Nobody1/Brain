import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys
from pathlib import Path

# Model architectures - Multiple variations to match your trained models

class CustomCNN_V1(nn.Module):
    """Version 1: Sequential features structure"""
    def __init__(self, num_classes=2):
        super(CustomCNN_V1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CustomCNN_V2(nn.Module):
    """Version 2: Conv block structure (matches your model!)"""
    def __init__(self, num_classes=2):
        super(CustomCNN_V2, self).__init__()
        # Conv blocks as separate modules
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Adaptive pooling and classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

class CustomCNN_V3(nn.Module):
    """Version 3: Simpler structure that might match your model"""
    def __init__(self, num_classes=2):
        super(CustomCNN_V3, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Single output for binary classification
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

class OneLayerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(OneLayerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def detect_model_type(model_name):
    """Auto-detect model type from filename"""
    model_name = model_name.lower()
    
    if 'efficientnet' in model_name:
        return 'efficientnet'
    elif 'onelayer' in model_name:
        return 'onelayer'
    elif 'custom' in model_name or 'cnn' in model_name:
        return 'custom_cnn'
    else:
        return 'custom_cnn'  # Default

def load_model(model_path, model_type):
    """Load the appropriate model architecture and weights with smart architecture detection"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint first to examine keys
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'best_accuracy' in checkpoint:
                print(f"📊 Model accuracy: {checkpoint['best_accuracy']:.2%}")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Analyze the state dict keys to determine architecture
    keys = list(state_dict.keys())
    print(f"🔍 Analyzing model keys: {keys[:5]}...")  # Show first 5 keys
    
    model = None
    model_loaded = False
    
    # Initialize model based on detected type
    if model_type == 'onelayer':
        model = OneLayerCNN(num_classes=2)
        print("🏗️  Using One Layer CNN architecture")
        try:
            model.load_state_dict(state_dict)
            model_loaded = True
        except Exception as e:
            print(f"⚠️  OneLayer CNN failed: {e}")
            
    elif model_type == 'efficientnet':
        try:
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=False)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, 2)
            print("🏗️  Using EfficientNet-B0 architecture")
            model.load_state_dict(state_dict)
            model_loaded = True
        except ImportError:
            print("⚠️  EfficientNet not available")
        except Exception as e:
            print(f"⚠️  EfficientNet loading failed: {e}")
            
    else:  # custom_cnn - try multiple versions
        print("🏗️  Trying Custom CNN architectures...")
        
        # Check if it has conv_block structure (your model!)
        has_conv_blocks = any('conv_block' in key for key in keys)
        has_single_output = any('classifier.5.weight' in key and state_dict[key].shape[0] == 1 for key in keys if 'classifier.5.weight' in key)
        
        if has_conv_blocks:
            print("🔍 Detected conv_block structure - trying CustomCNN_V2 and V3...")
            
            # Try V3 first (single output)
            if has_single_output:
                try:
                    model = CustomCNN_V3(num_classes=2)
                    print("🏗️  Trying CustomCNN_V3 (single output)...")
                    model.load_state_dict(state_dict)
                    model_loaded = True
                    print("✅ CustomCNN_V3 loaded successfully!")
                except Exception as e:
                    print(f"⚠️  CustomCNN_V3 failed: {e}")
            
            # Try V2 if V3 failed
            if not model_loaded:
                try:
                    model = CustomCNN_V2(num_classes=2)
                    print("🏗️  Trying CustomCNN_V2...")
                    model.load_state_dict(state_dict)
                    model_loaded = True
                    print("✅ CustomCNN_V2 loaded successfully!")
                except Exception as e:
                    print(f"⚠️  CustomCNN_V2 failed: {e}")
        
        # Try V1 if others failed
        if not model_loaded:
            try:
                model = CustomCNN_V1(num_classes=2)
                print("🏗️  Trying CustomCNN_V1 (sequential features)...")
                model.load_state_dict(state_dict)
                model_loaded = True
                print("✅ CustomCNN_V1 loaded successfully!")
            except Exception as e:
                print(f"⚠️  CustomCNN_V1 failed: {e}")
    
    if not model_loaded:
        print("❌ All architecture attempts failed. Let's try with strict=False...")
        # Final attempt with most likely architecture and strict=False
        if model is None:
            model = CustomCNN_V3(num_classes=2)
        
        try:
            model.load_state_dict(state_dict, strict=False)
            print("⚠️  Loaded with strict=False - some layers may be mismatched")
            model_loaded = True
        except Exception as e:
            print(f"❌ Final attempt failed: {e}")
            return None, None
    
    if model_loaded:
        model.to(device)
        model.eval()
        print("✅ Model successfully loaded and ready!")
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total parameters: {total_params:,}")
        
        return model, device
    else:
        return None, None

def select_model_from_folder(models_folder):
    """Interactive model selection from folder"""
    models_path = Path(models_folder)
    
    if not models_path.exists():
        print(f"❌ Models folder not found: {models_folder}")
        return None
    
    # Find all .pth files
    model_files = list(models_path.glob("*.pth"))
    
    if not model_files:
        print(f"❌ No .pth files found in: {models_folder}")
        return None
    
    print(f"\n📁 Found {len(model_files)} model(s) in {models_folder}:")
    print("=" * 60)
    
    # Display models with details
    for i, model_file in enumerate(model_files, 1):
        model_type = detect_model_type(model_file.name)
        file_size = model_file.stat().st_size / (1024*1024)  # Size in MB
        
        print(f"{i:2d}. {model_file.name}")
        print(f"     Type: {model_type} | Size: {file_size:.1f} MB")
        print()
    
    # Get user selection
    while True:
        try:
            choice = input(f"Select model (1-{len(model_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(model_files):
                selected_model = model_files[choice_num - 1]
                print(f"\n✅ Selected: {selected_model.name}")
                return selected_model
            else:
                print(f"❌ Please enter a number between 1 and {len(model_files)}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")

def get_image_path():
    """Interactive image path input with validation"""
    while True:
        image_path = input("\n📸 Enter image path (or 'q' to quit): ").strip().strip('"')
        
        if image_path.lower() == 'q':
            return None
        
        image_file = Path(image_path)
        if image_file.exists() and image_file.is_file():
            # Check if it's an image file
            valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
            if image_file.suffix.lower() in valid_extensions:
                return image_file
            else:
                print(f"❌ Not a valid image file. Supported: {', '.join(valid_extensions)}")
        else:
            print(f"❌ File not found: {image_path}")

def classify_image(model, device, image_path):
    """Classify a single image"""
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        print(f"📐 Original image size: {image.size}")
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        print(f"📐 Preprocessed tensor shape: {input_tensor.shape}")
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Handle different output formats
            if outputs.shape[1] == 1:  # Single output (sigmoid)
                # Apply sigmoid for binary classification
                sigmoid_output = torch.sigmoid(outputs)
                prob_positive = sigmoid_output.item()
                prob_negative = 1 - prob_positive
                probabilities = torch.tensor([[prob_negative, prob_positive]])
                print(f"🔍 Single output detected, using sigmoid: {sigmoid_output.item():.4f}")
            else:  # Multiple outputs (softmax)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                print(f"🔍 Multiple outputs detected, using softmax")
        
        # Results
        class_names = ['BreastHisto', 'MRI']
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 PREDICTION RESULTS")
        print("=" * 50)
        print(f"🎯 Predicted Class: {class_names[predicted_idx]}")
        print(f"🎲 Confidence: {confidence:.2%}")
        print(f"\n📈 Class Probabilities:")
        
        for i, class_name in enumerate(class_names):
            prob = probabilities[0][i].item()
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {class_name:12}: {prob:.2%} |{bar}|")
        
        print(f"\n🔢 Raw outputs: {outputs.cpu().numpy()[0]}")
        
        return {
            'predicted_class': class_names[predicted_idx],
            'confidence': confidence,
            'probabilities': {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}
        }
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🧠 MRI vs BreastHisto Image Classifier")
    print("=" * 40)
    
    if len(sys.argv) != 2:
        print("Usage: python classifier.py <models_folder_path>")
        print("Example: python classifier.py models/")
        sys.exit(1)
    
    models_folder = sys.argv[1]
    
    # Step 1: Select model
    selected_model_path = select_model_from_folder(models_folder)
    if selected_model_path is None:
        sys.exit(1)
    
    # Step 2: Load model
    model_type = detect_model_type(selected_model_path.name)
    print(f"\n🔄 Loading model...")
    print(f"🖥️  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    model, device = load_model(selected_model_path, model_type)
    if model is None:
        print("❌ Failed to load model")
        sys.exit(1)
    
    print("✅ Model loaded successfully!")
    
    # Step 3: Interactive image classification
    print(f"\n🎯 Ready for classification!")
    print("💡 You can classify multiple images. Type 'q' to quit.")
    
    while True:
        # Get image path
        image_path = get_image_path()
        if image_path is None:
            break
        
        # Classify image
        print(f"\n🔍 Analyzing: {image_path.name}")
        result = classify_image(model, device, image_path)
        
        if result:
            # Ask if user wants to classify another image
            print(f"\n" + "-" * 50)
            continue_choice = input("Classify another image? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
        else:
            print("❌ Classification failed")
    
    print("\n👋 Thank you for using the classifier!")

if __name__ == "__main__":
    main()