from torchvision import transforms
from PIL import Image

def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def preprocess_image(img_path, img_size=224):
    image = Image.open(img_path).convert('RGB')
    transform = get_transforms(img_size)
    return transform(image).unsqueeze(0)  # Add batch dim
