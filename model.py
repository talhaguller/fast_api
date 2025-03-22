import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
import torchvision.models as models
from scipy.spatial.distance import cosine
from tqdm import tqdm

def extract_features(image_tensor, model, device):
    with torch.no_grad():
        features = model(image_tensor)
    features = features.squeeze(-1).squeeze(-1)
    return features.cpu().numpy()

class LogoDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor, image_path
        except Exception as e:
            print(f"Resim işlenemedi: {image_path} - {e}")
            return None, image_path

def process_logo(user_image_path, logo_folder, threshold=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    try:
        user_image = Image.open(user_image_path).convert('RGB')
        user_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        user_tensor = user_transform(user_image).unsqueeze(0).to(device)
        user_features = extract_features(user_tensor, model, device)[0]
        user_features = user_features / np.linalg.norm(user_features)
    except Exception as e:
        return {"error": f"Kullanıcı logosu işlenemedi: {e}"}

    logo_paths = [os.path.join(logo_folder, f) for f in os.listdir(logo_folder) 
                  if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    dataset = LogoDataset(logo_paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    similarities = []
    for batch in tqdm(dataloader, desc="Logolar işleniyor"):
        images, paths = batch
        if images is None:
            continue
        
        valid_images = []
        valid_paths = []
        for img, path in zip(images, paths):
            if img is not None:
                valid_images.append(img)
                valid_paths.append(path)
        
        if not valid_images:
            continue
        
        img_tensor = torch.stack(valid_images).to(device)
        batch_features = extract_features(img_tensor, model, device)
        
        for feat, path in zip(batch_features, valid_paths):
            feat = feat / np.linalg.norm(feat)
            cosine_sim = 1 - cosine(user_features, feat)
            if cosine_sim >= threshold:  # Eşik değere göre filtrele
                similarities.append((cosine_sim, path))

    similarities.sort(reverse=True)
    return similarities  # Tüm eşik üzeri sonuçları döndür (10 ile sınırlamıyoruz)