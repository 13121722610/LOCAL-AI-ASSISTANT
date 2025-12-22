# image_processor.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List
import config

class ImageProcessor:
    """图像处理模块"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL_NAME)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print(f"Image model loaded on {self.device}")
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """编码单个图像为向量（L2归一化）"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(
                    inputs["pixel_values"].to(self.device)
                )
                # L2 归一化 - 关键修复！
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy()[0]
            
            return embedding
        except Exception as e:
            print(f"❌ 处理图片失败 {image_path}: {e}")
            return np.zeros(config.IMAGE_EMBEDDING_DIM)
    
    def encode_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """批量编码图像（L2归一化）"""
        embeddings = []
        for path in image_paths:
            embedding = self.encode_image(path)
            embeddings.append(embedding)
        return embeddings
    
    def encode_text_for_image_search(self, text: str) -> np.ndarray:
        """编码文本用于图像搜索（L2归一化）"""
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            text_features = self.model.get_text_features(
                inputs["input_ids"].to(self.device)
            )
            # L2 归一化 - 关键修复！
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embedding = text_features.cpu().numpy()[0]
        
        return embedding
    
    def compute_similarity(self, image_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """计算余弦相似度"""
        # 转换为numpy数组
        img_vec = np.array(image_embedding)
        text_vec = np.array(text_embedding)
        
        # 余弦相似度 = (A·B) / (||A|| * ||B||)
        # 由于已经归一化，||A|| = ||B|| = 1，所以简化为点积
        similarity = np.dot(img_vec, text_vec)
        
        # 确保在-1到1之间（理论上应该在0-1之间，因为特征都是正的）
        similarity = max(-1.0, min(1.0, similarity))
        
        return float(similarity)
    
    def test_normalization(self):
        """测试归一化效果"""
        print("\n🔧 测试特征归一化...")
        
        # 测试文本编码归一化
        test_text = "a photo of sunset"
        text_emb = self.encode_text_for_image_search(test_text)
        text_norm = np.linalg.norm(text_emb)
        print(f"文本特征 '{test_text}' 的范数: {text_norm:.6f}")
        
        # 如果有图片，测试图片编码归一化
        import os
        if os.path.exists("1.png"):
            img_emb = self.encode_image("1.png")
            img_norm = np.linalg.norm(img_emb)
            print(f"图片 '1.png' 的范数: {img_norm:.6f}")
            
            # 计算相似度
            similarity = self.compute_similarity(img_emb, text_emb)
            print(f"图片与文本的余弦相似度: {similarity:.4f}")
        
        print("✅ 归一化测试完成")

