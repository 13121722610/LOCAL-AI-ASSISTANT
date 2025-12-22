import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
from .file_utils import FileUtils
import config

class TextProcessor:
    """文本处理模块"""
    
    def __init__(self):
        self.model = SentenceTransformer(config.TEXT_MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Text model loaded on {self.device}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码单个文本为向量"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[str], List[np.ndarray]]:
        """处理PDF文件，返回文本chunks和对应的向量"""
        # 提取文本
        text = FileUtils.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"Warning: No text extracted from {pdf_path}")
            return [], []
        
        # 分割文本
        chunks = FileUtils.split_text(text)
        
        # 生成向量
        if len(chunks) == 0:
            return [], []
        
        embeddings = self.encode_texts(chunks)
        
        return chunks, embeddings
    
    def get_pdf_summary(self, pdf_path: str, max_chars: int = 500) -> str:
        """获取PDF摘要（前N个字符）"""
        text = FileUtils.extract_text_from_pdf(pdf_path)
        return text[:max_chars] + "..." if len(text) > max_chars else text