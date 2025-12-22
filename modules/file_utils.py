import os
import shutil
from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm
import config

class FileUtils:
    """文件处理工具类"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str, method: str = "fitz") -> str:
        """从PDF提取文本"""
        try:
            if method == "fitz":
                # 使用PyMuPDF
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
            elif method == "pdfplumber":
                # 使用pdfplumber（保持布局）
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    @staticmethod
    def get_all_pdfs(folder_path: str) -> List[str]:
        """获取文件夹中所有PDF文件"""
        pdf_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files
    
    @staticmethod
    def get_all_images(folder_path: str) -> List[str]:
        """获取文件夹中所有图片文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        return image_files
    
    @staticmethod
    def organize_file(source_path: str, target_topic: str) -> str:
        """将文件整理到对应的主题文件夹"""
        filename = os.path.basename(source_path)
        target_dir = config.PAPERS_DIR / target_topic
        target_path = target_dir / filename
        
        # 如果目标文件已存在，添加序号
        counter = 1
        while target_path.exists():
            name, ext = os.path.splitext(filename)
            target_path = target_dir / f"{name}_{counter}{ext}"
            counter += 1
        
        # 移动文件
        shutil.move(source_path, target_path)
        return str(target_path)
    
    @staticmethod
    def split_text(text: str, chunk_size: int = config.CHUNK_SIZE) -> List[str]:
        """将长文本分割为chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
            
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks