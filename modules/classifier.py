# classifier.py
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import config
from .text_processor import TextProcessor

class Classifier:
    """分类器模块"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
    
    def classify_by_keywords(self, text: str, topics: List[str] = None) -> str:
        """基于关键词分类"""
        if topics is None:
            topics = config.TOPICS
        
        # 关键词映射（可扩展）
        keyword_map = {
            "CV": ["computer vision", "image", "vision", "detection", "segmentation", 
                   "recognition", "convolutional", "cnn", "yolo"],
            "NLP": ["natural language", "nlp", "transformer", "bert", "gpt", 
                    "language model", "text", "attention"],
            "RL": ["reinforcement", "rl", "q-learning", "policy", "agent", 
                   "reward", "environment", "dqn"],
            "ML": ["machine learning", "regression", "classification", 
                   "clustering", "svm", "random forest", "neural network"],
            "Robotics": ["robot", "robotics", "manipulation", "motion", 
                         "kinematics", "control", "slam"]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for topic in topics:
            if topic in keyword_map:
                score = sum(1 for keyword in keyword_map[topic] 
                           if keyword in text_lower)
                scores[topic] = score
        
        if not scores:
            return "Other"
        
        # 返回得分最高的主题
        best_topic = max(scores, key=scores.get)
        return best_topic if scores[best_topic] > 0 else "Other"
    
    def classify_pdf(self, pdf_path: str, topics: List[str] = None) -> str:
        """分类PDF文件"""
        from .file_utils import FileUtils
        
        text = FileUtils.extract_text_from_pdf(pdf_path)
        if not text:
            return "Other"
        
        # 获取摘要进行分类
        summary = text[:2000]  # 使用前2000字符
        topic = self.classify_by_keywords(summary, topics)
        
        return topic 

        
