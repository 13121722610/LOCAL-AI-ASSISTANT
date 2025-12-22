# modules/vector_db.py - 完整修复版（支持归一化特征）
import chromadb
from typing import List, Tuple, Optional
import uuid
import numpy as np
import config

class VectorDB:
    """向量数据库管理"""
    
    def __init__(self):
        # ChromaDB 1.3.7 新版API
        self.client = chromadb.PersistentClient(
            path=str(config.DB_DIR)  # 新版不需要 Settings 类
        )
        
        # 创建或获取集合（新版API）
        self.text_collection = self.client.get_or_create_collection(
            name="papers"
        )
        
        self.image_collection = self.client.get_or_create_collection(
            name="images"
        )
        print("✅ VectorDB 初始化成功")
    
    def add_paper(self, pdf_path: str, chunks: List[str], 
                  embeddings: List[np.ndarray], metadata: dict = None):
        """添加论文到数据库"""
        if not chunks or len(chunks) == 0:
            print(f"⚠️  没有文本块可添加: {pdf_path}")
            return False
        
        # 生成唯一ID
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # 准备metadata
        if metadata is None:
            metadata = {}
        
        metadatas = []
        for i in range(len(chunks)):
            chunk_meta = metadata.copy()
            chunk_meta["source"] = pdf_path
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = len(chunks)
            metadatas.append(chunk_meta)
        
        # 转换为列表
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        try:
            # 添加到数据库
            self.text_collection.add(
                embeddings=embeddings_list,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ 添加成功: {len(chunks)} chunks from {pdf_path}")
            return True
            
        except Exception as e:
            print(f"❌ 添加失败 {pdf_path}: {e}")
            return False
    
    def add_image(self, image_path: str, embedding: np.ndarray, 
                  metadata: dict = None):
        """添加图像到数据库"""
        if metadata is None:
            metadata = {}
        
        metadata["source"] = image_path
        
        try:
            self.image_collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
            
            print(f"✅ 图片添加成功: {image_path}")
            return True
            
        except Exception as e:
            print(f"❌ 图片添加失败 {image_path}: {e}")
            return False
    
    def search_text(self, query_embedding: np.ndarray, k: int = config.SEARCH_TOP_K,
               filter_metadata: Optional[dict] = None) -> List[Tuple[float, str, dict]]:
        """在文本中搜索（按论文去重）"""
        try:
            results = self.text_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k * 3,  # 获取更多结果用于去重
                where=filter_metadata
            )
            
            formatted_results = []
            seen_papers = set()  # 记录已看到的论文
            
            if results['distances'] and results['documents']:
                distances = results['distances'][0]
                
                for i in range(len(distances)):
                    distance = distances[i]
                    
                    # 相似度计算
                    if distance < 0:
                        similarity = 1.0 / (1.0 + abs(distance))
                    else:
                        similarity = 1.0 / (1.0 + distance)
                    
                    similarity = max(0.0, min(1.0, similarity))
                    
                    document = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    source = metadata.get('source', '')
                    
                    # 按论文去重
                    if source and source not in seen_papers:
                        seen_papers.add(source)
                        formatted_results.append((similarity, document, metadata))
                    
                    # 达到要求的论文数量就停止
                    if len(formatted_results) >= k:
                        break
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def search_images(self, query_embedding: np.ndarray, k: int = config.SEARCH_TOP_K,
                     filter_metadata: Optional[dict] = None) -> List[Tuple[float, str, dict]]:
        """在图像中搜索（优化版，支持归一化特征）"""
        try:
            # 使用余弦相似度而不是默认的欧氏距离
            results = self.image_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=filter_metadata
            )
            
            formatted_results = []
            if results['distances'] and results['metadatas']:
                distances = results['distances'][0]
                
                for i in range(len(distances)):
                    # ChromaDB 返回的是欧氏距离的平方
                    # 对于归一化向量：distance² = 2*(1-cos_sim)
                    # 所以：cos_sim = 1 - distance²/2
                    distance_squared = distances[i]
                    
                    # 计算余弦相似度（假设特征已经L2归一化）
                    # 注意：这假设ChromaDB返回的是平方距离
                    cosine_similarity = 1.0 - (distance_squared / 2.0)
                    
                    # 确保在合理范围内（余弦相似度应该在-1到1之间）
                    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
                    
                    # 转换为0-1范围（对于展示更友好）
                    # 余弦相似度-1到1 → 映射到0-1
                    normalized_similarity = (cosine_similarity + 1.0) / 2.0
                    
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    formatted_results.append((normalized_similarity, metadata.get('source', ''), metadata))
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 图片搜索失败: {e}")
            return []
    
    def search_images_simple(self, query_embedding: np.ndarray, k: int = config.SEARCH_TOP_K,
                       filter_metadata: Optional[dict] = None) -> List[Tuple[float, str, dict]]:
        """在图像中搜索（简化版，确保返回结果）"""
        try:
            # 先尝试获取一些结果
            try:
                results = self.image_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k,
                    where=filter_metadata
                )
            except Exception as query_error:
                print(f"[警告] 查询失败: {query_error}")
                # 如果查询失败，直接获取所有图片
                results = self.image_collection.get()
                return self._search_manually(query_embedding, results, k)
            
            formatted_results = []
            
            if results.get('distances') and results['distances']:
                distances = results['distances'][0]
                metadatas = results['metadatas'][0] if results.get('metadatas') else []
                
                for i in range(len(distances)):
                    distance = distances[i]
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    # 距离转换为相似度（距离越小，相似度越高）
                    similarity = 1.0 / (1.0 + max(0, distance))
                    
                    img_path = metadata.get('source', '') or metadata.get('path', '')
                    if img_path:
                        formatted_results.append((similarity, img_path, metadata))
            
            # 如果没找到结果，尝试手动搜索
            if not formatted_results:
                all_results = self.image_collection.get()
                return self._search_manually(query_embedding, all_results, k)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 图片搜索失败: {e}")
            return []
    
    def _search_manually(self, query_embedding: np.ndarray, all_results: dict, k: int):
        """手动计算相似度"""
        formatted_results = []
        
        if all_results.get('embeddings') and all_results['embeddings']:
            embeddings = all_results['embeddings']
            metadatas = all_results['metadatas'] if all_results.get('metadatas') else []
            
            for i, emb in enumerate(embeddings):
                try:
                    stored_emb = np.array(emb).flatten()
                    query_emb = query_embedding.flatten()
                    
                    # 计算余弦相似度
                    dot_product = np.dot(query_emb, stored_emb)
                    query_norm = np.linalg.norm(query_emb)
                    stored_norm = np.linalg.norm(stored_emb)
                    
                    if query_norm > 0 and stored_norm > 0:
                        similarity = dot_product / (query_norm * stored_norm)
                        similarity = max(0.0, min(1.0, similarity))
                        
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        img_path = metadata.get('source', '') or metadata.get('path', '')
                        
                        if img_path:
                            formatted_results.append((similarity, img_path, metadata))
                except Exception as e:
                    print(f"[警告] 计算相似度失败: {e}")
                    continue
        
        # 按相似度排序
        formatted_results.sort(key=lambda x: x[0], reverse=True)
        return formatted_results[:k]
    
    def get_all_papers(self) -> List[str]:
        """获取所有论文路径"""
        try:
            results = self.text_collection.get()
            sources = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'source' in metadata:
                        sources.add(metadata['source'])
            return list(sources)
        except Exception as e:
            print(f"❌ 获取论文列表失败: {e}")
            return []
    
    def clear_database(self):
        """清空数据库"""
        try:
            # 删除集合
            self.client.delete_collection("papers")
            self.client.delete_collection("images")
            
            # 重新创建空集合
            self.text_collection = self.client.get_or_create_collection(name="papers")
            self.image_collection = self.client.get_or_create_collection(name="images")
            
            print("✅ 数据库已清空")
            return True
            
        except Exception as e:
            print(f"❌ 清空数据库失败: {e}")
            return False
    
    def get_collection_stats(self):
        """获取数据库统计信息"""
        stats = {
            "text_collection": {
                "name": self.text_collection.name,
                "count": self.text_collection.count()
            },
            "image_collection": {
                "name": self.image_collection.name,
                "count": self.image_collection.count()
            }
        }
        return stats
    
    def debug_search(self, query_embedding: np.ndarray):
        """调试搜索功能"""
        print("🔍 调试搜索信息:")
        print(f"查询向量形状: {query_embedding.shape}")
        print(f"查询向量范数: {np.linalg.norm(query_embedding):.6f}")
        
        # 获取数据库中的一些样本
        sample_results = self.image_collection.get(limit=2)
        if sample_results['embeddings']:
            sample_emb = np.array(sample_results['embeddings'][0])
            print(f"样本向量范数: {np.linalg.norm(sample_emb):.6f}")
            
            # 手动计算相似度
            cos_sim = np.dot(query_embedding, sample_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(sample_emb)
            )
            print(f"手动计算余弦相似度: {cos_sim:.6f}")

