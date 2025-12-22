# config.py - 简化版
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
IMAGES_DIR = DATA_DIR / "images"
DB_DIR = DATA_DIR / "chroma_db"

# 创建目录
for dir_path in [DATA_DIR, PAPERS_DIR, IMAGES_DIR, DB_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 模型配置
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"

# 分类主题
TOPICS = ["CV", "NLP", "RL", "ML", "Robotics", "Other"]

# 创建主题子文件夹
for topic in TOPICS:
    topic_dir = PAPERS_DIR / topic
    topic_dir.mkdir(exist_ok=True)

# 模型参数
EMBEDDING_DIM = 384
IMAGE_EMBEDDING_DIM = 512
BATCH_SIZE = 32
CHUNK_SIZE = 1000

# 搜索参数
SEARCH_TOP_K = 5
SIMILARITY_THRESHOLD = 0.5


# 删除所有 ChromaDB 相关配置！
# 不再需要 CHROMA_SETTINGS 或 Settings 导入