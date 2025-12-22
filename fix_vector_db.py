# 修复 vector_db.py 使用新版API
import re

with open('modules/vector_db.py', 'r') as f:
    content = f.read()

# 替换旧的初始化代码
old_code = '''    def __init__(self):
        # 新版 ChromaDB 客户端初始化
        self.client = chromadb.PersistentClient(
            path=str(config.DB_DIR),  # 持久化路径
            settings=Settings(
                anonymized_telemetry=False,
                chroma_db_impl="duckdb+parquet"
            )
        )'''

new_code = '''    def __init__(self):
        # ChromaDB 1.3.7 新版API
        self.client = chromadb.PersistentClient(
            path=str(config.DB_DIR)  # 新版不需要 Settings 类
        )'''

# 替换
content = content.replace(old_code, new_code)

# 删除 Settings 导入（如果存在）
content = content.replace('from chromadb.config import Settings\n', '')
content = content.replace(', Settings', '')

with open('modules/vector_db.py', 'w') as f:
    f.write(content)

print('✅ 已修复 vector_db.py')
