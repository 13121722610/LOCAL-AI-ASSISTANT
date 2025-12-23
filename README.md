# 【Local AI Assistant - 本地多模态AI智能助手】

# 🎯 项目概述
Local AI Assistant 是一个基于Python的本地多模态AI智能助手，旨在解决本地大量文献和图像素材管理困难的问题。不同于传统的文件名搜索，本项目利用多模态神经网络技术，实现智能文献管理和智能图像管理。

# 📚 核心功能列表
### 智能文献管理
*   **语义搜索**: 支持使用自然语言提问（如“Transformer 的核心架构是什么？”）。系统需基于语义理解返回最相关的论文文件，进阶要求可返回具体的论文片段或页码。
*   **自动分类与整理**:
    *   **单文件处理**: 添加新论文时，根据指定的主题（如 "CV, NLP, RL"）自动分析内容，将其归类并移动到对应的子文件夹中。
    *   **批量整理**: 支持对现有的混乱文件夹进行“一键整理”，自动扫描所有 PDF，识别主题并归档到相应目录。
*   **文件索引**: 支持仅返回相关文件列表，方便快速定位所需文献。

### 智能图像管理
*   **以文搜图**: 利用多模态图文匹配技术，支持通过自然语言描述（如“海边的日落”）来查找本地图片库中最匹配的图像。

# 🚀 环境配置与依赖安装说明
【环境要求：】  
Python: 3.8+  
操作系统: Windows / macOS / Linux  
内存: 8GB+ (推荐16GB)  

【一键安装：】  
·创建虚拟环境：  
conda create -n ai-assistant python=3.10 -y  
conda activate ai-assistant  

·安装依赖：  
pip install -r requirements.txt  

·安装Web界面依赖：  
pip install gradio  

# 📖 使用说明
### 【命令行模式：】  
·添加论文（自动分类）  
python main.py add_paper "your_paper.pdf"  

·语义搜索论文  
python main.py search_paper "your_paperxxxxxxx"  

·添加图片  
python main.py add_image "your_image.jpg"  

·以文搜图  
python main.py search_image "your_imageyyyyy"  

·批量整理文件夹  
python main.py organize "downloads_folder"  

·列出所有内容  
python main.py list_papers  
python main.py list_images  

### 【Web界面模式：】
  
·设置环境变量，让Gradio使用当前目录
export GRADIO_TEMP_DIR="./gradio_temp"
mkdir -p gradio_temp
··启动Web服务
python web_app.py  

访问 http://localhost:7860 打开Web界面  

# 🎨 可视化Web界面
<img width="650" height="359" alt="image" src="https://github.com/user-attachments/assets/e9c564eb-657d-4de9-9bf2-9cb8b55a6536" />
<img width="628" height="307" alt="image" src="https://github.com/user-attachments/assets/b7ea9166-1488-40be-80a0-10e90bd232dd" />

# 🏗️ 技术选型说明 
### 【文本处理技术选型：】  
1、文本嵌入模型：Sentence-BERT (all-MiniLM-L6-v2)  
模型名称: sentence-transformers/all-MiniLM-L6-v2  
嵌入维度: 384维  
模型大小: 80MB  
推理速度: 极快 (CPU: 1000句/秒, GPU: 5000句/秒)  
准确率: 高 (在语义相似度任务上达到~85%准确率)  

2. PDF文本提取：PyMuPDF (fitz)  
库名称: PyMuPDF  
功能: PDF文本、图像、元数据提取  

### 【图像处理技术选型：】  
1. 多模态嵌入模型：CLIP (Vision Transformer Base)  
模型名称: openai/clip-vit-base-patch32  
图像编码维度: 512维  
文本编码维度: 512维  
模型大小: 400MB  
特点:  
• 同一向量空间编码图像和文本  
• 零样本学习能力  
• 强大的跨模态检索性能  

### 【向量数据库技术选型：】  
1. 核心数据库：ChromaDB  
版本: ChromaDB 1.3.7 (新版API)  
类型: 嵌入式向量数据库  
存储格式: DuckDB + Parquet  

### 【Web界面技术选型：】  
1. 前端框架：Gradio  
版本: Gradio 3.50.2 (兼容旧版依赖)  
特点:  
• 专为AI应用设计的Python框架  
• 支持实时交互和文件上传  
• 自动生成可分享的Web链接  

界面组件:  
• 文本输入/输出  
• 文件上传组件  
• 图片画廊展示  

# 🧠 运行结果截图

<img width="712" height="232" alt="image" src="https://github.com/user-attachments/assets/4e40f2c6-416d-4169-a755-23e8f1e2e295" />
<img width="570" height="236" alt="image" src="https://github.com/user-attachments/assets/9fa98673-2239-4ea9-9c64-4890436629b9" />
<img width="943" height="416" alt="image" src="https://github.com/user-attachments/assets/c97ab7d3-7ae7-43c4-aaf3-993038d3e213" />
<img width="906" height="515" alt="image" src="https://github.com/user-attachments/assets/7b161a92-cdf6-4fe2-8727-d997ebf3c6fc" />



# 📞 支持与联系
电子邮件: 25120410@bjtu.edu.cn  
电话：13121722610







