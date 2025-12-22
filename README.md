# 【Local AI Assistant - 本地多模态AI智能助手】

# 🎯 项目概述
Local AI Assistant 是一个基于Python的本地多模态AI智能助手，旨在解决本地大量文献和图像素材管理困难的问题。不同于传统的文件名搜索，本项目利用多模态神经网络技术，实现智能文献管理和智能图像管理。

# 🚀 快速开始
【一键安装：】  
·创建虚拟环境：  
conda create -n ai-assistant python=3.10 -y  
conda activate ai-assistant  

·安装依赖：  
pip install -r requirements.txt  

·安装Web界面依赖：  
pip install gradio  

# 📖 使用指南
【命令行模式：】  
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

【Web界面模式：】

·启动Web服务  
python web_app.py  

访问 http://localhost:7860 打开Web界面  

# 🎨 可视化Web界面
<img width="1627" height="898" alt="image" src="https://github.com/user-attachments/assets/e9c564eb-657d-4de9-9bf2-9cb8b55a6536" />
<img width="1570" height="769" alt="image" src="https://github.com/user-attachments/assets/b7ea9166-1488-40be-80a0-10e90bd232dd" />

# 📞 支持与联系
电子邮件: 25120410@bjtu.edu.cn  
电话：13121722610




