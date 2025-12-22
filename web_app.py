# web_app.py
import gradio as gr
import sys
from pathlib import Path

# 添加项目路径
sys.path.append('.')
from modules.text_processor import TextProcessor
from modules.image_processor import ImageProcessor
from modules.vector_db import VectorDB
from modules.classifier import Classifier
from modules.file_utils import FileUtils
import config

class WebAssistant:
    def __init__(self):
        print("正在初始化AI助手...")
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.vector_db = VectorDB()
        self.classifier = Classifier()
        print("✅ 初始化完成")
    
    def search_papers(self, query, top_k=5):
        """搜索论文"""
        try:
            query_embedding = self.text_processor.encode_text(query)
            results = self.vector_db.search_text(query_embedding, k=top_k)
            
            if not results:
                return "没有找到相关论文"
            
            output = f"找到 {len(results)} 篇相关论文：\n\n"
            for i, (score, document, metadata) in enumerate(results, 1):
                source = metadata.get('source', 'Unknown')
                topic = metadata.get('topic', 'Unknown')
                filename = Path(source).name
                
                output += f"**{i}. [{topic}] {filename}** (相似度: {score:.3f})\n"
                output += f"   路径: `{source}`\n"
                output += f"   预览: {document[:150]}...\n\n"
            
            return output
        except Exception as e:
            return f"搜索失败: {str(e)}"
    
    def search_images(self, query, top_k=5):
        """搜索图片"""
        try:
            query_embedding = self.image_processor.encode_text_for_image_search(query)
            results = self.vector_db.search_images(query_embedding, k=top_k)
            
            if not results:
                return "没有找到相关图片", []
            
            # 只取第一条结果（相似度最高的）
            output = f"找到相关图片：\n\n"
            image_paths = []
            
            # 只处理第一个结果
            score, img_path, metadata = results[0]
            
            # 尝试多种方式获取路径
            paths_to_try = [
                img_path,
                metadata.get('path') if metadata else None,
                metadata.get('source') if metadata else None,
                metadata.get('organized_path') if metadata else None
            ]
            
            found = False
            for path in paths_to_try:
                if path and Path(path).exists():
                    image_paths.append(path)
                    filename = Path(path).name
                    output += f"**1. {filename}** (相似度: {score:.3f})\n"
                    found = True
                    break
            
            if not found:
                output += "1. 图片路径无效\n"
            
            return output, image_paths
        except Exception as e:
            return f"搜索失败: {str(e)}", []
    
    def add_paper(self, file):
        """添加论文"""
        try:
            if not file:
                return "请选择PDF文件"
            
            # 保存上传的文件
            upload_dir = Path("data/uploads")
            upload_dir.mkdir(exist_ok=True)
            file_path = upload_dir / Path(file.name).name
            
            with open(file_path, "wb") as f:
                f.write(file.read())
            
            # 处理论文
            chunks, embeddings = self.text_processor.process_pdf(str(file_path))
            if not chunks:
                return "无法提取文本内容"
            
            # 分类
            topic = self.classifier.classify_pdf(str(file_path))
            
            # 整理文件
            target_path = FileUtils.organize_file(str(file_path), topic)
            
            # 添加到数据库
            metadata = {
                "title": file_path.stem,
                "topic": topic,
                "original_path": str(file_path),
                "organized_path": target_path
            }
            
            success = self.vector_db.add_paper(target_path, chunks, embeddings, metadata)
            
            if success:
                return f"✅ 论文添加成功！\n分类: {topic}\n保存到: {target_path}"
            else:
                return "❌ 添加到数据库失败"
                
        except Exception as e:
            return f"处理失败: {str(e)}"
    
    def add_images(self, files):
        """添加图片"""
        try:
            if not files:
                return "请选择图片文件"
            
            success_count = 0
            total_count = len(files)
            
            output_messages = []
            for file_info in files:
                file_path = file_info.name
                filename = Path(file_path).name
                print(f"[上传] 处理: {filename}")
                
                try:
                    # 编码图片
                    embedding = self.image_processor.encode_image(file_path)
                    
                    # 添加到数据库
                    metadata = {
                        "filename": filename,
                        "path": file_path,
                        "size": Path(file_path).stat().st_size
                    }
                    
                    if self.vector_db.add_image(file_path, embedding, metadata):
                        success_count += 1
                        output_messages.append(f"✅ {filename}: 添加成功")
                    else:
                        output_messages.append(f"❌ {filename}: 添加到数据库失败")
                        
                except Exception as e:
                    output_messages.append(f"❌ {filename}: 处理失败 - {str(e)}")
            
            summary = f"### 上传完成\n成功: {success_count}/{total_count} 张\n\n"
            summary += "\n".join(output_messages)
            
            return summary
            
        except Exception as e:
            return f"❌ 上传失败: {str(e)}"
    
    def get_database_stats(self):
        """获取数据库统计信息"""
        try:
            stats = self.vector_db.get_collection_stats()
            text_count = stats["text_collection"]["count"]
            image_count = stats["image_collection"]["count"]
            
            output = f"""
            ## 📊 数据库统计
            - **论文数量**: {text_count} 篇
            - **图片数量**: {image_count} 张
            - **总数据量**: {text_count + image_count} 条记录
            
            ## 📁 文件结构
            ```
            data/
            ├── papers/
            │   ├── CV/      (计算机视觉)
            │   ├── NLP/     (自然语言处理)
            │   ├── ML/      (机器学习)
            │   └── Other/   (其他)
            ├── images/      (图片库)
            └── uploads/     (上传临时文件)
            ```
            """
            return output
        except Exception as e:
            return f"获取统计信息失败: {str(e)}"

def create_interface():
    assistant = WebAssistant()
    
    with gr.Blocks(title="本地AI智能助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 本地AI智能助手
        智能管理你的文献和图像素材
        """)
        
        with gr.Tabs():
            # Tab 1: 论文管理
            with gr.TabItem("📄 论文管理"):
                with gr.Row():
                    with gr.Column(scale=3):
                        paper_query = gr.Textbox(
                            label="搜索论文",
                            placeholder="输入搜索内容，如：'transformer architecture'",
                            lines=2
                        )
                        with gr.Row():
                            paper_search_btn = gr.Button("🔍 搜索", variant="primary")
                            paper_top_k = gr.Slider(1, 20, value=5, label="显示数量", scale=2)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 上传新论文")
                        paper_upload = gr.File(
                            label="选择PDF文件",
                            file_types=[".pdf"]
                        )
                        paper_upload_btn = gr.Button("📤 上传并分类", variant="secondary")
                
                paper_output = gr.Markdown(label="搜索结果")
            
            # Tab 2: 图片管理
            with gr.TabItem("🖼️ 图片管理"):
                with gr.Row():
                    with gr.Column(scale=3):
                        # 搜索区域
                        image_query = gr.Textbox(
                            label="搜索图片",
                            placeholder="描述你想找的图片，如：'海边日落'",
                            lines=2
                        )
                        with gr.Row():
                            image_search_btn = gr.Button("🔍 搜索", variant="primary")
                            image_top_k = gr.Slider(1, 10, value=3, label="显示数量", scale=2)
                        
                        # 图片预览区域
                        image_gallery = gr.Gallery(
                            label="图片预览",
                            columns=3,
                            height="300px",
                            object_fit="cover"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 上传新图片")
                        image_upload = gr.File(
                            label="选择图片文件",
                            file_types=[".png", ".jpg", ".jpeg", ".webp"],
                            file_count="multiple"
                        )
                        image_upload_btn = gr.Button("📤 上传图片到数据库", variant="secondary")
                        image_upload_result = gr.Textbox(
                            label="上传结果",
                            lines=8,
                            interactive=False
                        )
                
                image_output = gr.Markdown(label="图片信息")
                
                # 示例查询按钮
                gr.Markdown("### 🎯 示例查询")
                with gr.Row():
                    example_queries = ["自然风景", "城市建筑", "动物", "食物", "科技"]
                    for query in example_queries:
                        gr.Button(
                            query,
                            size="sm"
                        ).click(
                            lambda q=query: q,
                            outputs=image_query
                        ).then(
                            assistant.search_images,
                            inputs=[image_query, image_top_k],
                            outputs=[image_output, image_gallery]
                        )
            
            # Tab 3: 数据库状态
            with gr.TabItem("📊 系统状态"):
                with gr.Row():
                    with gr.Column(scale=2):
                        stats_btn = gr.Button("🔄 刷新状态", variant="secondary", size="lg")
                        stats_output = gr.Markdown()
                        
                        def update_stats():
                            return assistant.get_database_stats()
                        
                        # 初始加载状态
                        stats_output.value = update_stats()
                        
                        stats_btn.click(
                            update_stats,
                            outputs=stats_output
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 使用说明")
                        gr.Markdown("""
                        **论文管理:**
                        - 支持PDF文件上传和自动分类
                        - 基于语义搜索论文内容
                        
                        **图片管理:**
                        - 支持多种图片格式上传
                        - 使用CLIP模型进行图文搜索
                        - 上传后即可通过文字描述搜索
                        
                        **系统要求:**
                        - 确保有足够磁盘空间
                        - 首次使用需要初始化模型
                        - 支持批量上传
                        """)
        
        # 绑定事件 - 论文管理
        paper_search_btn.click(
            assistant.search_papers,
            inputs=[paper_query, paper_top_k],
            outputs=paper_output
        )
        
        paper_upload_btn.click(
            assistant.add_paper,
            inputs=paper_upload,
            outputs=paper_output
        )
        
        # 绑定事件 - 图片管理
        image_search_btn.click(
            assistant.search_images,
            inputs=[image_query, image_top_k],
            outputs=[image_output, image_gallery]
        )
        
        image_upload_btn.click(
            assistant.add_images,
            inputs=image_upload,
            outputs=image_upload_result
        ).then(
            update_stats,  # 上传后刷新状态
            outputs=stats_output
        )
        
        # 回车键触发搜索
        paper_query.submit(
            assistant.search_papers,
            inputs=[paper_query, paper_top_k],
            outputs=paper_output
        )
        
        image_query.submit(
            assistant.search_images,
            inputs=[image_query, image_top_k],
            outputs=[image_output, image_gallery]
        )
    
    return demo

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧠 本地AI智能助手 Web应用")
    print("="*50)
    print("访问: http://localhost:7860")
    print("功能: 论文管理 | 图片搜索 | 智能分类")
    print("按 Ctrl+C 停止")
    print("="*50 + "\n")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设置为True可生成公共链接
        debug=True,
        show_error=True
    )

    