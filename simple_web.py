# simple_web.py - 最小化可工作的版本
import gradio as gr
import sys
from pathlib import Path

# 添加项目路径
sys.path.append('.')
from modules.image_processor import ImageProcessor
from modules.vector_db import VectorDB

print("正在初始化...")

# 全局初始化（避免重复初始化）
try:
    image_processor = ImageProcessor()
    vector_db = VectorDB()
    print("✅ 初始化成功")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    raise

def search_images_simple(query, top_k=3):
    """最简单的图片搜索"""
    try:
        if not query or query.strip() == "":
            return "请输入搜索内容", []
        
        print(f"[搜索] 查询: '{query}'")
        
        # 编码查询
        query_emb = image_processor.encode_text_for_image_search(query)
        print(f"[搜索] 编码完成")
        
        # 搜索
        results = vector_db.search_images(query_emb, k=top_k)
        print(f"[搜索] 找到 {len(results)} 个结果")
        
        if not results:
            return "没有找到相关图片", []
        
        # 收集有效的图片路径
        image_paths = []
        output_text = f"找到 {len(results)} 张相关图片:\n\n"
        
        for i, (score, img_path, metadata) in enumerate(results, 1):
            # 尝试多种方式获取路径
            paths_to_try = [
                img_path,
                metadata.get('path') if metadata else None,
                metadata.get('source') if metadata else None
            ]
            
            found = False
            for path in paths_to_try:
                if path and Path(path).exists():
                    image_paths.append(path)
                    filename = Path(path).name
                    output_text += f"{i}. **{filename}** (相似度: {score:.3f})\n"
                    found = True
                    break
            
            if not found:
                output_text += f"{i}. 图片文件不存在\n"
        
        if not image_paths:
            return "没有找到可显示的图片文件", []
        
        return output_text, image_paths
        
    except Exception as e:
        error_msg = f"搜索出错: {str(e)}"
        print(f"[错误] {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg, []

def add_image_simple(files):
    """添加图片"""
    try:
        if not files:
            return "请选择图片文件"
        
        success_count = 0
        for file_info in files:
            file_path = file_info.name
            print(f"[上传] 处理: {Path(file_path).name}")
            
            # 编码图片
            embedding = image_processor.encode_image(file_path)
            
            # 添加到数据库
            metadata = {
                "filename": Path(file_path).name,
                "path": file_path,
                "size": Path(file_path).stat().st_size
            }
            
            if vector_db.add_image(file_path, embedding, metadata):
                success_count += 1
        
        return f"✅ 成功添加 {success_count}/{len(files)} 张图片"
        
    except Exception as e:
        return f"❌ 添加失败: {str(e)}"

# 创建界面
with gr.Blocks(title="AI图片搜索", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ AI图片搜索")
    gr.Markdown("通过文字描述搜索本地图片库")
    
    with gr.Row():
        with gr.Column(scale=2):
            # 搜索区域
            query_input = gr.Textbox(
                label="搜索图片",
                placeholder="例如：海边日落、城市夜景、猫咪",
                lines=2
            )
            
            with gr.Row():
                search_btn = gr.Button("🔍 搜索", variant="primary", scale=1)
                top_k_slider = gr.Slider(1, 10, value=3, label="显示数量", scale=2)
            
            # 结果显示
            result_text = gr.Markdown(label="搜索结果")
        
        with gr.Column(scale=1):
            # 图片预览
            image_gallery = gr.Gallery(
                label="图片预览",
                columns=2,
                height="400px",
                object_fit="cover"
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        # 上传区域
        with gr.Column(scale=2):
            file_upload = gr.File(
                label="上传图片",
                file_types=[".png", ".jpg", ".jpeg"],
                file_count="multiple"
            )
            upload_btn = gr.Button("📤 上传图片到数据库", variant="secondary")
            upload_result = gr.Textbox(label="上传结果", interactive=False)
        
        with gr.Column(scale=1):
            # 状态信息
            status_btn = gr.Button("🔄 刷新状态", variant="secondary")
            status_output = gr.Markdown()
            
            def get_status():
                count = vector_db.image_collection.count()
                return f"**数据库状态**\n\n📊 图片数量: {count} 张"
            
            status_output.value = get_status()
            status_btn.click(get_status, outputs=status_output)
    
    # 绑定事件
    search_btn.click(
        search_images_simple,
        inputs=[query_input, top_k_slider],
        outputs=[result_text, image_gallery]
    )
    
    query_input.submit(
        search_images_simple,
        inputs=[query_input, top_k_slider],
        outputs=[result_text, image_gallery]
    )
    
    upload_btn.click(
        add_image_simple,
        inputs=file_upload,
        outputs=upload_result
    ).then(
        get_status,  # 上传后刷新状态
        outputs=status_output
    )
    
    # 示例查询按钮
    gr.Markdown("### 🎯 示例查询")
    with gr.Row():
        example_queries = ["海边日落", "城市夜景", "猫咪", "建筑", "食物"]
        for query in example_queries:
            gr.Button(
                query,
                size="sm"
            ).click(
                lambda q=query: q,  # 设置查询文本
                outputs=query_input
            ).then(
                search_images_simple,
                inputs=[query_input, top_k_slider],
                outputs=[result_text, image_gallery]
            )

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌐 AI图片搜索 Web应用")
    print("="*50)
    print("访问: http://localhost:7860")
    print("按 Ctrl+C 停止")
    print("="*50 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )