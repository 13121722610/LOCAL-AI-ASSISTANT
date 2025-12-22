# main.py
#!/usr/bin/env python3
"""
Local Multimodal AI Assistant - 完整修复版
包含所有图像功能实现
"""

import argparse
import sys
from pathlib import Path
from typing import List

from modules.text_processor import TextProcessor
from modules.image_processor import ImageProcessor
from modules.vector_db import VectorDB
from modules.classifier import Classifier
from modules.file_utils import FileUtils
import config

def setup_argparse() -> argparse.ArgumentParser:
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(
        description="Local Multimodal AI Assistant - Manage your papers and images intelligently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py add_paper "path/to/paper.pdf"
  python main.py add_paper "path/to/paper.pdf" --topics "CV,NLP"
  python main.py search_paper "transformer architecture"
  python main.py search_image "sunset by the sea"
  python main.py organize "path/to/papers_folder"
  python main.py list_papers
  python main.py list_images
  python main.py add_image "path/to/image.jpg"
  python main.py add_images "path/to/images_folder"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # 添加论文命令
    add_paper = subparsers.add_parser("add_paper", help="Add and classify a paper")
    add_paper.add_argument("path", help="Path to PDF file")
    add_paper.add_argument("--topics", help="Comma-separated topics for classification")
    
    # 搜索论文命令
    search_paper = subparsers.add_parser("search_paper", help="Search papers semantically")
    search_paper.add_argument("query", help="Search query")
    search_paper.add_argument("-k", type=int, default=5, help="Number of results")
    
    # 搜索图片命令
    search_image = subparsers.add_parser("search_image", help="Search images by text")
    search_image.add_argument("query", help="Image search query")
    search_image.add_argument("-k", type=int, default=5, help="Number of results")
    
    # 整理文件夹命令
    organize = subparsers.add_parser("organize", help="Organize all papers in folder")
    organize.add_argument("folder", help="Folder to organize")
    organize.add_argument("--topics", help="Comma-separated topics")
    
    # 添加图片命令
    add_image = subparsers.add_parser("add_image", help="Add an image to database")
    add_image.add_argument("path", help="Path to image file")
    
    # 批量添加图片
    add_images = subparsers.add_parser("add_images", help="Add all images in folder")
    add_images.add_argument("folder", help="Folder containing images")
    
    # 列出所有论文
    list_papers = subparsers.add_parser("list_papers", help="List all indexed papers")
    
    # 列出所有图片
    list_images = subparsers.add_parser("list_images", help="List all indexed images")
    
    # 清除数据库
    clear_db = subparsers.add_parser("clear_db", help="Clear vector database")
    clear_db.add_argument("--confirm", action="store_true", help="Confirm deletion")
    
    return parser

def handle_add_paper(args, text_processor: TextProcessor, 
                     vector_db: VectorDB, classifier: Classifier):
    """处理添加论文命令"""
    pdf_path = Path(args.path)
    if not pdf_path.exists():
        print(f"❌ 错误：文件不存在: {args.path}")
        return
    
    print(f"📄 处理论文: {pdf_path.name}")
    
    # 提取文本和生成向量
    chunks, embeddings = text_processor.process_pdf(str(pdf_path))
    
    if not chunks:
        print("❌ 错误：无法从PDF提取文本")
        return
    
    # 分类
    topics = args.topics.split(",") if args.topics else None
    topic = classifier.classify_pdf(str(pdf_path), topics)
    print(f"🏷️  分类为: {topic}")
    
    # 整理文件
    target_path = FileUtils.organize_file(str(pdf_path), topic)
    
    # 添加到数据库
    metadata = {
        "title": pdf_path.stem,
        "topic": topic,
        "original_path": str(pdf_path),
        "organized_path": target_path
    }
    
    success = vector_db.add_paper(target_path, chunks, embeddings, metadata)
    if success:
        print(f"✅ 论文添加成功，分类: {topic}")
    else:
        print("❌ 添加到数据库失败")

def handle_search_paper(args, text_processor: TextProcessor, vector_db: VectorDB):
    """处理搜索论文命令"""
    print(f"🔍 搜索: '{args.query}'")
    
    # 编码查询文本
    query_embedding = text_processor.encode_text(args.query)
    
    # 在数据库中搜索
    results = vector_db.search_text(query_embedding, k=args.k)
    
    if not results:
        print("没有找到结果")
        return
    
    print(f"\n找到 {len(results)} 个结果:\n")
    for i, (score, document, metadata) in enumerate(results, 1):
        source = metadata.get('source', 'Unknown')
        topic = metadata.get('topic', 'Unknown')
        print(f"{i}. [{topic}] {Path(source).name} (相似度: {score:.3f})")
        print(f"   来源: {source}")
        print(f"   预览: {document[:150]}...\n")

def handle_search_image(args, image_processor: ImageProcessor, vector_db: VectorDB):
    """处理搜索图片命令"""
    print(f"🔍 搜索图片: '{args.query}'")
    
    # 编码查询文本
    query_embedding = image_processor.encode_text_for_image_search(args.query)
    
    # 在数据库中搜索
    results = vector_db.search_images(query_embedding, k=args.k)
    
    if not results:
        print("没有找到相关图片")
        return
    
    print(f"\n找到 {len(results)} 张相关图片:\n")
    for i, (score, image_path, metadata) in enumerate(results, 1):
        filename = Path(image_path).name if image_path else "Unknown"
        print(f"{i}. {filename} (相似度: {score:.3f})")
        if image_path:
            print(f"   路径: {image_path}")
        if metadata and 'format' in metadata:
            print(f"   格式: {metadata['format']}")
        print()

def handle_add_image(args, image_processor: ImageProcessor, vector_db: VectorDB):
    """处理添加单张图片命令"""
    image_path = Path(args.path)
    if not image_path.exists():
        print(f"❌ 错误：图片不存在: {args.path}")
        return
    
    print(f"📸 添加图片: {image_path.name}")
    
    try:
        # 编码图片
        embedding = image_processor.encode_image(str(image_path))
        print(f"   编码完成，向量维度: {embedding.shape}")
        
        # 添加到数据库
        metadata = {
            "filename": image_path.name,
            "path": str(image_path),
            "size": f"{image_path.stat().st_size} bytes",
            "format": image_path.suffix[1:].upper()
        }
        
        success = vector_db.add_image(str(image_path), embedding, metadata)
        if success:
            print(f"✅ 图片添加成功: {image_path.name}")
        else:
            print("❌ 添加到数据库失败")
            
    except Exception as e:
        print(f"❌ 处理图片时出错: {e}")
        import traceback
        traceback.print_exc()

def handle_add_images(args, image_processor: ImageProcessor, vector_db: VectorDB):
    """处理批量添加图片命令"""
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"❌ 错误：文件夹不存在: {args.folder}")
        return
    
    # 获取所有图片文件
    image_files = FileUtils.get_all_images(str(folder_path))
    
    if not image_files:
        print("没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，正在添加...\n")
    
    added_count = 0
    for img_file in image_files:
        try:
            img_path = Path(img_file)
            print(f"处理: {img_path.name}", end=" ")
            
            embedding = image_processor.encode_image(str(img_path))
            
            metadata = {
                "filename": img_path.name,
                "path": str(img_path),
                "size": f"{img_path.stat().st_size} bytes",
                "format": img_path.suffix[1:].upper()
            }
            
            if vector_db.add_image(str(img_path), embedding, metadata):
                added_count += 1
                print("✅")
            else:
                print("❌")
                
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    print(f"\n📊 完成: 成功添加 {added_count}/{len(image_files)} 张图片")

def handle_organize(args, classifier: Classifier):
    """处理整理文件夹命令"""
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"❌ 错误：文件夹不存在: {args.folder}")
        return
    
    # 获取所有PDF文件
    pdf_files = FileUtils.get_all_pdfs(str(folder_path))
    
    if not pdf_files:
        print("没有找到PDF文件")
        return
    
    print(f"找到 {len(pdf_files)} 个PDF文件，正在整理...\n")
    
    topics = args.topics.split(",") if args.topics else None
    
    for pdf_file in pdf_files:
        try:
            # 分类
            topic = classifier.classify_pdf(pdf_file, topics)
            
            # 整理文件
            target_path = FileUtils.organize_file(pdf_file, topic)
            
            print(f"✅ {Path(pdf_file).name} → {topic}/")
        except Exception as e:
            print(f"❌ 处理失败 {pdf_file}: {e}")

def handle_list_papers(vector_db: VectorDB):
    """处理列出所有论文命令"""
    papers = vector_db.get_all_papers()
    
    if not papers:
        print("数据库中没有论文")
        return
    
    # 按主题分组
    papers_by_topic = {}
    for paper in papers:
        topic = Path(paper).parent.name
        if topic not in papers_by_topic:
            papers_by_topic[topic] = []
        papers_by_topic[topic].append(paper)
    
    print(f"\n📚 已索引论文 ({len(papers)} 篇):\n")
    
    total_count = 0
    for topic in sorted(papers_by_topic.keys()):
        topic_papers = papers_by_topic[topic]
        print(f"【{topic}】({len(topic_papers)} 篇):")
        
        for i, paper in enumerate(sorted(topic_papers), 1):
            print(f"  {total_count + i}. {Path(paper).name}")
            print(f"      路径: {paper}")
        
        total_count += len(topic_papers)
        print()

def handle_list_images(vector_db: VectorDB):
    """处理列出所有图片命令"""
    try:
        # 直接查询数据库
        results = vector_db.image_collection.get()
        if results['metadatas']:
            images = []
            for metadata in results['metadatas']:
                if 'path' in metadata:
                    images.append(metadata['path'])
                elif 'source' in metadata:
                    images.append(metadata['source'])
            
            unique_images = list(set(images))
            
            print(f"\n📸 已索引图片 ({len(unique_images)} 张):\n")
            for i, img_path in enumerate(sorted(unique_images), 1):
                img_name = Path(img_path).name
                print(f"{i}. {img_name}")
                print(f"   路径: {img_path}")
                
                # 显示元数据
                for metadata in results['metadatas']:
                    path = metadata.get('path') or metadata.get('source')
                    if path == img_path:
                        if 'size' in metadata:
                            print(f"   大小: {metadata['size']}")
                        if 'format' in metadata:
                            print(f"   格式: {metadata['format']}")
                        break
                print()
        else:
            print("数据库中没有图片")
    except Exception as e:
        print(f"❌ 列出图片时出错: {e}")

def main():
    """主函数"""
    parser = setup_argparse()
    
    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # 初始化组件
    print("=" * 60)
    print("Local Multimodal AI Assistant")
    print("=" * 60)
    
    try:
        text_processor = TextProcessor()
        image_processor = ImageProcessor()
        vector_db = VectorDB()
        classifier = Classifier()
    except Exception as e:
        print(f"❌ 初始化组件失败: {e}")
        sys.exit(1)
    
    # 根据命令执行相应操作
    if args.command == "add_paper":
        handle_add_paper(args, text_processor, vector_db, classifier)
    
    elif args.command == "search_paper":
        handle_search_paper(args, text_processor, vector_db)
    
    elif args.command == "search_image":
        handle_search_image(args, image_processor, vector_db)
    
    elif args.command == "add_image":
        handle_add_image(args, image_processor, vector_db)
    
    elif args.command == "add_images":
        handle_add_images(args, image_processor, vector_db)
    
    elif args.command == "organize":
        handle_organize(args, classifier)
    
    elif args.command == "list_papers":
        handle_list_papers(vector_db)
    
    elif args.command == "list_images":
        handle_list_images(vector_db)
    
    elif args.command == "clear_db":
        if args.confirm:
            print("正在清除数据库...")
            if hasattr(vector_db, 'clear_database'):
                if vector_db.clear_database():
                    print("✅ 数据库已清空")
                else:
                    print("❌ 清空数据库失败")
            else:
                print("❌ clear_database 方法未实现")
        else:
            print("⚠️  警告：这将删除所有索引数据！")
            print("使用 --confirm 参数确认操作")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
