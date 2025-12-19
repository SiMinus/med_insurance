#!/usr/bin/env python3
"""
PDF 文本提取脚本 - 使用 Unstructured 库自动处理

功能：
1. 使用 unstructured 库自动判断 PDF 类型
2. 自动决定是否需要 OCR
3. 支持文本型和图像型 PDF
4. 输出为 documents_example.json 格式

用法：
直接修改 main() 函数中的参数配置后运行：
python test/parse_pdfs.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import re


def extract_with_pypdf(pdf_path: Path) -> str:
    """使用 pypdf 提取文本"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  pypdf 失败: {e}")
        return ""


def extract_with_unstructured(pdf_path: Path) -> str:
    """使用 unstructured 提取文本（使用 OCR）"""
    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",  # 高精度模式，使用 OCR
            languages=["chi_sim", "eng"],
        )
        text_parts = [element.text for element in elements if hasattr(element, 'text') and element.text]
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  unstructured 失败: {e}")
        return ""


def is_valid_text(text: str, min_length: int = 100, min_chinese_ratio: float = 0.1) -> bool:
    """
    检查文本是否有效
    
    Args:
        text: 待检查的文本
        min_length: 最小长度
        min_chinese_ratio: 最小中文字符比例
        
    Returns:
        True 表示文本有效，False 表示无效（可能是乱码或太短）
    """
    if not text or len(text.strip()) < min_length:
        return False
        
    # 计算中文字符比例
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    ratio = chinese_chars / len(text)
    
    return ratio >= min_chinese_ratio


def process_pdf(pdf_path: Path) -> Dict:
    """
    处理单个 PDF 文件
    先用 pypdf，如果文本太少或乱码再用 unstructured
    
    Args:
        pdf_path: PDF 文件路径
    
    Returns:
        包含 text 和 title 的字典
    """
    print(f"处理: {pdf_path.name}")
    
    # 先尝试 pypdf
    text = extract_with_pypdf(pdf_path)
    
    # 检查文本质量
    if not is_valid_text(text):
        print(f"  pypdf 提取文本质量不佳（长度={len(text)}），尝试 unstructured...")
        text = extract_with_unstructured(pdf_path)
    
    if not text or len(text.strip()) < 50:
        print(f"  ⚠️  未提取到足够文本")
        return None
    
    # 清理文本
    text = clean_text(text)
    
    # 使用文件名作为 title
    title = pdf_path.stem
    
    print(f"  ✓ 已提取 {len(text)} 字符")
    
    return {
        "text": text,
        "title": title
    }


def clean_text(text: str) -> str:
    """清理提取的文本"""
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊控制字符
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 移除汉字之间的空格
    # 匹配模式：汉字 + 空格 + 汉字
    # 使用 lookbehind 和 lookahead 确保只删除汉字间的空格
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    
    return text.strip()


def find_pdfs(src_dir: Path, recursive: bool = True) -> List[Path]:
    """查找所有 PDF 文件"""
    if recursive:
        return sorted(src_dir.rglob("*.pdf"))
    else:
        return sorted(src_dir.glob("*.pdf"))


def main():
    # 直接在代码中设置参数
    src_dir = "data/raw/2025省市文件_有表格"
    output = "data/processed/medical_docs.json"
    recursive = True
    max_files = None  # 设置为 None 表示处理所有文件，或设置具体数字限制处理数量
    
    # 检查源目录
    src_dir_path = Path(src_dir)
    if not src_dir_path.exists():
        print(f"❌ 源目录不存在: {src_dir_path}")
        sys.exit(1)
    
    # 查找 PDF 文件
    print(f"\n📂 扫描目录: {src_dir_path}")
    pdf_files = find_pdfs(src_dir_path, recursive=recursive)
    
    if not pdf_files:
        print("❌ 未找到 PDF 文件")
        sys.exit(1)
    
    if max_files:
        pdf_files = pdf_files[:max_files]
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件\n")
    
    # 处理所有 PDF
    documents = []
    doc_id = 0
    
    for pdf_path in pdf_files:
        result = process_pdf(pdf_path)
        
        if result:
            result['id'] = doc_id
            documents.append(result)
            doc_id += 1
            print()
        else:
            print()
    
    # 保存结果 (JSON)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
        
    # 保存结果 (CSV)
    import csv
    csv_path = output_path.with_suffix('.csv')
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'title', 'text'])
        for doc in documents:
            writer.writerow([doc.get('id', ''), doc.get('title', ''), doc.get('text', '')])
    
    print(f"\n✅ 完成！")
    print(f"处理了 {len(documents)}/{len(pdf_files)} 个文件")
    print(f"输出文件 (JSON): {output_path}")
    print(f"输出文件 (CSV): {csv_path}")
    print(f"总计: {sum(len(doc['text']) for doc in documents)} 字符")


if __name__ == "__main__":
    main()
