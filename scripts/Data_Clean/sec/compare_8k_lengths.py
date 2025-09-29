#!/usr/bin/env python3
"""
比较三个8-K文档的长度和相似度
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import difflib


def get_8k_files() -> List[Path]:
    """获取所有8-K文档文件"""
    sec_dir = Path("data/raw/sec")
    return sorted(sec_dir.glob("*8-K*.raw.html"))


def analyze_content_length(content: str) -> Dict:
    """分析内容长度的各种指标"""
    return {
        'total_chars': len(content),
        'total_lines': len(content.splitlines()),
        'words': len(content.split()),
        'non_whitespace_chars': len(re.sub(r'\s+', '', content)),
        'substantive_text': len(re.sub(r'<[^>]+>|\s+', ' ', content).strip()),
    }


def extract_clean_text(content: str) -> str:
    """提取清洁的文本内容，去除HTML标签和多余空白"""
    # 去除HTML标签
    clean = re.sub(r'<[^>]+>', '', content)
    # 去除多余空白
    clean = re.sub(r'\s+', ' ', clean)
    # 去除特殊字符和制表符对齐
    clean = re.sub(r'[_\-=]{3,}', '', clean)
    return clean.strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度"""
    # 使用difflib计算序列匹配度
    matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
    return matcher.ratio()


def analyze_8k_structure(content: str) -> Dict:
    """分析8-K文档的结构特征"""
    structure = {
        'form_8k_mentions': len(re.findall(r'FORM\s+8-K', content, re.IGNORECASE)),
        'items_found': len(re.findall(r'Item\s+\d+\.\d+', content, re.IGNORECASE)),
        'sections_found': len(re.findall(r'Section\s+\d+', content, re.IGNORECASE)),
        'exhibits_found': len(re.findall(r'Exhibit\s+\d+', content, re.IGNORECASE)),
        'has_signature': bool(re.search(r'signature', content, re.IGNORECASE)),
        'has_date': bool(re.search(r'202[45]', content)),
        'has_salesforce': bool(re.search(r'salesforce', content, re.IGNORECASE)),
    }
    return structure


def main():
    print("📊 8-K文档长度和相似度分析")
    print("=" * 50)
    
    files = get_8k_files()
    if len(files) != 3:
        print(f"❌ 预期3个8-K文档，实际找到 {len(files)} 个")
        return
    
    docs = {}
    clean_texts = {}
    
    # 分析每个文档
    for file_path in files:
        doc_name = file_path.stem.replace('.raw', '')
        
        # 提取日期作为简短标识
        date_match = re.search(r'202[45]-\d{2}-\d{2}', doc_name)
        short_name = date_match.group() if date_match else doc_name[-10:]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        length_analysis = analyze_content_length(content)
        structure_analysis = analyze_8k_structure(content)
        clean_text = extract_clean_text(content)
        
        docs[short_name] = {
            'file_path': str(file_path),
            'content': content,
            'length_analysis': length_analysis,
            'structure_analysis': structure_analysis,
            'clean_text': clean_text
        }
        clean_texts[short_name] = clean_text
        
        print(f"\n📄 {short_name}")
        print(f"   📏 总字符数: {length_analysis['total_chars']:,}")
        print(f"   📝 总行数: {length_analysis['total_lines']:,}")
        print(f"   🔤 单词数: {length_analysis['words']:,}")
        print(f"   📋 实质文本: {length_analysis['substantive_text']:,} 字符")
        print(f"   🏗️ Items: {structure_analysis['items_found']}, Sections: {structure_analysis['sections_found']}")
        print(f"   📎 Exhibits: {structure_analysis['exhibits_found']}")
        print(f"   ✍️ 有签名: {'✅' if structure_analysis['has_signature'] else '❌'}")
    
    # 相似度分析
    print(f"\n" + "=" * 50)
    print("🔍 相似度分析")
    print("=" * 50)
    
    doc_names = list(docs.keys())
    similarities = {}
    
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            name1, name2 = doc_names[i], doc_names[j]
            similarity = calculate_similarity(clean_texts[name1], clean_texts[name2])
            similarities[f"{name1} vs {name2}"] = similarity
            print(f"📊 {name1} vs {name2}: {similarity:.1%}")
    
    # 长度对比
    print(f"\n" + "=" * 50)
    print("📏 长度对比分析")
    print("=" * 50)
    
    lengths = [(name, docs[name]['length_analysis']['total_chars']) for name in doc_names]
    lengths.sort(key=lambda x: x[1])
    
    min_length = lengths[0][1]
    max_length = lengths[-1][1]
    avg_length = sum(l[1] for l in lengths) / len(lengths)
    
    print(f"📊 长度统计:")
    print(f"   最短: {lengths[0][0]} - {min_length:,} 字符")
    print(f"   最长: {lengths[-1][0]} - {max_length:,} 字符")
    print(f"   平均: {avg_length:,.0f} 字符")
    print(f"   差异倍数: {max_length/min_length:.1f}x")
    
    # 判断修复后的2月文档是否合理
    feb_doc = None
    for name in doc_names:
        if '2025-02-26' in name:
            feb_doc = name
            break
    
    if feb_doc:
        feb_length = docs[feb_doc]['length_analysis']['total_chars']
        print(f"\n🎯 2月文档 ({feb_doc}) 分析:")
        print(f"   长度: {feb_length:,} 字符")
        
        if feb_length < avg_length * 0.3:
            print(f"   ⚠️  明显偏短 (< 30% 平均长度)")
        elif feb_length < avg_length * 0.7:
            print(f"   ⚡ 偏短但可接受 (< 70% 平均长度)")
        else:
            print(f"   ✅ 长度合理")
    
    # 保存详细结果
    results = {
        'documents': {name: {
            'length_analysis': docs[name]['length_analysis'],
            'structure_analysis': docs[name]['structure_analysis']
        } for name in doc_names},
        'similarities': similarities,
        'length_stats': {
            'min_length': min_length,
            'max_length': max_length,
            'avg_length': avg_length,
            'length_ratio': max_length/min_length
        }
    }
    
    output_file = "reports/8k_length_similarity_analysis.json"
    Path(output_file).parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 详细结果已保存至: {output_file}")
    print("🎉 分析完成！")


if __name__ == "__main__":
    main()
