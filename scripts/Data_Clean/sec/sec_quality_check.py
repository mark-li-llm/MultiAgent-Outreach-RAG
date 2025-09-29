#!/usr/bin/env python3
"""
SEC数据质量检测脚本
专门检查SEC文档的重复性、差异性和完整性
"""

import json
import os
import re
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
from pathlib import Path
import difflib
from datetime import datetime


def load_sec_documents() -> Dict[str, Dict]:
    """加载所有SEC文档的元数据和内容"""
    sec_dir = Path("data/raw/sec")
    docs = {}
    
    for meta_file in sec_dir.glob("*.meta.json"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            doc_id = meta.get('doc_id')
            if not doc_id:
                continue
                
            # 查找对应的内容文件
            content_file = None
            for ext in ['.raw.html', '.pdf', '.json']:
                candidate = sec_dir / f"{doc_id}{ext}"
                if candidate.exists():
                    content_file = candidate
                    break
            
            if content_file:
                # 读取内容
                try:
                    if content_file.suffix == '.pdf':
                        content = f"[PDF文件: {content_file.stat().st_size} bytes]"
                    else:
                        with open(content_file, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                except Exception as e:
                    content = f"[读取失败: {str(e)}]"
                
                docs[doc_id] = {
                    'meta': meta,
                    'content': content,
                    'content_file': str(content_file),
                    'file_size': content_file.stat().st_size if content_file.exists() else 0
                }
            else:
                print(f"警告: 找不到 {doc_id} 的内容文件")
                
        except Exception as e:
            print(f"加载 {meta_file} 失败: {str(e)}")
    
    return docs


def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度 (0-1)"""
    if not text1 or not text2:
        return 0.0
    
    # 清理文本
    def clean_text(text):
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', ' ', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 转小写
        return text.lower().strip()
    
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    if not clean1 or not clean2:
        return 0.0
    
    # 使用SequenceMatcher计算相似度
    matcher = difflib.SequenceMatcher(None, clean1, clean2)
    return matcher.ratio()


def extract_key_sections(content: str, doctype: str) -> Dict[str, str]:
    """提取文档的关键部分"""
    sections = {}
    
    if doctype == '10-K':
        # 查找10-K的关键部分
        patterns = {
            'business_overview': r'(?i)(item\s*1[^\d].*?business|overview.*?business)',
            'risk_factors': r'(?i)(item\s*1a.*?risk\s*factors|risk\s*factors)',
            'financial_statements': r'(?i)(item\s*8.*?financial\s*statements|consolidated\s*statements)',
            'revenue': r'(?i)(revenue|total.*?revenue|net.*?revenue)',
        }
    elif doctype == '10-Q':
        patterns = {
            'financial_statements': r'(?i)(consolidated\s*statements|financial\s*statements)',
            'revenue': r'(?i)(revenue|total.*?revenue|net.*?revenue)',
            'quarterly_results': r'(?i)(quarterly.*?results|three.*?months)',
        }
    elif doctype == '8-K':
        patterns = {
            'current_report': r'(?i)(current\s*report|form\s*8-k)',
            'item_disclosure': r'(?i)(item\s*\d+|disclosure)',
        }
    else:
        patterns = {
            'content_sample': r'.{0,500}'  # 前500字符
        }
    
    for section_name, pattern in patterns.items():
        matches = re.search(pattern, content, re.DOTALL)
        if matches:
            # 取匹配后的一定长度内容
            start = matches.start()
            sections[section_name] = content[start:start+1000]
    
    return sections


def check_document_completeness(doc: Dict) -> Dict[str, any]:
    """检查文档完整性"""
    meta = doc['meta']
    content = doc['content']
    
    issues = []
    stats = {}
    
    # 基础检查
    content_length = len(content)
    stats['content_length'] = content_length
    
    if content_length < 100:
        issues.append("内容过短，可能不完整")
    
    if content.startswith('[') and '失败' in content:
        issues.append("内容读取失败")
    
    # 检查是否有截断标志
    truncation_signs = ['...', '(continued)', '[truncated]', '省略', '截断']
    for sign in truncation_signs:
        if sign in content.lower():
            issues.append(f"发现可能的截断标志: {sign}")
    
    # 检查HTML格式问题
    if '<html' in content.lower():
        if not content.strip().endswith('>') and not content.strip().endswith('</html>'):
            issues.append("HTML文档可能不完整")
    
    # 元数据一致性检查
    expected_size = meta.get('content_length', 0)
    actual_size = doc['file_size']
    
    if expected_size > 0 and abs(expected_size - actual_size) > 1000:
        issues.append(f"文件大小不匹配: 期望{expected_size}, 实际{actual_size}")
    
    # 文档类型特定检查
    doctype = meta.get('doctype', '')
    if doctype in ['10-K', '10-Q']:
        required_elements = ['financial', 'statements', 'revenue']
        missing_elements = []
        for element in required_elements:
            if element.lower() not in content.lower():
                missing_elements.append(element)
        if missing_elements:
            issues.append(f"缺少关键财务元素: {missing_elements}")
    
    return {
        'issues': issues,
        'stats': stats,
        'is_complete': len(issues) == 0
    }


def analyze_8k_differences(docs: Dict[str, Dict]) -> Dict:
    """分析8-K文档之间的差异"""
    eightk_docs = {doc_id: doc for doc_id, doc in docs.items() 
                   if doc['meta'].get('doctype') == '8-K'}
    
    if len(eightk_docs) < 2:
        return {"message": "8-K文档数量不足，无法进行差异分析"}
    
    analysis = {
        'doc_count': len(eightk_docs),
        'documents': {},
        'similarities': {},
        'unique_content': {}
    }
    
    # 分析每个8-K文档
    for doc_id, doc in eightk_docs.items():
        meta = doc['meta']
        content = doc['content']
        
        analysis['documents'][doc_id] = {
            'date': meta.get('publish_date', meta.get('visible_date', 'unknown')),
            'title': meta.get('visible_title', meta.get('headline', 'unknown')),
            'word_count': len(content.split()),
            'char_count': len(content),
            'key_topics': extract_topics_from_8k(content)
        }
    
    # 计算两两相似度
    doc_ids = list(eightk_docs.keys())
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            doc1_id, doc2_id = doc_ids[i], doc_ids[j]
            similarity = calculate_text_similarity(
                eightk_docs[doc1_id]['content'],
                eightk_docs[doc2_id]['content']
            )
            analysis['similarities'][f"{doc1_id} vs {doc2_id}"] = similarity
    
    return analysis


def extract_topics_from_8k(content: str) -> List[str]:
    """从8-K内容中提取主要主题"""
    topics = []
    
    # 常见的8-K主题关键词
    topic_patterns = {
        'earnings': r'(?i)(earnings|financial\s*results|quarterly\s*results)',
        'acquisition': r'(?i)(acquisition|merger|acquire|purchase)',
        'executive_changes': r'(?i)(executive|officer|director|appointment|resignation)',
        'agreements': r'(?i)(agreement|contract|partnership)',
        'securities': r'(?i)(securities|shares|stock|equity)',
        'litigation': r'(?i)(litigation|lawsuit|legal\s*proceedings)',
        'regulatory': r'(?i)(regulatory|sec|compliance)',
    }
    
    for topic, pattern in topic_patterns.items():
        if re.search(pattern, content):
            topics.append(topic)
    
    return topics


def compare_10k_and_ars_pdf(docs: Dict[str, Dict]) -> Dict:
    """比较10-K和ars_pdf的重复度"""
    tenk_doc = None
    ars_pdf_doc = None
    
    for doc_id, doc in docs.items():
        doctype = doc['meta'].get('doctype', '')
        if doctype == '10-K':
            tenk_doc = (doc_id, doc)
        elif doctype == 'ars_pdf':
            ars_pdf_doc = (doc_id, doc)
    
    if not tenk_doc or not ars_pdf_doc:
        return {"error": "找不到10-K或ars_pdf文档"}
    
    tenk_id, tenk = tenk_doc
    ars_id, ars = ars_pdf_doc
    
    # 计算整体相似度
    overall_similarity = calculate_text_similarity(tenk['content'], ars['content'])
    
    # 提取关键部分进行比较
    tenk_sections = extract_key_sections(tenk['content'], '10-K')
    ars_sections = extract_key_sections(ars['content'], 'ars_pdf')
    
    section_similarities = {}
    for section in set(tenk_sections.keys()) | set(ars_sections.keys()):
        if section in tenk_sections and section in ars_sections:
            sim = calculate_text_similarity(tenk_sections[section], ars_sections[section])
            section_similarities[section] = sim
    
    return {
        'tenk_doc': tenk_id,
        'ars_pdf_doc': ars_id,
        'overall_similarity': overall_similarity,
        'section_similarities': section_similarities,
        'tenk_size': len(tenk['content']),
        'ars_pdf_size': len(ars['content']),
        'size_ratio': len(ars['content']) / len(tenk['content']) if len(tenk['content']) > 0 else 0,
        'tenk_sections_found': list(tenk_sections.keys()),
        'ars_pdf_sections_found': list(ars_sections.keys())
    }


def generate_quality_report(docs: Dict[str, Dict]) -> Dict:
    """生成完整的质量报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_documents': len(docs),
        'document_types': Counter(doc['meta'].get('doctype') for doc in docs.values()),
        'completeness_check': {},
        'similarity_analysis': {},
        'file_integrity': {},
        'summary': {}
    }
    
    # 完整性检查
    for doc_id, doc in docs.items():
        completeness = check_document_completeness(doc)
        report['completeness_check'][doc_id] = completeness
    
    # 10-K vs ars_pdf 比较
    comparison = compare_10k_and_ars_pdf(docs)
    report['similarity_analysis']['10k_vs_ars_pdf'] = comparison
    
    # 8-K差异分析
    eightk_analysis = analyze_8k_differences(docs)
    report['similarity_analysis']['8k_differences'] = eightk_analysis
    
    # 文件完整性统计
    total_issues = 0
    complete_docs = 0
    
    for doc_id, completeness in report['completeness_check'].items():
        if completeness['is_complete']:
            complete_docs += 1
        else:
            total_issues += len(completeness['issues'])
    
    report['summary'] = {
        'complete_documents': complete_docs,
        'total_issues': total_issues,
        'completion_rate': complete_docs / len(docs) if docs else 0,
        'avg_issues_per_doc': total_issues / len(docs) if docs else 0
    }
    
    return report


def main():
    print("🔍 SEC数据质量检测开始...")
    
    # 加载文档
    print("📖 加载SEC文档...")
    docs = load_sec_documents()
    print(f"✅ 加载了 {len(docs)} 个文档")
    
    # 生成质量报告
    print("📊 生成质量报告...")
    report = generate_quality_report(docs)
    
    # 保存报告
    output_file = "reports/sec_quality_analysis.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "="*60)
    print("📋 SEC数据质量检测报告摘要")
    print("="*60)
    
    print(f"📄 总文档数: {report['total_documents']}")
    print(f"📊 文档类型分布: {dict(report['document_types'])}")
    print(f"✅ 完整文档数: {report['summary']['complete_documents']}")
    print(f"⚠️  总问题数: {report['summary']['total_issues']}")
    print(f"📈 完整率: {report['summary']['completion_rate']:.1%}")
    
    # 10-K vs ars_pdf 相似度
    if '10k_vs_ars_pdf' in report['similarity_analysis']:
        comp = report['similarity_analysis']['10k_vs_ars_pdf']
        if 'overall_similarity' in comp:
            print(f"🔄 10-K vs ars_pdf 相似度: {comp['overall_similarity']:.1%}")
            print(f"📏 大小比例 (ars_pdf/10-K): {comp.get('size_ratio', 0):.2f}")
    
    # 8-K 差异性
    if '8k_differences' in report['similarity_analysis']:
        eightk = report['similarity_analysis']['8k_differences']
        if 'similarities' in eightk:
            similarities = list(eightk['similarities'].values())
            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                print(f"📋 8-K文档间平均相似度: {avg_sim:.1%}")
    
    print(f"\n📁 详细报告已保存至: {output_file}")
    print("🎉 质量检测完成！")


if __name__ == "__main__":
    main()
