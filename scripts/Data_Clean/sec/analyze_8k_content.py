#!/usr/bin/env python3
"""
深度分析8-K文档内容质量和完整性
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def analyze_8k_content(file_path: Path) -> Dict:
    """深度分析单个8-K文档内容"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    analysis = {
        'file_path': str(file_path),
        'total_chars': len(content),
        'content_type': 'unknown',
        'is_xbrl_viewer': False,
        'substantive_content': {},
        'document_structure': {},
        'quality_issues': []
    }
    
    # 检测内容类型
    if 'XBRL Viewer' in content or 'loadViewer' in content:
        analysis['content_type'] = 'xbrl_viewer_redirect'
        analysis['is_xbrl_viewer'] = True
        analysis['quality_issues'].append('这是XBRL查看器重定向页面，不是实际的8-K内容')
        analysis['document_structure'] = {'items_found': [], 'sections_found': [], 'exhibits_found': [], 'has_proper_header': False, 'has_signature': False}
        analysis['substantive_content'] = {'substantive_paragraphs': 0, 'earnings_related': False, 'governance_related': False, 'acquisition_related': False, 'key_dates': [], 'financial_figures': [], 'executive_names': []}
        return analysis
    
    if 'FORM 8-K' in content and 'CURRENT REPORT' in content:
        analysis['content_type'] = 'proper_8k_form'
    
    # 分析文档结构
    structure = analyze_document_structure(content)
    analysis['document_structure'] = structure
    
    # 提取实质性内容
    substantive = extract_substantive_content(content)
    analysis['substantive_content'] = substantive
    
    # 质量检查
    issues = check_content_quality(content, structure, substantive)
    analysis['quality_issues'].extend(issues)
    
    return analysis


def analyze_document_structure(content: str) -> Dict:
    """分析8-K文档结构"""
    structure = {
        'has_proper_header': False,
        'has_signature': False,
        'sections_found': [],
        'items_found': [],
        'exhibits_found': []
    }
    
    # 检查标准8-K头部
    if re.search(r'FORM\s+8-K', content, re.IGNORECASE):
        structure['has_proper_header'] = True
    
    # 检查签名
    if re.search(r'signature', content, re.IGNORECASE):
        structure['has_signature'] = True
    
    # 查找Items
    item_pattern = r'Item\s+(\d+\.\d+)'
    items = re.findall(item_pattern, content, re.IGNORECASE)
    structure['items_found'] = list(set(items))
    
    # 查找Sections
    section_pattern = r'Section\s+(\d+)'
    sections = re.findall(section_pattern, content, re.IGNORECASE)
    structure['sections_found'] = list(set(sections))
    
    # 查找Exhibits
    exhibit_pattern = r'Exhibit\s+(\d+\.?\d*)'
    exhibits = re.findall(exhibit_pattern, content, re.IGNORECASE)
    structure['exhibits_found'] = list(set(exhibits))
    
    return structure


def extract_substantive_content(content: str) -> Dict:
    """提取8-K的实质性内容"""
    substantive = {
        'earnings_related': False,
        'governance_related': False,
        'acquisition_related': False,
        'key_dates': [],
        'financial_figures': [],
        'executive_names': [],
        'substantive_paragraphs': 0
    }
    
    # 检测主题类型
    if re.search(r'(earnings|financial\s+results|quarterly\s+results)', content, re.IGNORECASE):
        substantive['earnings_related'] = True
    
    if re.search(r'(stockholders|directors|proxy|meeting|election)', content, re.IGNORECASE):
        substantive['governance_related'] = True
    
    if re.search(r'(acquisition|merger|purchase|agreement)', content, re.IGNORECASE):
        substantive['acquisition_related'] = True
    
    # 提取日期
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
    dates = re.findall(date_pattern, content)
    substantive['key_dates'] = list(set(dates))
    
    # 提取财务数字（简单版本）
    financial_pattern = r'\$[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand)?'
    figures = re.findall(financial_pattern, content, re.IGNORECASE)
    substantive['financial_figures'] = figures[:10]  # 限制数量
    
    # 提取高管姓名（基于签名和常见模式）
    name_pattern = r'/s/\s*([A-Z\s]+)'
    names = re.findall(name_pattern, content)
    substantive['executive_names'] = [name.strip() for name in names]
    
    # 计算实质性段落（不包括格式化内容）
    paragraphs = content.split('\n')
    substantive_paras = 0
    for para in paragraphs:
        clean_para = re.sub(r'<[^>]+>', '', para).strip()
        # 实质性段落：长度>50字符，包含常见词汇
        if len(clean_para) > 50 and re.search(r'\b(company|report|fiscal|business|stockholder|result)\b', clean_para, re.IGNORECASE):
            substantive_paras += 1
    
    substantive['substantive_paragraphs'] = substantive_paras
    
    return substantive


def check_content_quality(content: str, structure: Dict, substantive: Dict) -> List[str]:
    """检查内容质量问题"""
    issues = []
    
    # 基本完整性检查
    if not structure['has_proper_header']:
        issues.append('缺少标准8-K表单头部')
    
    if not structure['has_signature']:
        issues.append('缺少签名部分')
    
    if not structure['items_found']:
        issues.append('未找到任何Item条目')
    
    # 内容实质性检查
    if substantive['substantive_paragraphs'] < 3:
        issues.append(f'实质性内容段落过少: {substantive["substantive_paragraphs"]}个')
    
    if len(content) < 10000:  # 10KB阈值
        issues.append(f'文档内容过短: {len(content)}字符')
    
    # 特定内容检查
    if not any([substantive['earnings_related'], substantive['governance_related'], substantive['acquisition_related']]):
        issues.append('未识别出明确的业务主题')
    
    if not substantive['key_dates']:
        issues.append('未找到关键日期信息')
    
    return issues


def main():
    print("🔍 8-K文档深度内容分析开始...")
    
    sec_dir = Path("data/raw/sec")
    eightk_files = list(sec_dir.glob("*8-K*.raw.html"))
    
    if not eightk_files:
        print("❌ 未找到8-K文档")
        return
    
    print(f"📄 找到 {len(eightk_files)} 个8-K文档")
    
    results = {}
    
    for file_path in sorted(eightk_files):
        print(f"\n📋 分析: {file_path.name}")
        analysis = analyze_8k_content(file_path)
        
        # 提取doc_id
        doc_id = file_path.stem.replace('.raw', '')
        results[doc_id] = analysis
        
        # 打印关键信息
        print(f"  📊 内容类型: {analysis['content_type']}")
        print(f"  📏 文档大小: {analysis['total_chars']:,} 字符")
        print(f"  🏗️ 结构完整性: Items={len(analysis['document_structure']['items_found'])}, Sections={len(analysis['document_structure']['sections_found'])}")
        print(f"  📝 实质性段落: {analysis['substantive_content']['substantive_paragraphs']}")
        
        if analysis['quality_issues']:
            print(f"  ⚠️ 质量问题 ({len(analysis['quality_issues'])}个):")
            for issue in analysis['quality_issues']:
                print(f"     • {issue}")
        else:
            print(f"  ✅ 未发现质量问题")
    
    # 保存详细分析报告
    output_file = "reports/8k_detailed_analysis.json"
    Path(output_file).parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成总结
    print(f"\n" + "="*60)
    print("📊 8-K文档分析总结")
    print("="*60)
    
    total_docs = len(results)
    proper_8k_count = sum(1 for r in results.values() if r['content_type'] == 'proper_8k_form')
    xbrl_viewer_count = sum(1 for r in results.values() if r['is_xbrl_viewer'])
    
    print(f"📄 总文档数: {total_docs}")
    print(f"✅ 正常8-K文档: {proper_8k_count}")
    print(f"🔄 XBRL查看器重定向: {xbrl_viewer_count}")
    
    # 分析问题分布
    all_issues = []
    for result in results.values():
        all_issues.extend(result['quality_issues'])
    
    if all_issues:
        print(f"\n⚠️ 发现的主要问题:")
        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common():
            print(f"  • {issue}: {count}次")
    
    print(f"\n📁 详细报告已保存至: {output_file}")
    print("🎉 8-K深度分析完成！")


if __name__ == "__main__":
    main()
