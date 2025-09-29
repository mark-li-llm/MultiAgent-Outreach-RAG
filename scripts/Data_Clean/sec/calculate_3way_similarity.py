#!/usr/bin/env python3
"""
计算三个8-K文档之间的相似度，并人工检查内容合理性
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import difflib
from collections import Counter


def extract_clean_content(content: str) -> str:
    """提取清洁的实质性内容"""
    # 去除HTML标签和XBRL标记
    clean = re.sub(r'<[^>]*>', '', content)
    # 去除XML声明和注释
    clean = re.sub(r'<\?xml[^>]*\?>|<!--[^>]*-->', '', clean)
    # 去除多余空白和格式字符
    clean = re.sub(r'\s+', ' ', clean)
    # 去除制表符对齐和下划线
    clean = re.sub(r'[_\-=]{3,}', '', clean)
    # 去除特殊字符
    clean = re.sub(r'[&#\d;]+', ' ', clean)
    return clean.strip().lower()


def extract_key_content(content: str) -> Dict:
    """提取关键内容特征"""
    clean = extract_clean_content(content)
    
    # 提取Items
    items = re.findall(r'item\s+(\d+\.\d+)', clean)
    
    # 提取日期
    dates = re.findall(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}', clean)
    
    # 提取关键业务词汇
    business_terms = []
    patterns = [
        r'(earnings?|revenue|income|profit|loss)',
        r'(stockholder|shareholder|meeting|vote|voting)',
        r'(director|board|election|appointment)',
        r'(exhibit|attachment|press release)',
        r'(quarterly|annual|fiscal year)',
        r'(results?|performance|operations?)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, clean)
        business_terms.extend(matches)
    
    # 计算实质性内容长度（去除标准格式文本）
    substantive_content = clean
    # 去除标准SEC头部信息
    substantive_content = re.sub(r'united states securities and exchange commission.*?form 8-k', '', substantive_content)
    substantive_content = re.sub(r'current report.*?securities exchange act of 1934', '', substantive_content)
    substantive_content = re.sub(r'salesforce, inc.*?exact name of registrant', '', substantive_content)
    
    return {
        'items': list(set(items)),
        'dates': list(set(dates)),
        'business_terms': list(set(business_terms)),
        'substantive_length': len(substantive_content),
        'clean_text': clean
    }


def calculate_content_similarity(content1: Dict, content2: Dict) -> Dict:
    """计算内容相似度的多个维度"""
    
    # Items相似度
    items1, items2 = set(content1['items']), set(content2['items'])
    items_similarity = len(items1 & items2) / max(len(items1 | items2), 1)
    
    # 业务词汇相似度
    terms1, terms2 = set(content1['business_terms']), set(content2['business_terms'])
    terms_similarity = len(terms1 & terms2) / max(len(terms1 | terms2), 1)
    
    # 文本相似度
    matcher = difflib.SequenceMatcher(None, content1['clean_text'], content2['clean_text'])
    text_similarity = matcher.ratio()
    
    return {
        'items_similarity': items_similarity,
        'terms_similarity': terms_similarity,
        'text_similarity': text_similarity,
        'overall_similarity': (items_similarity + terms_similarity + text_similarity) / 3
    }


def analyze_document_purpose(content: Dict, filename: str) -> Dict:
    """分析文档的目的和类型"""
    items = content['items']
    terms = content['business_terms']
    dates = content['dates']
    
    purpose = {
        'primary_purpose': 'unknown',
        'secondary_purposes': [],
        'confidence': 0.0,
        'evidence': []
    }
    
    # 基于Items判断
    if '2.02' in items:
        purpose['primary_purpose'] = 'earnings_results'
        purpose['confidence'] += 0.4
        purpose['evidence'].append('Contains Item 2.02 (Results of Operations)')
    
    if '5.07' in items:
        purpose['primary_purpose'] = 'shareholder_meeting'
        purpose['confidence'] += 0.4
        purpose['evidence'].append('Contains Item 5.07 (Shareholder Vote Results)')
    
    if '5.02' in items:
        purpose['secondary_purposes'].append('executive_changes')
        purpose['confidence'] += 0.2
        purpose['evidence'].append('Contains Item 5.02 (Officer/Director Changes)')
    
    if '9.01' in items:
        purpose['secondary_purposes'].append('exhibits_attached')
        purpose['confidence'] += 0.1
        purpose['evidence'].append('Contains Item 9.01 (Exhibits)')
    
    # 基于业务词汇判断
    if any(term in terms for term in ['earnings', 'revenue', 'results']):
        if purpose['primary_purpose'] == 'unknown':
            purpose['primary_purpose'] = 'financial_results'
        purpose['confidence'] += 0.2
        purpose['evidence'].append('Contains financial result terms')
    
    if any(term in terms for term in ['vote', 'voting', 'election', 'meeting']):
        if purpose['primary_purpose'] == 'unknown':
            purpose['primary_purpose'] = 'governance_matter'
        purpose['confidence'] += 0.2
        purpose['evidence'].append('Contains governance terms')
    
    # 基于文件名判断
    if 'results' in filename.lower():
        purpose['confidence'] += 0.1
        purpose['evidence'].append('Filename suggests results announcement')
    
    if 'proxy' in filename.lower() or 'meeting' in filename.lower():
        purpose['confidence'] += 0.1
        purpose['evidence'].append('Filename suggests meeting-related content')
    
    return purpose


def main():
    print("🔍 三个8-K文档相似度和内容分析")
    print("=" * 60)
    
    # 获取文档
    sec_dir = Path("data/raw/sec")
    files = sorted(sec_dir.glob("*8-K*.raw.html"))
    
    if len(files) != 3:
        print(f"❌ 预期3个8-K文档，实际找到 {len(files)} 个")
        return
    
    docs = {}
    contents = {}
    
    # 分析每个文档
    for file_path in files:
        # 提取日期作为标识
        date_match = re.search(r'202[45]-\d{2}-\d{2}', file_path.name)
        short_name = date_match.group() if date_match else file_path.name[-15:]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        key_content = extract_key_content(raw_content)
        purpose = analyze_document_purpose(key_content, file_path.name)
        
        docs[short_name] = {
            'file_path': str(file_path),
            'raw_length': len(raw_content),
            'key_content': key_content,
            'purpose': purpose
        }
        contents[short_name] = key_content
        
        print(f"\n📄 {short_name}")
        print(f"   📏 原始长度: {len(raw_content):,} 字符")
        print(f"   📋 Items: {key_content['items']}")
        print(f"   📅 日期: {key_content['dates'][:3]}")
        print(f"   🎯 主要目的: {purpose['primary_purpose']} (置信度: {purpose['confidence']:.1f})")
        print(f"   💼 业务词汇: {key_content['business_terms'][:5]}")
        if purpose['evidence']:
            print(f"   📝 证据: {'; '.join(purpose['evidence'][:2])}")
    
    # 计算两两相似度
    print(f"\n" + "=" * 60)
    print("🔍 两两相似度分析")
    print("=" * 60)
    
    doc_names = list(docs.keys())
    similarities = {}
    
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            name1, name2 = doc_names[i], doc_names[j]
            sim = calculate_content_similarity(contents[name1], contents[name2])
            similarities[f"{name1} vs {name2}"] = sim
            
            print(f"\n📊 {name1} vs {name2}:")
            print(f"   🎯 Items相似度: {sim['items_similarity']:.1%}")
            print(f"   💼 业务词汇相似度: {sim['terms_similarity']:.1%}")
            print(f"   📝 文本相似度: {sim['text_similarity']:.1%}")
            print(f"   🎯 综合相似度: {sim['overall_similarity']:.1%}")
    
    # 三个文档的综合相似度
    print(f"\n" + "=" * 60)
    print("📊 三文档综合分析")
    print("=" * 60)
    
    all_items = set()
    all_terms = set()
    for content in contents.values():
        all_items.update(content['items'])
        all_terms.update(content['business_terms'])
    
    print(f"📋 所有Items: {sorted(all_items)}")
    print(f"💼 所有业务词汇: {sorted(all_terms)[:10]}")
    
    # 计算平均相似度
    avg_items_sim = sum(s['items_similarity'] for s in similarities.values()) / len(similarities)
    avg_terms_sim = sum(s['terms_similarity'] for s in similarities.values()) / len(similarities)
    avg_text_sim = sum(s['text_similarity'] for s in similarities.values()) / len(similarities)
    avg_overall_sim = sum(s['overall_similarity'] for s in similarities.values()) / len(similarities)
    
    print(f"\n📈 平均相似度:")
    print(f"   🎯 Items: {avg_items_sim:.1%}")
    print(f"   💼 业务词汇: {avg_terms_sim:.1%}")
    print(f"   📝 文本: {avg_text_sim:.1%}")
    print(f"   🎯 综合: {avg_overall_sim:.1%}")
    
    # 合理性判断
    print(f"\n" + "=" * 60)
    print("✅ 合理性评估")
    print("=" * 60)
    
    purposes = [docs[name]['purpose']['primary_purpose'] for name in doc_names]
    unique_purposes = len(set(purposes))
    
    print(f"📋 文档类型多样性: {unique_purposes}/3 种不同目的")
    print(f"📊 平均综合相似度: {avg_overall_sim:.1%}")
    
    if unique_purposes >= 2:
        print("✅ 文档类型多样性良好")
    else:
        print("⚠️  文档类型可能过于相似")
    
    if 0.3 <= avg_overall_sim <= 0.7:
        print("✅ 相似度适中，既有共同格式又有不同内容")
    elif avg_overall_sim < 0.3:
        print("⚠️  相似度较低，可能缺乏一致性")
    else:
        print("⚠️  相似度较高，可能存在重复内容")
    
    # 是否需要更多文档
    print(f"\n🤔 是否需要更多8-K文档？")
    
    covered_items = all_items
    common_items = ['1.01', '1.02', '2.01', '2.03', '3.01', '4.01', '7.01', '8.01']
    missing_items = set(common_items) - covered_items
    
    if missing_items:
        print(f"📋 缺失的常见Items: {sorted(missing_items)}")
        print("💡 建议: 可以考虑添加更多类型的8-K文档")
    else:
        print("✅ 已覆盖主要的8-K事件类型")
    
    # 保存结果
    results = {
        'documents': {name: {
            'purpose': docs[name]['purpose'],
            'key_metrics': {
                'items_count': len(contents[name]['items']),
                'terms_count': len(contents[name]['business_terms']),
                'substantive_length': contents[name]['substantive_length']
            }
        } for name in doc_names},
        'similarities': similarities,
        'overall_assessment': {
            'document_diversity': unique_purposes,
            'average_similarity': avg_overall_sim,
            'covered_items': sorted(all_items),
            'missing_common_items': sorted(missing_items) if missing_items else [],
            'recommendation': 'adequate' if unique_purposes >= 2 and 0.3 <= avg_overall_sim <= 0.7 else 'needs_review'
        }
    }
    
    output_file = "reports/8k_comprehensive_analysis.json"
    Path(output_file).parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 详细结果已保存至: {output_file}")
    print("🎉 分析完成！")


if __name__ == "__main__":
    main()
