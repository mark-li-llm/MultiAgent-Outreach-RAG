#!/usr/bin/env python3
"""
从SEC EDGAR索引文件获取实际的8-K文档
"""

import asyncio
import aiohttp
import json
import re
from pathlib import Path


async def get_8k_from_index():
    """从索引文件获取8-K文档"""
    
    index_url = "https://www.sec.gov/Archives/edgar/data/1108524/000110852425000002/0001108524-25-000002-index.htm"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    print(f"🔍 获取索引文件内容...")
    print(f"📄 索引URL: {index_url}")
    
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            async with session.get(index_url) as response:
                print(f"📊 状态: {response.status}")
                
                if response.status == 200:
                    content = await response.text()
                    print(f"📏 索引内容长度: {len(content):,} 字符")
                    
                    # 保存索引内容供分析
                    with open("temp_index.html", "w", encoding='utf-8') as f:
                        f.write(content)
                    print("💾 索引内容已保存到 temp_index.html")
                    
                    # 分析索引内容，查找8-K相关文件
                    await analyze_index_content(content, session)
                    
                else:
                    print(f"❌ 无法访问索引: {response.status}")
                    
        except Exception as e:
            print(f"❌ 异常: {str(e)}")


async def analyze_index_content(content: str, session):
    """分析索引内容，查找8-K文档"""
    
    print(f"\n📋 分析索引内容...")
    
    # 查找所有文件链接
    file_pattern = r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
    matches = re.findall(file_pattern, content, re.IGNORECASE)
    
    print(f"🔍 找到 {len(matches)} 个文件链接:")
    
    potential_8k_files = []
    for href, text in matches:
        print(f"   • {text} -> {href}")
        
        # 查找可能的8-K文件
        if any(keyword in text.lower() for keyword in ['8-k', 'form', 'htm', 'xml']) or \
           any(keyword in href.lower() for keyword in ['crm-', '8-k', 'form']):
            potential_8k_files.append((href, text))
    
    print(f"\n🎯 潜在的8-K文件 ({len(potential_8k_files)} 个):")
    for href, text in potential_8k_files:
        print(f"   📄 {text} -> {href}")
    
    # 尝试获取最有希望的文件
    base_url = "https://www.sec.gov/Archives/edgar/data/1108524/000110852425000002/"
    
    for href, text in potential_8k_files:
        if not href.startswith('http'):
            file_url = base_url + href
        else:
            file_url = href
            
        print(f"\n🎯 尝试获取: {text}")
        print(f"   📄 URL: {file_url}")
        
        try:
            async with session.get(file_url) as response:
                print(f"   📊 状态: {response.status}")
                
                if response.status == 200:
                    file_content = await response.text()
                    print(f"   📏 长度: {len(file_content):,} 字符")
                    
                    # 检查是否是真正的8-K内容
                    if check_8k_content(file_content):
                        print(f"   🎉 找到真正的8-K内容！")
                        await save_8k_content(file_content, file_url)
                        return True
                    else:
                        print(f"   ⚠️  不是标准8-K内容")
                        # 保存前500字符供分析
                        print(f"   📝 前500字符: {file_content[:500]}")
                else:
                    print(f"   ❌ HTTP错误: {response.status}")
                    
        except Exception as e:
            print(f"   ❌ 异常: {str(e)}")
    
    return False


def check_8k_content(content: str) -> bool:
    """检查内容是否是真正的8-K文档"""
    
    # 检查关键标识符
    key_indicators = [
        "FORM 8-K",
        "CURRENT REPORT",
        "Item 2.02",  # Results of Operations
        "Item 8.01",  # Other Events  
        "Item 9.01",  # Financial Statements and Exhibits
        "SALESFORCE",
        "Commission File Number"
    ]
    
    found_indicators = []
    for indicator in key_indicators:
        if indicator in content.upper():
            found_indicators.append(indicator)
    
    print(f"   🔍 找到指示符: {found_indicators}")
    
    # 至少需要包含Form 8-K和Current Report
    return len(found_indicators) >= 3 and "FORM 8-K" in found_indicators


async def save_8k_content(content: str, source_url: str):
    """保存8-K内容"""
    
    raw_file = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.raw.html")
    backup_file = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.raw.html.backup")
    
    # 备份原文件
    if raw_file.exists() and not backup_file.exists():
        raw_file.rename(backup_file)
        print(f"💾 原XBRL文件已备份")
    
    # 保存新内容
    with open(raw_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📝 真正的8-K内容已保存 ({len(content):,} 字符)")
    
    # 更新meta.json
    meta_file = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.meta.json")
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    meta['final_url'] = source_url
    meta['content_length'] = len(content)
    meta['content_type'] = 'text/html'
    meta['notes'] = f"fixed_from_edgar_index: {source_url}"
    
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"📋 元数据已更新")
    
    # 快速分析内容
    analyze_8k_quality(content)


def analyze_8k_quality(content: str):
    """分析8-K内容质量"""
    print(f"\n📊 8-K内容质量分析:")
    print(f"📏 总字符数: {len(content):,}")
    
    # 查找关键项目
    items_found = []
    for item in ["Item 1.01", "Item 2.02", "Item 8.01", "Item 9.01"]:
        if item in content:
            items_found.append(item)
    print(f"📋 包含项目: {items_found}")
    
    # 查找关键日期
    import re
    dates = re.findall(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', content)
    if dates:
        print(f"📅 关键日期: {list(set(dates))[:3]}")
    
    # 查找财务数字
    financial_figures = re.findall(r'\$[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand)?', content, re.IGNORECASE)
    if financial_figures:
        print(f"💰 财务数字: {list(set(financial_figures))[:5]}")
    
    # 检查签名
    if "SIGNATURE" in content.upper():
        print(f"✅ 包含签名部分")
    
    # 检查附件
    exhibits = re.findall(r'Exhibit\s+[\d.]+', content, re.IGNORECASE)
    if exhibits:
        print(f"📎 附件: {list(set(exhibits))}")


async def main():
    print("🔧 从EDGAR索引获取真正的8-K内容...")
    
    success = await get_8k_from_index()
    
    if success:
        print("\n🎉 成功获取真正的8-K内容！")
        print("🔄 重新运行质量检测...")
        
        import subprocess
        try:
            result = subprocess.run(['python', 'scripts/analyze_8k_content.py'], 
                                  capture_output=True, text=True, cwd='.')
            print("📊 更新后的质量检测结果:")
            print(result.stdout)
        except Exception as e:
            print(f"❌ 无法运行质量检测: {str(e)}")
    else:
        print("\n❌ 未能获取到真正的8-K内容")
        print("💡 建议: 可能需要手动从SEC网站下载或使用其他数据源")


if __name__ == "__main__":
    asyncio.run(main())
