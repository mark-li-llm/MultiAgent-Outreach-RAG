#!/usr/bin/env python3
"""
尝试其他方法获取2月8-K的实际内容
"""

import asyncio
import aiohttp
import json
from pathlib import Path


async def try_alternative_urls():
    """尝试不同的URL格式来获取8-K内容"""
    
    # 基础信息
    accession = "000110852425000002"  # 从原URL提取
    cik = "1108524"  # Salesforce的CIK
    filename = "crm-20250226.htm"
    
    # 尝试多个可能的URL格式
    urls_to_try = [
        # 原始EDGAR格式
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}",
        
        # 尝试不同的查看器参数
        f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{cik}/{accession}/{filename}",
        
        # 尝试不同的文件名（可能有变体）
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/crm-20250226.html",
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/crm-20250226.xml",
        
        # 尝试使用XBRL查看器的不同版本
        f"https://www.sec.gov/ixviewer/ix.html?doc=/Archives/edgar/data/{cik}/{accession}/{filename}",
        
        # 尝试直接访问目录
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/",
    ]
    
    print(f"🔍 尝试多个URL来获取8-K内容...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        for i, url in enumerate(urls_to_try, 1):
            print(f"\n🎯 尝试 {i}/{len(urls_to_try)}: {url}")
            
            try:
                async with session.get(url, timeout=10) as response:
                    print(f"   📊 状态: {response.status}")
                    print(f"   📋 类型: {response.content_type}")
                    
                    if response.status == 200:
                        content = await response.text()
                        print(f"   📏 长度: {len(content):,} 字符")
                        
                        # 检查内容类型
                        if "FORM 8-K" in content and "CURRENT REPORT" in content:
                            print("   🎉 找到真正的8-K内容！")
                            await save_fixed_content(content, url)
                            return True
                        elif len(content) > 10000 and "Salesforce" in content:
                            print("   ⚠️  找到相关内容，但可能不是标准8-K格式")
                            print(f"   📝 前200字符: {content[:200]}")
                        elif "loadViewer" in content or "XBRL" in content:
                            print("   🔄 仍然是查看器重定向页面")
                        elif response.status == 200 and len(content) > 1000:
                            print("   📂 可能是目录列表或其他内容")
                            if "crm-" in content:
                                print("   🔍 内容中包含CRM相关文件")
                                # 尝试从目录中提取文件链接
                                await extract_files_from_directory(content, url)
                    else:
                        print(f"   ❌ HTTP错误: {response.status}")
                        
            except asyncio.TimeoutError:
                print("   ⏰ 请求超时")
            except Exception as e:
                print(f"   ❌ 异常: {str(e)}")
    
    return False


async def extract_files_from_directory(directory_content: str, base_url: str):
    """从目录页面提取可能的8-K文件链接"""
    import re
    
    # 查找可能的8-K文件链接
    file_patterns = [
        r'href="([^"]*crm-20250226[^"]*\.htm[^"]*)"',
        r'href="([^"]*8-k[^"]*)"',
        r'href="([^"]*\.htm[^"]*)"',
    ]
    
    found_files = []
    for pattern in file_patterns:
        matches = re.findall(pattern, directory_content, re.IGNORECASE)
        found_files.extend(matches)
    
    if found_files:
        print(f"   📁 在目录中找到 {len(found_files)} 个可能的文件:")
        for file in found_files[:5]:  # 显示前5个
            print(f"      • {file}")
        
        # 尝试访问最有希望的文件
        base_path = base_url.rstrip('/')
        for file in found_files[:3]:  # 尝试前3个
            file_url = f"{base_path}/{file}" if not file.startswith('http') else file
            print(f"   🎯 尝试文件: {file_url}")
            # 这里可以递归调用获取文件内容的逻辑


async def save_fixed_content(content: str, source_url: str):
    """保存修复后的内容"""
    raw_file = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.raw.html")
    backup_file = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.raw.html.backup")
    
    # 备份原文件
    if raw_file.exists() and not backup_file.exists():
        raw_file.rename(backup_file)
        print(f"💾 原文件已备份")
    
    # 保存新内容
    with open(raw_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📝 新内容已保存 ({len(content):,} 字符)")
    
    # 更新meta.json
    meta_file = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.meta.json")
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    meta['final_url'] = source_url
    meta['content_length'] = len(content)
    meta['notes'] = f"fixed_from_alternative_url: {source_url}"
    
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"📋 元数据已更新")


async def search_sec_edgar():
    """使用SEC EDGAR搜索API查找文档"""
    print(f"\n🔍 尝试通过SEC EDGAR API搜索...")
    
    # SEC EDGAR公司搜索API
    search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        'CIK': '1108524',  # Salesforce CIK
        'Find': 'Search',
        'owner': 'exclude',
        'action': 'getcompany',
        'type': '8-K',
        'dateb': '20250228',  # 2025-02-28之前
        'count': '10'
    }
    
    headers = {
        'User-Agent': 'research@example.com',  # SEC要求标识用户
    }
    
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    print(f"📊 搜索结果长度: {len(content)} 字符")
                    
                    # 查找2025-02-26的8-K文档链接
                    import re
                    pattern = r'href="([^"]*Archives[^"]*000110852425000002[^"]*)"'
                    matches = re.findall(pattern, content)
                    
                    if matches:
                        print(f"🎯 找到 {len(matches)} 个相关链接:")
                        for match in matches:
                            full_url = f"https://www.sec.gov{match}" if match.startswith('/') else match
                            print(f"   • {full_url}")
                    else:
                        print("❌ 未找到相关文档链接")
                else:
                    print(f"❌ 搜索失败: {response.status}")
    except Exception as e:
        print(f"❌ 搜索异常: {str(e)}")


async def main():
    print("🔧 尝试替代方法修复2月8-K文档...")
    
    # 方法1: 尝试不同URL
    success = await try_alternative_urls()
    
    if not success:
        print("\n🔍 URL尝试失败，尝试SEC EDGAR搜索...")
        await search_sec_edgar()
    
    print(f"\n📋 修复尝试完成")


if __name__ == "__main__":
    asyncio.run(main())
