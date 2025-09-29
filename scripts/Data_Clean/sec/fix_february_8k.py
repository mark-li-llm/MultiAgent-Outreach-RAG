#!/usr/bin/env python3
"""
修复2月份的8-K文档，获取实际内容而不是XBRL查看器重定向页面
"""

import asyncio
import aiohttp
from urllib.parse import unquote
import json
from pathlib import Path


async def fetch_actual_8k_content():
    """获取2月8-K的实际内容"""
    
    # 从meta.json解析出实际的文档路径
    original_url = "https://www.sec.gov/ix?doc=%2FArchives%2Fedgar%2Fdata%2F1108524%2F000110852425000002%2Fcrm-20250226.htm"
    
    # 解码URL参数，获取实际文档路径
    # %2F = /
    doc_path = "/Archives/edgar/data/1108524/000110852425000002/crm-20250226.htm"
    
    # 构建直接访问URL（绕过XBRL查看器）
    direct_url = f"https://www.sec.gov{doc_path}"
    
    print(f"🔍 尝试获取实际8-K内容...")
    print(f"📄 原始URL: {original_url}")
    print(f"🎯 直接URL: {direct_url}")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(direct_url) as response:
                print(f"📊 HTTP状态: {response.status}")
                print(f"📋 内容类型: {response.content_type}")
                print(f"📏 内容长度: {response.content_length}")
                
                if response.status == 200:
                    content = await response.text()
                    print(f"✅ 成功获取内容: {len(content)} 字符")
                    
                    # 检查是否是真正的8-K内容
                    if "FORM 8-K" in content and "CURRENT REPORT" in content:
                        print("🎉 找到真正的8-K内容！")
                        
                        # 保存新内容
                        raw_file = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.raw.html")
                        backup_file = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.raw.html.backup")
                        
                        # 备份原文件
                        if raw_file.exists():
                            raw_file.rename(backup_file)
                            print(f"💾 原文件已备份为: {backup_file}")
                        
                        # 写入新内容
                        with open(raw_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"📝 新内容已保存至: {raw_file}")
                        
                        # 更新meta.json
                        meta_file = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.meta.json")
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        
                        # 更新元数据
                        meta['requested_url'] = direct_url
                        meta['final_url'] = direct_url
                        meta['content_type'] = response.content_type
                        meta['content_length'] = len(content)
                        meta['notes'] = "fixed_from_xbrl_viewer_redirect"
                        
                        # 备份原meta.json
                        meta_backup = Path("data/raw/sec/crm::8-K::2025-02-26::fy25-results-8-k::97457068.meta.json.backup")
                        if not meta_backup.exists():
                            with open(meta_backup, 'w', encoding='utf-8') as f:
                                json.dump(meta, f, indent=2)
                        
                        # 保存更新的meta.json
                        with open(meta_file, 'w', encoding='utf-8') as f:
                            json.dump(meta, f, indent=2)
                        print(f"📋 元数据已更新")
                        
                        # 简单内容分析
                        analyze_content(content)
                        
                        return True
                    else:
                        print("❌ 获取的内容不是标准8-K格式")
                        print(f"前500字符: {content[:500]}")
                        return False
                else:
                    print(f"❌ HTTP请求失败: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
            return False


def analyze_content(content: str):
    """快速分析8-K内容"""
    print(f"\n📊 内容分析:")
    print(f"📏 总字符数: {len(content):,}")
    
    # 查找关键结构
    if "Item 2.02" in content:
        print("✅ 找到 Item 2.02 (财务结果)")
    if "Item 8.01" in content:
        print("✅ 找到 Item 8.01 (其他事件)")
    if "Results of Operations" in content:
        print("✅ 包含运营结果信息")
    if "Exhibit" in content:
        print("✅ 包含附件引用")
    
    # 查找关键日期和数字
    import re
    dates = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', content)
    if dates:
        print(f"📅 关键日期: {dates[:3]}")  # 显示前3个
    
    # 查找财务数字
    figures = re.findall(r'\$[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand)?', content, re.IGNORECASE)
    if figures:
        print(f"💰 财务数字: {figures[:3]}")  # 显示前3个


async def main():
    print("🔧 开始修复2月8-K文档...")
    success = await fetch_actual_8k_content()
    
    if success:
        print("\n🎉 修复成功！现在重新运行质量检测...")
        import subprocess
        try:
            result = subprocess.run(['python', 'scripts/analyze_8k_content.py'], 
                                  capture_output=True, text=True)
            print("📊 更新后的质量检测结果:")
            print(result.stdout)
        except Exception as e:
            print(f"❌ 无法运行质量检测: {str(e)}")
    else:
        print("\n❌ 修复失败，可能需要手动处理")


if __name__ == "__main__":
    asyncio.run(main())
