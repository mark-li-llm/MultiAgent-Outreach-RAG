#!/usr/bin/env python3
"""
测试标题提取函数的脚本
"""
import re
import os
from typing import Optional

def extract_title(html: str) -> Optional[str]:
    # Prefer og:title
    m = re.search(r"<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"'](.*?)[\"']", html, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # h1
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    if m:
        txt = re.sub(r"<[^>]+>", " ", m.group(1))
        return re.sub(r"\s+", " ", txt).strip()
    # title tag
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        txt = re.sub(r"<[^>]+>", " ", m.group(1))
        return re.sub(r"\s+", " ", txt).strip()
    return None

def extract_title_improved(html: str) -> Optional[str]:
    """改进的标题提取函数，支持更多格式"""

    # 1. 优先og:title - 增加更多匹配格式
    patterns = [
        r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']*)["\']',
        r'<meta\s+content=["\']([^"\']*?)["\']\s+property=["\']og:title["\']',
        r'<meta[^>]+property\s*=\s*["\']og:title["\']\s*content\s*=\s*["\']([^"\']*)["\']',
    ]

    for pattern in patterns:
        m = re.search(pattern, html, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()

    # 2. Twitter标题
    m = re.search(r'<meta[^>]+name=["\']twitter:title["\']\s+content=["\']([^"\']*)["\']', html, re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # 3. h1标签
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    if m:
        txt = re.sub(r"<[^>]+>", " ", m.group(1))
        title = re.sub(r"\s+", " ", txt).strip()
        if title:
            return title

    # 4. title标签
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        txt = re.sub(r"<[^>]+>", " ", m.group(1))
        title = re.sub(r"\s+", " ", txt).strip()
        if title:
            return title

    return None

def test_with_sample_files():
    """测试几个样本文件"""
    test_files = [
        "data/raw/investor_news/crm::press::2025-09-03::news-details::2014af2d.raw.html",
        "data/raw/investor_news/crm::press::2025-05-27::news-details::56b542ba.raw.html",
        "data/raw/investor_news/crm::press::2024-04-30::news-details::05db958d.raw.html"
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n=== Testing {file_path} ===")
            with open(file_path, 'r', encoding='utf-8') as f:
                html = f.read()

            # 测试原始函数
            original_title = extract_title(html)
            print(f"原始函数: {original_title}")

            # 测试改进函数
            improved_title = extract_title_improved(html)
            print(f"改进函数: {improved_title}")

            # 手动检查og:title
            og_match = re.search(r'property=["\']og:title["\']\s+content=["\']([^"\']*)["\']', html, re.IGNORECASE)
            if og_match:
                print(f"手动og:title: {og_match.group(1)}")
            else:
                print("未找到og:title")

if __name__ == "__main__":
    test_with_sample_files()