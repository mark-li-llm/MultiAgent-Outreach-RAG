#!/usr/bin/env python3
"""
修复investor news文档的元数据标题
重新从HTML文件中提取正确的标题信息
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# 添加scripts目录到路径以导入common模块
sys.path.insert(0, os.path.dirname(__file__))
from common import extract_title, try_parse_date_from_meta

def load_json(path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def save_json(path: str, data: Dict[str, Any]) -> bool:
    """保存JSON文件"""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving {path}: {e}")
        return False

def load_html(path: str) -> str:
    """加载HTML文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading HTML {path}: {e}")
        return ""

def fix_metadata_for_file(meta_file: Path, backup_dir: str = "data/backup/metadata_fix") -> bool:
    """修复单个文档的元数据"""

    # 加载现有元数据
    meta_data = load_json(str(meta_file))
    if not meta_data:
        return False

    doc_id = meta_data.get("doc_id", "")
    html_file = meta_file.parent / f"{meta_file.stem.replace('.meta', '')}.raw.html"

    if not html_file.exists():
        print(f"HTML file not found for {doc_id}")
        return False

    # 加载HTML内容
    html_content = load_html(str(html_file))
    if not html_content:
        return False

    # 提取新的标题信息
    new_title = extract_title(html_content)
    new_date = try_parse_date_from_meta(html_content)

    # 检查是否有改进
    old_title = meta_data.get("visible_title", "")
    old_headline = meta_data.get("headline", "")

    changes_made = False
    changes_log = []

    if new_title and new_title != old_title and new_title != "News Details":
        # 创建备份目录
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, f"{doc_id}.meta.json.backup")

        # 备份原始文件
        if not save_json(backup_file, meta_data):
            print(f"Failed to backup {doc_id}")
            return False

        # 更新标题
        meta_data["visible_title"] = new_title
        meta_data["headline"] = new_title
        changes_made = True
        changes_log.append(f"Title: '{old_title}' -> '{new_title}'")

    # 更新日期（如果有更好的日期信息）
    if new_date and not meta_data.get("visible_date"):
        meta_data["visible_date"] = new_date
        changes_made = True
        changes_log.append(f"Added visible_date: {new_date}")

    # 添加修复标记
    if changes_made:
        meta_data["notes"] = f"{meta_data.get('notes', '')}; title_fixed=1".strip("; ")

        # 保存修复后的元数据
        if save_json(str(meta_file), meta_data):
            print(f"✅ Fixed {doc_id}")
            for change in changes_log:
                print(f"   - {change}")
            return True
        else:
            print(f"❌ Failed to save {doc_id}")
            return False
    else:
        print(f"⚪ No changes needed for {doc_id}")
        return True

def main():
    """主函数"""
    data_dir = Path("data/raw/investor_news")

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    # 找到所有元数据文件
    meta_files = list(data_dir.glob("*.meta.json"))
    if not meta_files:
        print("No metadata files found")
        return

    print(f"Found {len(meta_files)} metadata files to process")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for meta_file in sorted(meta_files):
        try:
            if fix_metadata_for_file(meta_file):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"❌ Error processing {meta_file.name}: {e}")
            error_count += 1

    print("=" * 60)
    print(f"Processing complete:")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  📁 Backups saved to: data/backup/metadata_fix/")

    if success_count > 0:
        print("\n🎉 Metadata fix completed! You can now re-run the quality check to verify.")

if __name__ == "__main__":
    main()