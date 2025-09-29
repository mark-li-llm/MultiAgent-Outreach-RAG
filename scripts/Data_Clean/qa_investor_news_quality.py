#!/usr/bin/env python3
"""
Investor News Data Quality Assessment Script
检测investor news数据集的重复性、完整性、多样性和一致性
"""

import json
import os
import re
import hashlib
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any
import difflib
from pathlib import Path

def load_json(path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def load_html(path: str) -> str:
    """加载HTML文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {str(e)}"

def extract_text_from_html(html_content: str) -> str:
    """从HTML中提取文本内容（简单版本）"""
    # 移除script和style标签
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', ' ', html_content)
    # 清理空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def jaccard_similarity(text1: str, text2: str) -> float:
    """计算两个文本的Jaccard相似度"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0

def levenshtein_ratio(s1: str, s2: str) -> float:
    """计算两个字符串的编辑距离相似度"""
    return difflib.SequenceMatcher(None, s1, s2).ratio()

class InvestorNewsQualityChecker:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents = []
        self.issues = defaultdict(list)
        self.stats = {}

    def load_all_documents(self):
        """加载所有文档的元数据和HTML内容"""
        print("Loading all documents...")
        meta_files = list(Path(self.data_dir).glob("*.meta.json"))

        for meta_file in meta_files:
            doc_id = meta_file.stem.replace(".meta", "")
            html_file = meta_file.parent / f"{doc_id}.raw.html"

            meta_data = load_json(str(meta_file))
            html_content = load_html(str(html_file)) if html_file.exists() else ""

            document = {
                "doc_id": doc_id,
                "meta_file": str(meta_file),
                "html_file": str(html_file),
                "meta_data": meta_data,
                "html_content": html_content,
                "text_content": extract_text_from_html(html_content) if html_content else ""
            }
            self.documents.append(document)

        print(f"Loaded {len(self.documents)} documents")

    def check_duplicates(self):
        """检查重复性"""
        print("Checking for duplicates...")

        # 1. SHA256哈希重复检查
        sha256_map = defaultdict(list)
        for doc in self.documents:
            sha256 = doc["meta_data"].get("sha256_raw", "")
            if sha256:
                sha256_map[sha256].append(doc["doc_id"])

        duplicates_by_hash = {k: v for k, v in sha256_map.items() if len(v) > 1}
        if duplicates_by_hash:
            self.issues["duplicate_hash"] = list(duplicates_by_hash.values())

        # 2. 标题相似性检查
        title_duplicates = []
        for i, doc1 in enumerate(self.documents):
            title1 = doc1["meta_data"].get("visible_title", "")
            for j, doc2 in enumerate(self.documents[i+1:], i+1):
                title2 = doc2["meta_data"].get("visible_title", "")
                if title1 and title2:
                    similarity = levenshtein_ratio(title1, title2)
                    if similarity > 0.9:  # 90%相似度认为是重复
                        title_duplicates.append((doc1["doc_id"], doc2["doc_id"], similarity))

        if title_duplicates:
            self.issues["duplicate_titles"] = title_duplicates

        # 3. 内容相似性检查
        content_duplicates = []
        for i, doc1 in enumerate(self.documents):
            text1 = doc1["text_content"][:2000]  # 只比较前2000字符以提高效率
            for j, doc2 in enumerate(self.documents[i+1:], i+1):
                text2 = doc2["text_content"][:2000]
                if len(text1) > 100 and len(text2) > 100:
                    similarity = jaccard_similarity(text1, text2)
                    if similarity > 0.85:  # 85%相似度认为是重复
                        content_duplicates.append((doc1["doc_id"], doc2["doc_id"], similarity))

        if content_duplicates:
            self.issues["duplicate_content"] = content_duplicates

        # 4. URL重复检查
        url_map = defaultdict(list)
        for doc in self.documents:
            url = doc["meta_data"].get("requested_url", "")
            if url:
                url_map[url].append(doc["doc_id"])

        duplicates_by_url = {k: v for k, v in url_map.items() if len(v) > 1}
        if duplicates_by_url:
            self.issues["duplicate_urls"] = list(duplicates_by_url.values())

    def check_completeness(self):
        """检查内容完整性"""
        print("Checking completeness...")

        incomplete_docs = []
        missing_pairs = []
        short_content = []

        for doc in self.documents:
            doc_id = doc["doc_id"]
            meta_data = doc["meta_data"]
            html_content = doc["html_content"]
            text_content = doc["text_content"]

            # 检查文件配对完整性
            if not os.path.exists(doc["html_file"]):
                missing_pairs.append((doc_id, "missing_html"))
            elif "error" in meta_data:
                missing_pairs.append((doc_id, "invalid_meta"))

            # 检查关键字段
            required_fields = ["visible_title", "source_domain", "doctype", "requested_url"]
            missing_fields = [field for field in required_fields if not meta_data.get(field)]
            if missing_fields:
                incomplete_docs.append((doc_id, missing_fields))

            # 检查内容长度
            if len(text_content) < 1000:
                short_content.append((doc_id, len(text_content)))

            # 检查HTTP状态
            if meta_data.get("http_status") != 200:
                incomplete_docs.append((doc_id, f"http_status_{meta_data.get('http_status')}"))

        if incomplete_docs:
            self.issues["incomplete_metadata"] = incomplete_docs
        if missing_pairs:
            self.issues["missing_file_pairs"] = missing_pairs
        if short_content:
            self.issues["short_content"] = short_content

    def check_diversity(self):
        """检查数据多样性"""
        print("Checking diversity...")

        # 1. 时间分布检查
        dates = []
        for doc in self.documents:
            rss_date = doc["meta_data"].get("rss_pubdate", "")
            visible_date = doc["meta_data"].get("visible_date", "")
            if rss_date:
                dates.append(rss_date)
            elif visible_date:
                dates.append(visible_date)

        date_counter = Counter(dates)

        # 2. 内容类型分析
        content_types = []
        keywords = {
            "earnings": ["quarter", "fiscal", "results", "revenue", "earnings"],
            "announcement": ["announces", "appoints", "launches", "unveils"],
            "acquisition": ["acquire", "acquisition", "agreement", "definitive"],
            "dividend": ["dividend", "quarterly dividend"]
        }

        for doc in self.documents:
            title = doc["meta_data"].get("visible_title", "").lower()
            text = doc["text_content"][:500].lower()  # 检查前500字符

            doc_type = "other"
            for category, kws in keywords.items():
                if any(kw in title or kw in text for kw in kws):
                    doc_type = category
                    break
            content_types.append(doc_type)

        content_type_counter = Counter(content_types)

        # 3. 长度分布
        content_lengths = [len(doc["text_content"]) for doc in self.documents]

        self.stats["diversity"] = {
            "date_distribution": dict(date_counter),
            "content_type_distribution": dict(content_type_counter),
            "length_stats": {
                "min": min(content_lengths) if content_lengths else 0,
                "max": max(content_lengths) if content_lengths else 0,
                "avg": sum(content_lengths) / len(content_lengths) if content_lengths else 0
            }
        }

    def check_consistency(self):
        """检查元数据一致性"""
        print("Checking consistency...")

        date_inconsistencies = []
        title_inconsistencies = []
        redirect_issues = []

        for doc in self.documents:
            doc_id = doc["doc_id"]
            meta_data = doc["meta_data"]

            # 1. 日期一致性检查
            visible_date = meta_data.get("visible_date", "")
            rss_date = meta_data.get("rss_pubdate", "")

            if visible_date and rss_date and visible_date != rss_date:
                # 计算日期差异（简单检查）
                try:
                    if abs(len(visible_date.split('-')[0]) - len(rss_date.split('-')[0])) == 0:  # 同年格式
                        date_inconsistencies.append((doc_id, visible_date, rss_date))
                except:
                    pass

            # 2. 标题一致性检查
            visible_title = meta_data.get("visible_title", "")
            headline = meta_data.get("headline", "")

            if visible_title and headline and visible_title != headline:
                similarity = levenshtein_ratio(visible_title, headline)
                if similarity < 0.8:  # 相似度低于80%认为不一致
                    title_inconsistencies.append((doc_id, visible_title, headline, similarity))

            # 3. 重定向检查
            requested_url = meta_data.get("requested_url", "")
            final_url = meta_data.get("final_url", "")
            redirect_chain = meta_data.get("redirect_chain", [])

            if requested_url != final_url and not redirect_chain:
                redirect_issues.append((doc_id, "redirect_without_chain"))

        if date_inconsistencies:
            self.issues["date_inconsistencies"] = date_inconsistencies
        if title_inconsistencies:
            self.issues["title_inconsistencies"] = title_inconsistencies
        if redirect_issues:
            self.issues["redirect_issues"] = redirect_issues

    def generate_report(self) -> Dict[str, Any]:
        """生成质量报告"""
        total_docs = len(self.documents)
        total_issues = sum(len(issues) for issues in self.issues.values())

        # 计算质量评分
        duplicate_score = max(0, 100 - len(self.issues.get("duplicate_hash", [])) * 10 -
                             len(self.issues.get("duplicate_content", [])) * 5)
        completeness_score = max(0, 100 - len(self.issues.get("incomplete_metadata", [])) * 5 -
                                len(self.issues.get("short_content", [])) * 2)
        consistency_score = max(0, 100 - len(self.issues.get("date_inconsistencies", [])) * 3 -
                               len(self.issues.get("title_inconsistencies", [])) * 2)

        overall_score = (duplicate_score + completeness_score + consistency_score) / 3

        report = {
            "summary": {
                "total_documents": total_docs,
                "total_issues_found": total_issues,
                "overall_quality_score": round(overall_score, 2),
                "assessment_timestamp": datetime.now().isoformat()
            },
            "quality_scores": {
                "duplicate_detection": round(duplicate_score, 2),
                "completeness": round(completeness_score, 2),
                "consistency": round(consistency_score, 2)
            },
            "detailed_issues": dict(self.issues),
            "diversity_stats": self.stats.get("diversity", {}),
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """基于发现的问题生成建议"""
        recommendations = []

        if "duplicate_hash" in self.issues:
            recommendations.append("发现完全重复的文档，建议删除重复项")

        if "duplicate_content" in self.issues:
            recommendations.append("发现内容高度相似的文档，建议检查是否为不同版本的同一新闻")

        if "short_content" in self.issues:
            recommendations.append("发现内容过短的文档，可能是抓取不完整或页面加载问题")

        if "incomplete_metadata" in self.issues:
            recommendations.append("发现元数据缺失的文档，建议重新抓取或补充信息")

        if "date_inconsistencies" in self.issues:
            recommendations.append("发现日期不一致的问题，建议统一日期字段的来源和格式")

        return recommendations

def main():
    data_dir = "data/raw/investor_news"
    output_file = "reports/qa/investor_news_quality_report.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 执行质量检查
    checker = InvestorNewsQualityChecker(data_dir)
    checker.load_all_documents()
    checker.check_duplicates()
    checker.check_completeness()
    checker.check_diversity()
    checker.check_consistency()

    # 生成报告
    report = checker.generate_report()

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n质量检测完成！报告已保存到: {output_file}")
    print(f"总体质量评分: {report['summary']['overall_quality_score']}/100")
    print(f"发现问题总数: {report['summary']['total_issues_found']}")

    # 打印主要问题
    if report['detailed_issues']:
        print("\n主要问题类型:")
        for issue_type, issues in report['detailed_issues'].items():
            print(f"- {issue_type}: {len(issues)}个")

if __name__ == "__main__":
    main()