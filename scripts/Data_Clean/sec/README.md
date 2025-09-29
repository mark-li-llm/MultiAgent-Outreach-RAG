# SEC数据清洗和质量检测工具

本目录包含了用于SEC文档数据质量分析和修复的所有Python脚本。

## 📁 文件说明

### 🔍 质量检测脚本

1. **`sec_quality_check.py`** - 基础质量检测
   - 文档完整性检查
   - 重复内容检测 (SHA256哈希对比)
   - 文档间相似度分析
   - 使用方法: `python sec_quality_check.py`

2. **`analyze_8k_content.py`** - 8-K文档深度分析
   - XBRL查看器重定向检测
   - 文档结构完整性分析
   - 实质性内容评估
   - 使用方法: `python analyze_8k_content.py`

### 📊 相似度分析脚本

3. **`compare_8k_lengths.py`** - 长度和相似度对比
   - 文档长度统计分析
   - 两两相似度计算
   - 长度合理性判断
   - 使用方法: `python compare_8k_lengths.py`

4. **`calculate_3way_similarity.py`** - 综合相似度分析
   - 多维度相似度计算 (Items、业务词汇、文本)
   - 文档目的和类型识别
   - 综合质量评估报告
   - 使用方法: `python calculate_3way_similarity.py`

### 🔧 修复工具脚本

5. **`fix_february_8k.py`** - 直接URL修复尝试
   - 尝试直接访问SEC文档URL
   - 自动内容替换和元数据更新
   - 注意: 可能因SEC访问限制而失败

6. **`alternative_8k_fix.py`** - 替代修复方法
   - 多种URL格式尝试
   - SEC EDGAR搜索API调用
   - 目录文件分析

7. **`get_8k_from_index.py`** - 从EDGAR索引获取
   - EDGAR索引文件解析
   - 文档链接提取和验证
   - 内容质量检查

## 🚀 使用流程

### 基础质量检测
```bash
# 1. 运行基础质量检测
python sec_quality_check.py

# 2. 深度分析8-K文档
python analyze_8k_content.py
```

### 相似度分析
```bash
# 3. 长度和相似度对比
python compare_8k_lengths.py

# 4. 综合相似度分析
python calculate_3way_similarity.py
```

### 问题修复 (如需要)
```bash
# 5. 尝试自动修复 (可选)
python fix_february_8k.py
python alternative_8k_fix.py
python get_8k_from_index.py
```

## 📊 输出文件

脚本运行后会在以下位置生成报告：

- `reports/sec_quality_analysis.json` - 基础质量检测结果
- `reports/8k_detailed_analysis.json` - 8-K详细分析结果  
- `reports/8k_length_similarity_analysis.json` - 长度相似度数据
- `reports/8k_comprehensive_analysis.json` - 综合分析报告

## ⚠️ 注意事项

1. **运行环境**: 确保在项目根目录运行脚本
2. **网络访问**: 修复脚本可能需要访问SEC网站
3. **备份**: 修复操作会自动创建备份文件
4. **依赖**: 需要安装 `aiohttp`, `pathlib` 等依赖库

## 🎯 质量标准

- **完整性**: 100% (所有文档都有meta.json和内容文件)
- **重复率**: 0% (无重复内容)
- **相似度**: 30-70% (适中水平)
- **结构**: 符合SEC标准格式

## 📝 维护记录

- **创建时间**: 2025-09-28
- **最后更新**: 2025-09-28  
- **版本**: v1.0
- **状态**: 生产就绪 ✅
