# QR Code 实现方案

## 🎯 快速开始

我已经为你创建了几种QR码实现方案：

### 1. 独立QR码页面 (`qr_code.html`)
- **特点**：完整的QR码展示页面，包含下载、打印、调整大小等功能
- **使用**：直接在浏览器打开 `qr_code.html`
- **功能**：
  - 动态调整QR码大小（128px - 512px）
  - 下载QR码图片
  - 打印友好版本
  - 复制链接到剪贴板

### 2. 嵌入式QR码组件 (`qr_code_widget.html`)
- **特点**：可嵌入到现有页面的浮动按钮组件
- **使用**：将代码片段添加到你的主页面
- **效果**：右下角浮动按钮，点击展开QR码

### 3. Python生成脚本 (`generate_qr_code.py`)
- **特点**：生成静态QR码图片文件
- **安装依赖**：
  ```bash
  conda run -n age pip install qrcode[pil]
  ```
- **使用**：
  ```bash
  # 生成所有格式
  conda run -n age python generate_qr_code.py

  # 生成基础版本
  conda run -n age python generate_qr_code.py --type basic

  # 生成品牌版本（带标题）
  conda run -n age python generate_qr_code.py --type branded
  ```

## 📱 部署选项

### 选项1：添加到GitHub Pages（推荐）

1. 将 `qr_code.html` 复制到你的 GitHub Pages 仓库：
   ```bash
   cp qr_code.html ../MultiAgent-Outreach-RAG/qr.html
   ```

2. 推送到GitHub：
   ```bash
   cd ../MultiAgent-Outreach-RAG
   git add qr.html
   git commit -m "Add QR code page"
   git push
   ```

3. 访问：`https://mark-li-llm.github.io/MultiAgent-Outreach-RAG/qr.html`

### 选项2：嵌入到现有演示页面

在你的 `index.html` 中添加浮动QR码按钮：

```html
<!-- 在 </body> 前添加 -->
<div class="qr-float-button" onclick="showQR()">
    <svg width="24" height="24">...</svg>
</div>

<script src="https://cdn.jsdelivr.net/npm/qrcodejs@1.0.0/qrcode.min.js"></script>
<script>
function showQR() {
    // QR码弹窗逻辑
}
</script>
```

### 选项3：生成静态图片用于演示文稿

运行Python脚本生成高质量QR码图片：

```bash
conda run -n age python generate_qr_code.py --type branded --output presentation_qr.png
```

## 🎨 自定义选项

### 修改QR码内容
在所有文件中，将URL替换为你想要的链接：
```javascript
const demoURL = "https://mark-li-llm.github.io/MultiAgent-Outreach-RAG/";
```

### 修改样式
- **颜色**：修改 `colorDark` 和 `colorLight`
- **大小**：调整 `width` 和 `height` 参数
- **纠错级别**：`QRCode.CorrectLevel.H` (高容错率，推荐)

### 添加Logo
在QR码中心添加logo（需要保持30%以下覆盖率）：

```javascript
// 在生成QR码后添加
const canvas = document.querySelector('#qrcode canvas');
const ctx = canvas.getContext('2d');
const logo = new Image();
logo.onload = function() {
    const logoSize = canvas.width * 0.2;
    const x = (canvas.width - logoSize) / 2;
    const y = (canvas.height - logoSize) / 2;
    ctx.drawImage(logo, x, y, logoSize, logoSize);
};
logo.src = 'logo.png';
```

## 📊 使用场景

1. **线下活动**：打印 `qr_code.html` 页面，放置在展台
2. **演示文稿**：使用Python脚本生成高分辨率QR码图片
3. **网站集成**：使用widget版本添加到网站
4. **邮件签名**：生成小尺寸QR码图片嵌入邮件
5. **社交媒体**：分享带品牌的QR码图片

## 🔧 故障排查

### QR码无法扫描
- 确保QR码尺寸至少128px
- 提高纠错级别到 `ERROR_CORRECT_H`
- 增加border（空白边框）到4个单位以上

### 生成失败
- 检查是否安装了依赖：`qrcode[pil]`
- 确保URL格式正确（包含https://）

### 样式问题
- 清除浏览器缓存
- 检查CDN链接是否可访问

## 📝 最佳实践

1. **测试**：生成后务必用多种设备测试扫描
2. **尺寸**：打印用途建议至少300px，屏幕显示200px即可
3. **对比度**：保持黑白高对比度，避免使用浅色
4. **边距**：保留足够的空白边距（quiet zone）
5. **备份**：同时提供可点击的文字链接作为备选

## 🚀 快速测试

立即测试QR码页面：
```bash
open /Users/liyunxiao/repo/ag3/worktrees/agent-faiss/qr_code.html
```

生成所有格式的QR码：
```bash
conda run -n age python generate_qr_code.py --type all
ls -la qr_codes/
```

---

✅ **完成！** 你的Multi-Agent RAG演示现在可以通过QR码轻松分享了。