#!/usr/bin/env python3
"""
Generate QR Code for Multi-Agent RAG Demo
可以生成不同格式和尺寸的二维码图片
"""

import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

def generate_basic_qr(url, output_path="qr_code.png", size=10):
    """
    生成基础QR码

    Args:
        url: 要编码的URL
        output_path: 输出文件路径
        size: QR码大小（1-40）
    """
    qr = qrcode.QRCode(
        version=size,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )

    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)
    print(f"✅ Basic QR code saved to: {output_path}")
    return img


def generate_branded_qr(url, output_path="qr_code_branded.png", title="Multi-Agent RAG Demo"):
    """
    生成带标题和说明的品牌化QR码

    Args:
        url: 要编码的URL
        output_path: 输出文件路径
        title: 标题文字
    """
    # Generate QR code
    qr = qrcode.QRCode(
        version=10,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2,
    )

    qr.add_data(url)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white")

    # Create a larger canvas with space for text
    canvas_width = qr_img.width + 100
    canvas_height = qr_img.height + 200

    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

    # Paste QR code
    qr_position = ((canvas_width - qr_img.width) // 2, 80)
    canvas.paste(qr_img, qr_position)

    # Add text
    draw = ImageDraw.Draw(canvas)

    # Try to use a nice font, fallback to default if not available
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        url_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        url_font = ImageFont.load_default()

    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (canvas_width - title_width) // 2
    draw.text((title_x, 20), title, fill='black', font=title_font)

    # Draw subtitle
    subtitle = "扫码访问演示 | Scan to Access Demo"
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_x = (canvas_width - subtitle_width) // 2
    subtitle_y = qr_position[1] + qr_img.height + 20
    draw.text((subtitle_x, subtitle_y), subtitle, fill='gray', font=subtitle_font)

    # Draw URL
    url_display = url if len(url) < 50 else url[:47] + "..."
    url_bbox = draw.textbbox((0, 0), url_display, font=url_font)
    url_width = url_bbox[2] - url_bbox[0]
    url_x = (canvas_width - url_width) // 2
    url_y = subtitle_y + 30
    draw.text((url_x, url_y), url_display, fill='#666666', font=url_font)

    canvas.save(output_path)
    print(f"✅ Branded QR code saved to: {output_path}")
    return canvas


def generate_svg_qr(url, output_path="qr_code.svg"):
    """
    生成SVG格式的QR码（可无限缩放）

    Args:
        url: 要编码的URL
        output_path: 输出文件路径
    """
    import qrcode.image.svg

    factory = qrcode.image.svg.SvgPathImage

    qr = qrcode.QRCode(
        version=10,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        image_factory=factory,
    )

    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)
    print(f"✅ SVG QR code saved to: {output_path}")
    return img


def generate_all_formats():
    """生成所有格式的QR码"""
    url = "https://mark-li-llm.github.io/MultiAgent-Outreach-RAG/"

    # Create output directory
    os.makedirs("qr_codes", exist_ok=True)

    # Generate different formats
    generate_basic_qr(url, "qr_codes/basic.png", size=10)
    generate_basic_qr(url, "qr_codes/basic_small.png", size=5)
    generate_basic_qr(url, "qr_codes/basic_large.png", size=15)
    generate_branded_qr(url, "qr_codes/branded.png")

    try:
        generate_svg_qr(url, "qr_codes/scalable.svg")
    except ImportError:
        print("⚠️  SVG generation requires: pip install qrcode[pil]")

    print("\n📁 All QR codes generated in 'qr_codes' directory!")
    print("   - basic.png: Standard QR code")
    print("   - basic_small.png: Small QR code")
    print("   - basic_large.png: Large QR code")
    print("   - branded.png: QR code with title and description")
    print("   - scalable.svg: Vector format (if available)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate QR codes for the demo")
    parser.add_argument("--url", default="https://mark-li-llm.github.io/MultiAgent-Outreach-RAG/",
                        help="URL to encode (default: demo page)")
    parser.add_argument("--output", default="qr_code.png",
                        help="Output file path")
    parser.add_argument("--type", choices=["basic", "branded", "svg", "all"], default="all",
                        help="Type of QR code to generate")
    parser.add_argument("--size", type=int, default=10,
                        help="QR code size (1-40, default: 10)")

    args = parser.parse_args()

    if args.type == "all":
        generate_all_formats()
    elif args.type == "basic":
        generate_basic_qr(args.url, args.output, args.size)
    elif args.type == "branded":
        generate_branded_qr(args.url, args.output)
    elif args.type == "svg":
        generate_svg_qr(args.url, args.output.replace(".png", ".svg"))

    print("\n💡 Usage tips:")
    print("   - Use PNG format for web/email/presentations")
    print("   - Use SVG format for print materials (scalable)")
    print("   - Use branded version for marketing materials")
    print("   - Test QR code with your phone camera before distribution")