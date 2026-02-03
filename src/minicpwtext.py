# D:\qwenchange\minicpm_describe.py
import os
import torch
import json
from PIL import Image
from datetime import datetime
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


def load_model():
    """加载MiniCPM-V模型"""
    model_path = Path(r"D:\qwenchange\models\minicpm")

    print("🔄 正在加载MiniCPM-V模型...")

    # MiniCPM-V使用不同的加载方式
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()
    print("✅ MiniCPM-V模型加载完成")

    if torch.cuda.is_available():
        print(f"📊 使用设备: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("⚠️  使用设备: CPU")

    return model, tokenizer


def describe_image(image_path, model, tokenizer, question=None):
    """描述单张图片"""
    try:
        # 打开图片
        image = Image.open(image_path).convert("RGB")
        print(f"📷 处理图片: {os.path.basename(image_path)}")

        # 默认问题
        if question is None:
            question = "简单描述这张图片的内容。"

        # MiniCPM-V使用不同的调用方式
        print("🤖 生成描述中...")

        # 构建消息
        msgs = [{'role': 'user', 'content': question}]

        # 使用模型的chat方法
        response = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            max_new_tokens=256
        )

        description = response[0]

        # 构建结果
        result = {
            "image_path": str(image_path),
            "image_name": os.path.basename(image_path),
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "question": question,
            "description": description,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "MiniCPM-V-2"
        }

        print(f"✅ 描述完成: {description[:100]}...")
        return result

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_json(result, output_dir):
    """保存为JSON文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    image_name = Path(result["image_path"]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_name}_{timestamp}.json"
    output_path = output_dir / filename

    # 保存JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"💾 结果保存到: {output_path}")
    return output_path


def main():
    """主函数"""
    print("=" * 50)
    print("MiniCPM-V 图片描述工具")
    print("=" * 50)

    # 设置路径
    image_path = r"D:\qwenchange\data\images\1.jpg"
    output_dir = r"D:\qwenchange\data\results"

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        print(f"📁 请检查目录: {os.path.dirname(image_path)}")
        # 列出目录中的文件
        if os.path.exists(os.path.dirname(image_path)):
            print("目录中的文件:")
            for f in os.listdir(os.path.dirname(image_path)):
                print(f"  - {f}")
        return

    # 加载模型
    model, tokenizer = load_model()

    # 描述图片
    result = describe_image(image_path, model, tokenizer)

    if result:
        # 保存JSON
        save_json(result, output_dir)

        # 显示结果摘要
        print("\n" + "=" * 50)
        print("📊 结果摘要")
        print("=" * 50)
        print(f"📷 图片: {result['image_name']}")
        print(f"📐 尺寸: {result['image_size']}")
        print(f"❓ 问题: {result['question']}")
        print(f"📝 描述: {result['description']}")
        print("=" * 50)
    else:
        print("❌ 图片描述失败")


if __name__ == "__main__":
    # 检查模型目录
    model_dir = r"D:\qwenchange\models\minicpm"
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        print("请确保模型已下载到该目录")
        # 检查可能的目录名
        alt_dirs = [
            r"D:\qwenchange\models\minicpm_v2",
            r"D:\qwenchange\models\minicpm_v",
        ]
        for alt_dir in alt_dirs:
            if os.path.exists(alt_dir):
                print(f"📁 发现可能的模型目录: {alt_dir}")
                response = input(f"是否使用此目录? (y/n): ")
                if response.lower() == 'y':
                    # 修改代码中的路径
                    import sys

                    # 这里需要手动修改代码中的路径，或者重新运行
                    print(f"请将代码第9行修改为: model_path = Path(r'{alt_dir}')")
                    sys.exit(1)
    else:
        main()