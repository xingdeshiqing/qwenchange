# D:\qwenchange\simple_describe.py
import os
import torch
import json
from PIL import Image
from datetime import datetime
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def load_model():
    """加载模型"""
    model_path = Path(r"D:\qwenchange\models\qwen_vl")

    print("🔄 正在加载模型...")

    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 加载模型（使用半精度）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()
    print("✅ 模型加载完成")

    if torch.cuda.is_available():
        print(f"📊 使用设备: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("⚠️  使用设备: CPU")

    return model, processor


def describe_image(image_path, model, processor, question=None):
    """描述单张图片"""
    try:
        # 打开图片
        image = Image.open(image_path).convert("RGB")
        print(f"📷 处理图片: {os.path.basename(image_path)}")

        # 默认问题
        if question is None:
            question = "简单描述这张图片的内容。"

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # 预处理
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(text=text, images=image, return_tensors="pt")

        # 移动到GPU
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        print("🤖 生成描述中...")

        # 生成描述（使用较短长度）
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,  # 缩短生成长度
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # 解码结果
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # 提取助手回复
        if "assistant" in generated_text:
            description = generated_text.split("assistant")[-1].strip()
        else:
            description = generated_text.strip()

        # 构建结果
        result = {
            "image_path": str(image_path),
            "image_name": os.path.basename(image_path),
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "question": question,
            "description": description,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Qwen2.5-VL-7B-Instruct"
        }

        print(f"✅ 描述完成: {description[:100]}...")
        return result

    except Exception as e:
        print(f"❌ 处理失败: {e}")
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
    print("Qwen2.5-VL 图片描述工具")
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
    model, processor = load_model()

    # 描述图片
    result = describe_image(image_path, model, processor)

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
    main()