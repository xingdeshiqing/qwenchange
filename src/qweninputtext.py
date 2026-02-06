# D:\qwenchange\qwen_keyboard_input.py
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
    print("正在加载模型...")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()
    print("模型加载完成")

    if torch.cuda.is_available():
        print(f"使用设备: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("使用设备: CPU")

    return model, processor


def describe_image(image_path, model, processor, question):
    """描述单张图片"""
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"处理图片: {os.path.basename(image_path)}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(text=text, images=image, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        print("生成描述中...")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        if "assistant" in generated_text:
            description = generated_text.split("assistant")[-1].strip()
        else:
            description = generated_text.strip()

        result = {
            "image_path": str(image_path),
            "image_name": os.path.basename(image_path),
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "question": question,
            "description": description,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Qwen2.5-VL-7B-Instruct"
        }

        print(f"描述完成")
        return result

    except Exception as e:
        print(f"处理失败: {e}")
        return None


def save_json(result, output_dir):
    """保存为JSON文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(result["image_path"]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_name}_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"结果保存到: {output_path}")
    return output_path


def main():
    """主函数 - 只运行一次"""
    image_path = r"D:\qwenchange\data\images\1.jpg"
    output_dir = r"D:\qwenchange\data\results"

    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return

    # 加载模型
    model, processor = load_model()

    # 获取用户输入的问题
    question = input("请输入问题: ").strip()

    if not question:
        print("未输入问题，程序退出")
        return

    # 处理图片并生成描述
    print(f"\n正在处理问题: {question}")
    result = describe_image(image_path, model, processor, question)

    if result:
        # 保存结果
        save_json(result, output_dir)
        print(f"\n回答: {result['description']}")
    else:
        print("处理失败")

    print("\n程序运行完成，退出")


if __name__ == "__main__":
    main()