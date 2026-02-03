# D:\qwenchange\minicpm_final_fixed.py
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
    print("正在加载MiniCPM-V模型...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 加载模型时不自动分配设备
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None  # 不自动分配
    )

    # 手动移动到GPU
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    print("MiniCPM-V模型加载完成")

    if torch.cuda.is_available():
        print(f"使用设备: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("使用设备: CPU")

    return model, tokenizer


def describe_image(image_path, model, tokenizer, question=None):
    """描述单张图片"""
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"处理图片: {os.path.basename(image_path)}")

        if question is None:
            question = "简单描述这张图片的内容。"

        print("生成描述中...")

        msgs = [{'role': 'user', 'content': question}]

        # 确保模型在正确的设备上
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 调用chat方法
        response = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            max_new_tokens=256
        )

        description = response[0]

        result = {
            "image_path": str(image_path),
            "image_name": os.path.basename(image_path),
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "question": question,
            "description": description,
            "timestamp": datetime.now().strftime("%Y-%m-d %H:%M:%S"),
            "model": "MiniCPM-V-2"
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
    """主函数"""
    image_path = r"D:\qwenchange\data\images\1.jpg"
    output_dir = r"D:\qwenchange\data\results"

    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return

    model, tokenizer = load_model()

    print("\n可以开始提问了（输入 'q' 退出）\n")

    while True:
        question = input("问题: ").strip()

        if question.lower() == 'q':
            print("退出")
            break

        if not question:
            continue

        result = describe_image(image_path, model, tokenizer, question)

        if result:
            save_json(result, output_dir)
            print(f"结果: {result['description']}\n")
        else:
            print("处理失败\n")


if __name__ == "__main__":
    main()