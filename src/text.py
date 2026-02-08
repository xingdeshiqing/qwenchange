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
    return model, processor


def generate_response(model, processor, conversation_history, image_path=None, question=None):
    """生成响应，包含完整的对话历史"""
    try:
        # 准备当前消息
        current_message = {"role": "user", "content": []}

        # 如果有图片，添加到消息中
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            current_message["content"].append({"type": "image"})
        else:
            image = None

        # 添加文本问题
        if question:
            current_message["content"].append({"type": "text", "text": question})

        # 添加当前消息到历史
        conversation_history.append(current_message)

        # 使用完整的对话历史生成文本
        text = processor.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # 准备输入
        if image:
            inputs = processor(text=text, images=image, return_tensors="pt")
        else:
            inputs = processor(text=text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        print("生成回答中...")

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
            response = generated_text.split("assistant")[-1].strip()
        else:
            response = generated_text.strip()

        # 添加助手回复到对话历史
        conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response, conversation_history

    except Exception as e:
        print(f"处理失败: {e}")
        return f"处理失败: {str(e)}", conversation_history


def save_conversation(user_input, response, image_used, turn_number, output_dir):
    """保存对话记录"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"turn_{turn_number:03d}.json"
    output_path = output_dir / filename

    result = {
        "turn_number": turn_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "assistant_response": response,
        "image_used": image_used,
        "model": "Qwen2.5-VL-7B-Instruct"
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"对话已保存: {filename}")
    return output_path


def main():
    """主函数"""
    # 固定图片路径
    default_image_path = r"D:\qwenchange\data\images\1.jpg"
    output_dir = r"D:\qwenchange\data\results"

    # 创建本次会话的目录
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(output_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # 检查图片是否存在
    if not os.path.exists(default_image_path):
        print(f"错误：默认图片不存在: {default_image_path}")
        print(f"请确保文件存在: {default_image_path}")
        return

    # 加载模型
    print("正在初始化...")
    model, processor = load_model()

    # 初始化对话历史（包含系统消息）
    conversation_history = [
        {
            "role": "system",
            "content": "你是一个视觉语言助手，可以分析图片并回答问题。请记住对话的上下文，并基于之前的对话进行回答。"
        }
    ]

    turn_count = 0

    while True:
        try:
            user_input = input("用户: ").strip()

            if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
                print("结束对话")
                break

            if not user_input:
                continue

            turn_count += 1

            # 判断是否要处理图片
            image_keywords = ['图片', '图', 'image', 'img', '照片', '分析', '看看', '查看']
            has_image_keyword = any(keyword in user_input for keyword in image_keywords)

            image_path = None
            question = user_input

            if has_image_keyword:
                # 使用固定图片路径
                image_path = default_image_path
                print(f"使用默认图片: {os.path.basename(default_image_path)}")

                # 如果用户只是说"图片"或类似，添加默认问题
                if user_input.strip() in ['图片', '图', 'image', 'img']:
                    question = "描述这张图片"
                else:
                    question = user_input

            # 生成响应（包含完整的对话历史）
            response, conversation_history = generate_response(
                model, processor, conversation_history, image_path, question
            )

            print(f"\n助手: {response}\n")

            # 保存对话
            save_conversation(user_input, response, image_path is not None, turn_count, session_dir)

        except KeyboardInterrupt:
            print("\n结束对话")
            break
        except Exception as e:
            print(f"错误: {e}")
            continue

    # 保存完整的对话历史
    full_history_path = session_dir / "full_conversation.json"
    with open(full_history_path, 'w', encoding='utf-8') as f:
        json.dump({
            "session_id": session_id,
            "total_turns": turn_count,
            "conversation": conversation_history
        }, f, ensure_ascii=False, indent=2)

    print(f"完整对话历史已保存到: {full_history_path}")
    print("对话结束")


if __name__ == "__main__":
    main()