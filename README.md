# Qwen2.5-VL 图像描述系统部署文档

## 1. 系统概述
- **项目名称**：Qwen2.5-VL 图像描述系统
- **功能描述**：基于Qwen2.5-VL-7B多模态模型，输入图片和问题，获取AI描述回答
- **运行环境**：Windows 11 + Conda + Python 3.10
- **硬件需求**：NVIDIA GPU（RTX 4060 8GB）、32GB内存、30GB硬盘空间

## 2. 环境准备

### 2.1 安装Miniconda
1. 下载Miniconda（Python 3.10版本）：
   ```
   https://docs.conda.io/en/latest/miniconda.html
   ```
2. 运行安装程序，安装时勾选"Add to PATH"

### 2.2 设置项目目录
根据您的项目结构，确保有以下目录：
```
D:\qwenchange\  # 项目根目录
├── models\     # 存放模型文件
├── data\       # 数据目录
│   ├── images\  # 输入图片
│   └── results\ # 输出结果
└── src\        # 源代码（包含main.py等）
```

## 3. 快速部署步骤

### 步骤1：创建Conda环境
```bash
# 打开Anaconda Prompt或CMD
conda create -n qwen python=3.10 -y
conda activate qwen
```

### 步骤2：安装PyTorch（CUDA 12.1）
```bash
# 安装PyTorch（匹配RTX 4060）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 步骤3：安装项目依赖
在 `D:\qwenchange` 目录下执行：
```bash
pip install -r requirements.txt
```

### 步骤4：设置HuggingFace镜像
```bash
# 设置环境变量解决下载问题
setx HF_ENDPOINT "https://hf-mirror.com"
# 重启命令行窗口生效
```

### 步骤5：下载模型文件
```bash
# 进入项目目录
cd /d D:\qwenchange

# 下载Qwen2.5-VL-7B模型（约15GB）
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/qwen_vl
```

### 步骤6：准备测试图片
将测试图片复制到：
```
D:\qwenchange\data\images\1.jpg
```

## 4. 运行程序

### 方式1：使用main.py运行
```bash
conda activate qwen
cd /d D:\qwenchange\src
python main.py
```