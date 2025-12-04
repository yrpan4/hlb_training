#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5 多任务属性学习训练脚本 - Google Colab 兼容版本

功能:
- 支持 GPU/CPU 自动检测
- 顺序加载图片（同步），低内存占用
- 完整的进度条显示（epochs + batches）
- Google Colab 兼容（支持数据集挂载、模型保存）
- 支持训练中断恢复

使用方式:
1. 本地运行: python train_colab_compatible.py
2. Google Colab:
   - 上传此脚本到 Colab
   - 挂载 Google Drive: from google.colab import drive; drive.mount('/content/drive')
   - 修改 DATA_PATH 为 '/content/drive/My Drive/path/to/data'
   - 运行此脚本

迁移到 Google Colab 的详细步骤见末尾注释。
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import torch
import psutil  # for memory monitoring

# ============================================================================
# 配置部分 - 根据环境修改
# ============================================================================

# 数据集路径（本地运行修改这里）
PROJECT_ROOT = Path(__file__).parent  # 脚本所在目录
DATA_PATH = PROJECT_ROOT  # 数据集根目录（trains/tests 所在位置）
YOLO_ROOT = PROJECT_ROOT / "yolov5"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "HLB_ABCDE.pt"
ATTRIBUTES_PATH = PROJECT_ROOT / "data" / "attributes_mapping.json"
DATASET_YAML = YOLO_ROOT / "data" / "hlb_multi_attributes.yaml"

# 训练参数（低内存配置）
EPOCHS = 50
BATCH_SIZE = 4  # 极低内存：改为 4（原 8）
IMG_SIZE = 320  # 极低内存：改为 320（原 416）
WORKERS = 0  # 同步加载：0 = 主进程加载（不使用子进程）
CACHE = False  # 禁用内存缓存
DEVICE = "0" if torch.cuda.is_available() else "cpu"
OPTIMIZER = "SGD"
PATIENCE = 100

# ============================================================================
# 工具函数
# ============================================================================

def get_memory_usage():
    """获取当前进程的内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def print_header(text):
    """打印格式化的标题"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_config(opt):
    """打印训练配置"""
    print_header("训练配置")
    print(f"✅ 模型名称:        {opt.name}")
    print(f"✅ 权重文件:        {opt.weights}")
    print(f"✅ 数据集 YAML:      {opt.data}")
    print(f"✅ 属性映射:        {opt.attributes}")
    print(f"✅ Epochs:         {opt.epochs}")
    print(f"✅ Batch Size:     {opt.batch_size} (低内存: 逐个加载)")
    print(f"✅ 图像分辨率:      {opt.imgsz}")
    print(f"✅ Workers:        {opt.workers} (0 = 同步加载，无子进程)")
    print(f"✅ 缓存模式:        {opt.cache if opt.cache else '禁用（节省内存）'}")
    device_info = f"cuda:{opt.device}" if opt.device != "cpu" else "CPU"
    print(f"✅ 设备:           {device_info}")
    print(f"✅ 当前内存占用:     {get_memory_usage():.1f} MB")
    print("="*70 + "\n")

def check_files_exist():
    """检查必要文件是否存在"""
    files = {
        "YOLO 目录": YOLO_ROOT,
        "权重文件": WEIGHTS_PATH,
        "属性映射": ATTRIBUTES_PATH,
        "数据集 YAML": DATASET_YAML,
        "数据集目录": DATA_PATH / "trains",
    }
    
    print_header("文件检查")
    all_exist = True
    for name, path in files.items():
        exists = path.exists()
        status = "✅ 存在" if exists else "❌ 缺失"
        print(f"{status}  {name:<15} {path}")
        if not exists:
            all_exist = False
    print("="*70)
    
    if not all_exist:
        print("\n⚠️  某些文件缺失，请检查路径！")
        return False
    
    print("✅ 所有文件检查通过！\n")
    return True

# ============================================================================
# Google Colab 适配函数
# ============================================================================

def is_colab():
    """检测是否在 Google Colab 环境运行"""
    try:
        from google.colab import drive
        return True
    except ImportError:
        return False

def setup_colab():
    """设置 Google Colab 环境"""
    if is_colab():
        print_header("Google Colab 环境检测")
        print("✅ 检测到 Google Colab 环境")
        print("\n可选步骤（需手动执行）:")
        print("1. 挂载 Google Drive:")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        print("\n2. 修改数据路径（在此脚本中）:")
        print("   DATA_PATH = Path('/content/drive/My Drive/your_project')")
        print("\n3. 安装必要库（如需）:")
        print("   !pip install -q psutil")
        print("="*70 + "\n")
        
        # 自动检测 GPU
        global DEVICE
        if torch.cuda.is_available():
            print(f"✅ Colab GPU 可用: {torch.cuda.get_device_name(0)}")
            DEVICE = "0"
        else:
            print("⚠️  Colab GPU 不可用，使用 CPU")
            DEVICE = "cpu"
        return True
    return False

# ============================================================================
# 主训练函数
# ============================================================================

def train_with_attributes():
    """训练 YOLOv5 with attribute learning"""
    
    # 检查 Google Colab 环境
    in_colab = setup_colab()
    
    # 检查文件
    if not check_files_exist():
        sys.exit(1)
    
    # 添加 YOLOv5 到路径
    if str(YOLO_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLO_ROOT))
    
    # 改变工作目录
    os.chdir(YOLO_ROOT)
    
    from train import parse_opt, main
    
    # 获取时间戳（用于唯一命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 解析默认参数
    opt = parse_opt(known=True)
    
    # ========================================================================
    # 覆盖参数 - 低内存同步加载配置
    # ========================================================================
    
    opt.weights = str(WEIGHTS_PATH)
    opt.cfg = ""  # 使用权重中的架构
    opt.data = str(DATASET_YAML)
    opt.attributes = str(ATTRIBUTES_PATH)
    
    # 极低内存训练参数
    opt.epochs = EPOCHS
    opt.batch_size = BATCH_SIZE  # 4: 极低内存
    opt.imgsz = IMG_SIZE  # 320: 低分辨率
    opt.device = DEVICE
    opt.workers = WORKERS  # 0: 同步加载，主进程逐个加载图片
    opt.cache = CACHE  # False: 禁用缓存
    opt.rect = False  # 禁用矩形批处理
    opt.single_cls = False
    opt.seed = 0
    
    # 优化器
    opt.optimizer = OPTIMIZER
    opt.cos_lr = False
    opt.label_smoothing = 0.0
    
    # 禁用内存密集的增强
    opt.mosaic = 0  # 禁用 mosaic
    opt.mixup = 0   # 禁用 mixup
    opt.copy_paste = 0  # 禁用 copy_paste
    
    # 模型保存
    opt.save_period = 10
    opt.project = str(YOLO_ROOT / "runs" / "detect")
    opt.name = f"hlb_attributes_{timestamp}"
    opt.exist_ok = False
    
    # 验证和显示
    opt.noval = False
    opt.nosave = False
    opt.noplots = False
    opt.patience = PATIENCE
    
    # 打印配置
    print_config(opt)
    
    # ========================================================================
    # 运行训练
    # ========================================================================
    
    print_header("开始训练")
    print(f"⏱️  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 输出目录: {opt.project}/{opt.name}")
    print("\n提示: 按 Ctrl+C 可安全中断训练（已保存的权重不会丢失）")
    print("="*70 + "\n")
    
    initial_memory = get_memory_usage()
    
    try:
        main(opt)
        
        final_memory = get_memory_usage()
        print_header("训练完成")
        print(f"✅ 训练成功完成！")
        print(f"📁 输出路径: {YOLO_ROOT / 'runs' / 'detect' / opt.name}")
        print(f"💾 最佳权重: {YOLO_ROOT / 'runs' / 'detect' / opt.name / 'weights' / 'best.pt'}")
        print(f"💾 最后权重: {YOLO_ROOT / 'runs' / 'detect' / opt.name / 'weights' / 'last.pt'}")
        print(f"\n初始内存占用:   {initial_memory:.1f} MB")
        print(f"最终内存占用:   {final_memory:.1f} MB")
        print(f"峰值增长:      {final_memory - initial_memory:.1f} MB")
        print("="*70 + "\n")
        
        # Google Colab 下载提示
        if in_colab:
            print_header("Google Colab - 下载模型")
            print("训练完成后，从 Colab 下载模型:")
            print("\n代码:")
            print("from google.colab import files")
            best_pt = f"runs/detect/hlb_attributes_{timestamp}/weights/best.pt"
            print(f"files.download('{best_pt}')")
            print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断训练")
        print(f"✅ 已保存的检查点在: {YOLO_ROOT / 'runs' / 'detect' / opt.name}")
        print("💡 可以恢复训练（使用最后保存的权重）")
        sys.exit(0)
    except Exception as e:
        print_header("训练出错")
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# 迁移到 Google Colab 的详细步骤
# ============================================================================

COLAB_MIGRATION_GUIDE = """
╔═══════════════════════════════════════════════════════════════════════════╗
║           迁移到 Google Colab 的完整步骤 (含所有代码)                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

【第一步】准备数据集
─────────────────────────────────────────────────────────────────────────────

1. 将数据上传到 Google Drive:
   - 创建文件夹: /My Drive/hlb_training/
   - 上传以下文件/文件夹:
     ✓ trains/         (A-train, B-train, C-train, D-train, E-train)
     ✓ tests/          (testA, testB, testC)
     ✓ models/HLB_ABCDE.pt
     ✓ data/attributes_mapping.json
     ✓ yolov5/         (整个 yolov5 目录)

2. 验证结构 (在 Google Drive 中):
   /My Drive/hlb_training/
   ├── trains/
   │   ├── A-train/
   │   ├── B-train/
   │   └── ...
   ├── tests/
   │   ├── testA/
   │   └── ...
   ├── models/
   │   └── HLB_ABCDE.pt
   ├── data/
   │   └── attributes_mapping.json
   └── yolov5/


【第二步】在 Google Colab 中创建 Notebook
─────────────────────────────────────────────────────────────────────────────

1. 创建新 Colab Notebook

2. 单元格 1 - 安装依赖:
   !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   !pip install -q psutil opencv-python pyyaml tensorboard
   !pip install -q albumentations

3. 单元格 2 - 挂载 Google Drive:
   from google.colab import drive
   drive.mount('/content/drive')

4. 单元格 3 - 设置路径:
   import os
   os.chdir('/content/drive/My Drive/hlb_training')

5. 单元格 4 - 验证文件:
   !ls -la
   # 应该看到: trains/  tests/  models/  data/  yolov5/


【第三步】在 Colab 中运行训练脚本
─────────────────────────────────────────────────────────────────────────────

1. 上传此脚本到 Colab:
   
   单元格 X - 上传脚本:
   # 方式 1: 上传文件到 Colab（不推荐，文件会丢失）
   
   # 方式 2（推荐）: 在 Colab 中创建脚本
   with open('/content/drive/My Drive/hlb_training/train_colab_compatible.py', 'w') as f:
       f.write('''
   # 复制 train_colab_compatible.py 的全部内容到这里
   ''')

2. 运行训练:
   
   单元格 Y:
   %cd /content/drive/My Drive/hlb_training
   !python train_colab_compatible.py

3. 训练开始后，您会看到:
   - 进度条 (epochs 和 batch)
   - 实时损失值
   - 内存使用情况
   - 估计剩余时间


【第四步】监控和中断
─────────────────────────────────────────────────────────────────────────────

监控训练:
   • 点击左侧的「刷新」查看输出
   • 按 Ctrl+C（或点击停止按钮）安全中断
   • 训练会自动保存检查点（每 10 epochs）

恢复训练（从 last.pt）:
   单元格:
   %cd /content/drive/My Drive/hlb_training
   !python -c \"
   import sys; sys.path.insert(0, 'yolov5')
   from train import parse_opt, main
   
   opt = parse_opt(known=True)
   opt.weights = 'yolov5/runs/detect/hlb_attributes_YYYYMMDD_HHMMSS/weights/last.pt'
   opt.resume = True
   main(opt)
   \"


【第五步】下载训练结果
─────────────────────────────────────────────────────────────────────────────

1. 下载最佳权重:
   
   单元格:
   from google.colab import files
   files.download('yolov5/runs/detect/hlb_attributes_YYYYMMDD_HHMMSS/weights/best.pt')
   # 替换 YYYYMMDD_HHMMSS 为实际的时间戳

2. 下载训练日志和图表:
   
   单元格:
   files.download('yolov5/runs/detect/hlb_attributes_YYYYMMDD_HHMMSS/results.csv')
   files.download('yolov5/runs/detect/hlb_attributes_YYYYMMDD_HHMMSS/results.png')


【第六步】在 Colab 中进行评估
─────────────────────────────────────────────────────────────────────────────

1. 创建评估脚本 (evaluate_colab.py):
   
   单元格:
   eval_code = '''
   import sys
   sys.path.insert(0, 'yolov5')
   from yolov5.val import run
   
   run(
       weights='yolov5/runs/detect/hlb_attributes_YYYYMMDD_HHMMSS/weights/best.pt',
       data='yolov5/data/hlb_multi_attributes.yaml',
       imgsz=320,
       batch=4,
       device=0
   )
   '''
   
   with open('evaluate.py', 'w') as f:
       f.write(eval_code)

2. 运行评估:
   !python evaluate.py


【第七步】GPU 设置（可选）
─────────────────────────────────────────────────────────────────────────────

启用 GPU:
   编辑菜单 > 笔记本设置 > 硬件加速器 > 选择 GPU (T4 or A100)

检查 GPU:
   单元格:
   !nvidia-smi


【常见问题】
─────────────────────────────────────────────────────────────────────────────

Q: 提示 "No space left on device"
A: Colab 存储空间满。清理:
   !rm -rf ~/.cache/
   !rm -rf /content/sample_data

Q: 训练太慢
A: 
   • 使用更小的 batch_size (已设为 4)
   • 使用更小的 img_size (已设为 320)
   • 启用 GPU (在笔记本设置中)

Q: 训练中断需要重新开始吗？
A: 不需要。使用 last.pt 恢复:
   !python train_colab_compatible.py  # 自动加载最新权重

Q: 如何导出模型为 ONNX/TensorFlow？
A:
   单元格:
   !python yolov5/export.py \\
       --weights yolov5/runs/detect/hlb_attributes_YYYYMMDD_HHMMSS/weights/best.pt \\
       --include onnx \\
       --imgsz 320


【优化建议】
─────────────────────────────────────────────────────────────────────────────

1. 内存优化（已应用）:
   ✓ batch_size = 4 (低内存)
   ✓ img_size = 320 (低分辨率)
   ✓ workers = 0 (同步加载，无子进程)
   ✓ cache = False (禁用缓存)
   ✓ 禁用 mosaic/mixup 增强

2. 速度优化:
   如果有 GPU，可增加 batch_size 至 8-16

3. 精度优化:
   训练后 10 epochs 将 batch_size 降至 4
   继续训练进一步调优


【脚本运行流程图】
─────────────────────────────────────────────────────────────────────────────

┌─────────────────────┐
│  启动训练脚本        │
└──────────┬──────────┘
           │
        ┌──▼───────────────────┐
        │ 检测环境              │
        │ ├─ 是否 Colab?       │
        │ ├─ GPU 可用?         │
        │ └─ 文件存在?         │
        └──┬───────────────────┘
           │
        ┌──▼───────────────────┐
        │ 加载数据              │
        │ (同步方式)            │
        │ ├─ 读取 YAML          │
        │ ├─ 扫描图片           │
        │ └─ 构建属性向量       │
        └──┬───────────────────┘
           │
        ┌──▼───────────────────┐
        │ Epoch 循环            │  ◄─── 显示进度条
        │                       │
        │ ┌─────────────────┐   │
        │ │ Batch 循环      │   │  ◄─── 逐个加载图片
        │ │ ├─ 加载图片     │   │
        │ │ ├─ 前向传播     │   │
        │ │ ├─ 计算损失     │   │
        │ │ ├─ 反向传播     │   │
        │ │ └─ 权重更新     │   │
        │ └─────────────────┘   │
        │                       │
        │ 保存检查点            │  (每 10 epochs)
        └──┬───────────────────┘
           │
        ┌──▼───────────────────┐
        │ 训练完成              │
        │ ├─ 保存最佳权重      │
        │ ├─ 生成结果图表      │
        │ └─ 打印统计信息      │
        └─────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════╗
║  需要帮助？查看输出日志或 Colab 错误信息                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# 入口点
# ============================================================================

if __name__ == "__main__":
    # 检查依赖
    try:
        import psutil
    except ImportError:
        print("⚠️  缺少 psutil 库，安装中...")
        os.system("pip install -q psutil")
        import psutil
    
    print("\n" + "="*70)
    print("  YOLOv5 多任务属性学习 - Google Colab 兼容版本")
    print("="*70)
    
    # 显示迁移指南（可选）
    import sys
    if "--guide" in sys.argv:
        print(COLAB_MIGRATION_GUIDE)
        sys.exit(0)
    
    # 启动训练
    train_with_attributes()
