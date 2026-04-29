import os
import glob
import random
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= 1. 参数配置 =================
FOLDER_PATH = '/cpfs01/projects-HDD/cfff-139269c29e92_HDD/gsb_24110190023/wu/AIOptics/data/4f_twophases_2pi_experiment/train/pat'  # 替换为实际文件夹路径
N_SAMPLES = 10000  # 分析的前 N 张图像的数量 (10 <= N <= 总文件数)
BINS_COUNT = 500  # 全局像素分布的区间划分数量
OUTPUT_HTML = 'data_analysis_dashboard.html' 

# --- 裁剪范围设置 (针对最后两个维度 H 和 W) ---
H_START = 100
H_END = 300
W_START = 100
W_END = 300

# ================= 2. 文件读取与预处理 =================
all_files = glob.glob(os.path.join(FOLDER_PATH, "*.npy"))
all_files.sort()

total_files = len(all_files)
print(f"找到的 .npy 文件总数: {total_files}")

if total_files == 0:
    raise ValueError(f"在 {FOLDER_PATH} 中没有找到 .npy 文件，请检查路径。")

N = min(N_SAMPLES, total_files)
if N < 10:
    raise ValueError(f"指定的 N_SAMPLES ({N}) 小于 10，无法满足需求2。")

process_files = all_files[:N]
print(f"已选取前 {N} 张图像进行分析。")

# 预览第一张图的裁剪效果
sample_img = np.load(process_files[0])
cropped_sample = sample_img[..., H_START:H_END, W_START:W_END]
print(f"图像维度检测 - 原始: {sample_img.shape}, 裁剪后: {cropped_sample.shape}\n")

# ================= 3. 数据分析执行 =================

# --- 需求 1 & 3(部分)：全局最大值分布及绝对极值 ---
print("执行第 1 次遍历 (提取裁剪区域的最大值及全局极值)...")
max_values = []
global_min = float('inf')
global_max = float('-inf')

for file_path in tqdm(process_files):
    # 读取并立即执行裁剪
    img = np.load(file_path)[..., H_START:H_END, W_START:W_END]
    
    img_max = img.max()
    img_min = img.min()
    max_values.append(img_max)
    if img_min < global_min: global_min = img_min
    if img_max > global_max: global_max = img_max

# --- 需求 2：随机抽取 10 个样本 ---
print("随机抽取 10 个样本读取裁剪区域的全像素数据...")
random_10_files = random.sample(process_files, 10)
random_10_data = []
random_10_names = []

for file_path in random_10_files:
    # 读取并裁剪，然后展平
    img = np.load(file_path)[..., H_START:H_END, W_START:W_END]
    random_10_data.append(img.flatten())
    random_10_names.append(os.path.basename(file_path))

# --- 需求 3：全局像素分布 (增量计算) ---
print("\n执行第 2 次遍历 (计算全局裁剪区域像素的直方图频数)...")
bins_edges = np.linspace(global_min, global_max, BINS_COUNT + 1)
global_counts = np.zeros(BINS_COUNT, dtype=np.int64)

for file_path in tqdm(process_files):
    # 读取并裁剪
    img = np.load(file_path)[..., H_START:H_END, W_START:W_END]
    counts, _ = np.histogram(img, bins=bins_edges)
    global_counts += counts

bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2


# ================= 4. Plotly 可视化与 HTML 生成 =================
print("\n正在生成交互式 HTML 大屏...")

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        f"需求1：前 {N} 张图像裁剪区域 [{H_START}:{H_END}, {W_START}:{W_END}] 的最大值分布", 
        "需求2：随机抽取的 10 张图像裁剪区域亮度分布 (点击右侧图例可独立显示/隐藏)", 
        f"需求3：前 {N} 张图像裁剪区域全局亮度分布 (共 {global_counts.sum()} 个有效像素)"
    ),
    vertical_spacing=0.22 
)

fig.add_trace(go.Histogram(x=max_values, nbinsx=100, name="最大值分布", marker_color='indigo'), row=1, col=1)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i in range(10):
    fig.add_trace(
        go.Histogram(x=random_10_data[i], opacity=0.6, name=f"{random_10_names[i]}", marker_color=colors[i]),
        row=2, col=1
    )

fig.add_trace(go.Bar(x=bin_centers, y=global_counts, name="全局像素频数", marker_color='teal'), row=3, col=1)

fig.update_layout(
    title=f"数据集亮度分布综合分析面板 (区域范围: {H_START}:{H_END}, {W_START}:{W_END})",
    height=1800, 
    barmode='overlay',
    
    updatemenus=[
        dict(
            type="buttons", direction="right", x=0.35, y=1.05, xanchor="center", yanchor="bottom",
            buttons=list([
                dict(label="线性 Y 轴", method="relayout", 
                     args=[{"yaxis.type": "linear", "yaxis2.type": "linear", "yaxis3.type": "linear"}]),
                dict(label="对数 Y 轴", method="relayout", 
                     args=[{"yaxis.type": "log", "yaxis2.type": "log", "yaxis3.type": "log"}])
            ])
        ),
        dict(
            type="buttons", direction="right", x=0.75, y=1.05, xanchor="center", yanchor="bottom",
            buttons=list([
                dict(label="Y轴: 自适应", method="relayout", 
                     args=[{"yaxis.autorange": True, "yaxis2.autorange": True, "yaxis3.autorange": True}]),
                dict(label="Y轴上限: 10万", method="relayout", 
                     args=[{"yaxis.autorange": False, "yaxis.range": [0, 100000], 
                            "yaxis2.autorange": False, "yaxis2.range": [0, 100000],
                            "yaxis3.autorange": False, "yaxis3.range": [0, 100000]}]),
                dict(label="Y轴上限: 1万", method="relayout", 
                     args=[{"yaxis.autorange": False, "yaxis.range": [0, 10000], 
                            "yaxis2.autorange": False, "yaxis2.range": [0, 10000],
                            "yaxis3.autorange": False, "yaxis3.range": [0, 10000]}]),
                dict(label="Y轴上限: 1千", method="relayout", 
                     args=[{"yaxis.autorange": False, "yaxis.range": [0, 1000], 
                            "yaxis2.autorange": False, "yaxis2.range": [0, 1000],
                            "yaxis3.autorange": False, "yaxis3.range": [0, 1000]}])
            ])
        )
    ]
)

fig.update_yaxes(fixedrange=False)

fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=1, col=1)
fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=2, col=1)
fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=3, col=1)

fig.write_html(OUTPUT_HTML)
print(f"\n处理完成！请下载 '{OUTPUT_HTML}' 并在本地浏览器中打开。")