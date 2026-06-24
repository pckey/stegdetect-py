# Stegdetect Python 重写版

Stegdetect 的纯 Python 重写，用于自动检测 JPEG 图像中的隐写内容。基于 Niels Provos 原始 C 语言实现的算法，使用卡方分析和 DCT 系数统计特征进行隐写检测。

## 功能特性

- **多算法隐写检测** -- 支持 JSteg、Outguess、JPHide、Invisible Secrets、F5 五种主流 JPEG 隐写工具的检测
- **Fisher LDA 分类器** -- 基于特征提取变换和线性判别分析的分类检测
- **字典攻击破解** -- 支持对 JPHide (Blowfish)、JSteg (ARC4)、Outguess (ARC4) 的密码字典攻击
- **DCT 系数提取** -- 完整的 JPEG 位流解析，提取量化表和 DCT 系数
- **图像对比分析** -- 比较原始图像与隐写图像的 DCT 系数差异
- **JPHide 数据擦除** -- 移除 JPHide 嵌入的隐写数据
- **特征提取变换** -- 样条变换、梯度变换、粗糙度变换、差值平方变换
- **纯 Python 实现** -- 无需编译，跨平台运行

## 快速开始

本项目无需安装，使用 [uv](https://docs.astral.sh/uv/) 直接运行即可。uv 会根据 `pyproject.toml` 自动创建虚拟环境并安装依赖。

```bash
# 检测单张图像（首次运行会自动拉取依赖）
uv run stegdetect.py image.jpg
```

> 如需密码破解功能（JPHide Blowfish 解密），附加 `--extra crack` 以安装 `pycryptodome`：
> ```bash
> uv run --extra crack stegbreak.py -f wordlist.txt image.jpg
> ```

## 命令行工具

### stegdetect -- 隐写检测

```bash
# 检测单个文件（使用所有检测器）
uv run stegdetect.py image.jpg

# 检测多个文件
uv run stegdetect.py image1.jpg image2.jpg image3.jpg

# 仅检测特定类型
uv run stegdetect.py -t j image.jpg          # 仅 JSteg
uv run stegdetect.py -t o image.jpg          # 仅 Outguess
uv run stegdetect.py -t p image.jpg          # 仅 JPHide
uv run stegdetect.py -t i image.jpg          # 仅 Invisible Secrets
uv run stegdetect.py -t f image.jpg          # 仅 F5
uv run stegdetect.py -t jopfi image.jpg      # 组合多种类型

# 调整灵敏度（默认 1.0，值越大越敏感）
uv run stegdetect.py -s 2.0 image.jpg

# 详细输出
uv run stegdetect.py -v image.jpg

# 显示版本
uv run stegdetect.py -V
```

输出示例：

```
image.jpg: "jsteg" (probably, 4096 bits)
image.jpg: "outguess" (possibly)
```

### stegbreak -- 密码字典攻击

```bash
# 自动检测隐写类型并尝试破解（需 --extra crack）
uv run --extra crack stegbreak.py -f wordlist.txt image.jpg

# 指定隐写类型
uv run --extra crack stegbreak.py -f wordlist.txt -t jphide image.jpg
uv run --extra crack stegbreak.py -f wordlist.txt -t jsteg image.jpg
uv run --extra crack stegbreak.py -f wordlist.txt -t outguess image.jpg

# 详细输出
uv run --extra crack stegbreak.py -f wordlist.txt -v image.jpg
```

输出示例：

```
image.jpg: password="secret123" (jphide, v5, length=2048)
```

### stegcompare -- 图像对比

```bash
# 比较原始图像与隐写图像
uv run stegcompare.py original.jpg stego.jpg

# 显示差异直方图
uv run stegcompare.py -v original.jpg stego.jpg
```

输出示例：

```
Total coefficients: 524288
Changed coefficients: 12345 (2.36%)

Difference histogram:
   -1: 6172
   +1: 6173
```

### stegdeimage -- JPHide 数据擦除

```bash
# 移除 JPHide 隐写数据
uv run stegdeimage.py stego.jpg output.jpg
```

输出示例：

```
Modified 8432 coefficients
Saved to output.jpg
Note: Full lossless DCT restoration requires JPEG re-encoding infrastructure
```

## Python API

### 隐写检测

```python
from stegdetect.detectors import detect, FLAG_JSTEG, FLAG_OUTGUESS, FLAG_JPHIDE

# 检测图像
results = detect("image.jpg")

for r in results:
    print(f"类型: {r.stego_type}")
    print(f"置信度: {r.confidence:.4f}")
    print(f"估计嵌入位数: {r.estimated_bits}")
    print(f"详情: {r.details}")

# 仅检测特定类型
results = detect("image.jpg", flags=FLAG_JSTEG | FLAG_JPHIDE)

# 调整灵敏度
results = detect("image.jpg", scale=2.0)
```

### 密码破解

```python
from stegdetect.stegbreak import crack_file

result = crack_file("image.jpg", "wordlist.txt", stego_type="auto")

if result and result.success:
    print(f"密码: {result.password}")
    print(f"隐写类型: {result.stego_type}")
    print(f"详情: {result.details}")
```

### DCT 系数提取

```python
from stegdetect.jpeg_dct import read_jpeg, coefficient_stats

# 读取 JPEG 文件并提取 DCT 系数
info = read_jpeg("image.jpg")

print(f"图像尺寸: {info.width}x{info.height}")
print(f"组件数: {info.num_components}")

# 查看系数统计
stats = coefficient_stats(info)
print(f"总块数: {stats['total_blocks']}")
print(f"非零 AC 系数: {stats['nonzero_ac']}")
print(f"JSteg 容量: {stats['capacity_jsteg']} 位")

# 访问单个块的系数
comp = info.components[0]  # 亮度分量
print(f"块数: {comp.blocks_x}x{comp.blocks_y}")
print(f"第一个块的 DC 系数: {comp.coeffs[0, 0, 0]}")
print(f"第一个块的 AC 系数: {comp.coeffs[0, 0, 1:10]}")
```

### 特征提取

```python
from stegdetect.extraction import extract_features

# 使用样条变换提取特征（72 维特征向量）
features = extract_features("image.jpg", transform_name="spline")

# 可用的变换: spline, gradient, roughness, diffsquare
gradient_features = extract_features("image.jpg", transform_name="gradient")
roughness_features = extract_features("image.jpg", transform_name="roughness")
diffsquare_features = extract_features("image.jpg", transform_name="diffsquare")
```

### Fisher LDA 分类器

```python
from stegdetect.discrimination import fisher_lda_train, fisher_lda_classify
from stegdetect.extraction import extract_features

# 准备训练数据
stego_features = [extract_features(f, "spline") for f in stego_files]
clean_features = [extract_features(f, "spline") for f in clean_files]

# 训练分类器
classifier = fisher_lda_train(stego_features, clean_features)

# 分类
features = extract_features("unknown.jpg", "spline")
is_stego, confidence = fisher_lda_classify(classifier, features)
print(f"是否含隐写: {is_stego}, 置信度: {confidence:.4f}")
```

## Outguess 检测优化

针对小图片（低分辨率、DCT 系数少）的 Outguess 检测进行了专门优化。原始算法在直方图稀疏时因过度丢弃 pair 而失效，优化后采用**多策略融合检测**：

### 优化策略

1. **宽松 Pair 配对** (`unify_outguess_relaxed`) — 仅丢弃双方都为零的 pair，避免稀疏直方图下 `theo` 为空
2. **LSB chi-square 分析** — 多窗口尺寸滑动检测 LSB 分布异常
3. **Pair 不对称性分析** — 检测嵌入导致的相邻直方图 bin 失衡
4. **RS (Regular-Singular) 分析** — 检测 LSB 翻转对系数平滑度的影响
5. **LSB 随机性测试** — runs test 检测 LSB 序列的随机性

### 自适应权重

根据可用系数数量动态调整各策略权重：
- **小图片** (< 5000 系数)：chi2 20% + LSB chi2 25% + 不对称性 25% + RS 20% + 随机性 10%
- **大图片** (>= 5000 系数)：chi2 35% + LSB chi2 20% + 不对称性 20% + RS 15% + 随机性 10%

### chi2 修复

修复了 `chi2_test` 中的 nan 问题：
- `discard` 参数类型兼容 ratio 和 count
- discard 值 clamp 到 `[0, 1]`，防止负数/nan
- 添加 nan/inf 防护，返回 `0.0`

## 支持的隐写类型

| 隐写工具 | 检测 | 破解 | 擦除 | 说明 |
|---------|------|------|------|------|
| JSteg   | 是   | 是   | -    | LSB 替换非零 AC 系数 |
| Outguess| 是   | 是   | -    | LSB 替换，保留统计特性（已优化小图片检测） |
| JPHide  | 是   | 是   | 是   | Blowfish 加密，LSB 嵌入 |
| Invisible Secrets | 是 | - | - | 类似 JSteg 的 LSB 嵌入 |
| F5      | 是   | -    | -    | 矩阵编码 + 置换 |

## 项目结构

```
stegdetect-py/
├── pyproject.toml              # 项目配置和依赖（uv 管理，无需打包）
├── README.md
├── stegdetect.py               # 检测入口（uv run stegdetect.py）
├── stegbreak.py                # 破解入口（uv run stegbreak.py）
├── stegcompare.py              # 对比入口（uv run stegcompare.py）
├── stegdeimage.py              # 擦除入口（uv run stegdeimage.py）
└── stegdetect/
    ├── __init__.py              # 包初始化
    ├── cli.py                   # stegdetect 命令行逻辑
    ├── jpeg_dct.py              # JPEG 解析与 DCT 系数提取
    ├── dct.py                   # DCT/IDCT 变换
    ├── detectors.py             # 隐写检测算法（卡方分析）
    ├── histogram.py             # DCT 系数直方图与统一化
    ├── chi2.py                  # 卡方检验
    ├── discrimination.py         # Fisher LDA 分类器
    ├── extraction.py            # 特征提取变换
    ├── f5.py                    # F5 检测（重压缩分析）
    ├── stegbreak.py             # 密码字典攻击核心
    ├── stegbreak_cli.py         # stegbreak 命令行逻辑
    ├── stegcompare_cli.py       # stegcompare 命令行逻辑
    ├── stegdeimage_cli.py       # stegdeimage 命令行逻辑
    └── crypto/
        ├── __init__.py
        ├── arc4.py              # ARC4 (RC4) 流密码
        ├── blowfish.py          # Blowfish 分组密码
        └── md5.py               # MD5 哈希
```

## 依赖

依赖由 uv 根据 `pyproject.toml` 自动管理，无需手动安装。

**核心依赖：**

- Python >= 3.9
- Pillow >= 9.0
- NumPy >= 1.22
- SciPy >= 1.8

**可选依赖：**

- pycryptodome >= 3.15（密码破解功能，通过 `uv run --extra crack` 启用）

## 许可证

BSD-3-Clause

基于 Niels Provos 的原始 stegdetect 实现（C 语言版本）重写。
