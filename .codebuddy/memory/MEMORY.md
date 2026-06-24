# stegdetect-py 项目长期记忆

## 项目约定

- **运行方式**：项目不打包安装，通过 `uv run <script>.py` 直接执行。根目录有四个入口脚本：`stegdetect.py`、`stegbreak.py`、`stegcompare.py`、`stegdeimage.py`，各自导入 `stegdetect` 包内对应 `*_cli.main`。
- **依赖管理**：`pyproject.toml` 仅含 `[project]`（核心依赖 Pillow/numpy/scipy）与 `[project.optional-dependencies]`（crack=pycryptodome, dev=pytest），无 build-system/setuptools 配置。密码破解需 `uv run --extra crack`。
- **包结构**：`stegdetect/` 为 Python 包，CLI 模块使用相对导入，不能直接作为脚本运行，须经根目录入口脚本调用。

## 环境

- OS: Windows (PowerShell Core)
- 工作区: e:\SecTools\ctf-Misc\stegdetect-py

## 检测能力

支持的隐写检测类型（`-t` 参数）：
- `j` = JSteg, `o` = Outguess, `p` = JPHide, `i` = Invisible Secrets, `f` = F5 (快速), `F` = F5 (慢速), `h` = Steghide
- 默认 `-t ""` = 全部类型
- Steghide 检测使用3策略（窗口卡方均值/跨度/局部分块），无共识惩罚，含 gap_bonus 机制区分纯steghide vs 混合嵌入

## 关键修改

- **Outguess 检测**：新增 `prepare_outguess_detect`（含±1）和 `unify_outguess_k2`（扩展卡方 k=2），使 `mmm.jpg` 可被检测到 (2026-06-17)
- **F5 检测误报修复**：根因是质量不匹配（用 Q=90-98 重压缩 Q=75 原图）。修复：新增 `_estimate_jpeg_quality()` 从量化表反推质量，用预估质量重压缩。新增 `_gen_f5_test.py` 可生成不同嵌入率的 F5 测试图。beta 值随嵌入率单调递增，清洁图零误报。 (2026-06-17)
- CLI: `files` 参数改为 `nargs='*'`，`-V` 在文件校验前执行
