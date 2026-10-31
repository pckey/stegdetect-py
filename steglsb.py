#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steglsb — 通用 PNG 像素 LSB 隐写提取工具

这是 stegdetect-py 的配套小工具，专注于「PNG 像素最低有效位（LSB）」这一类
隐写。与仓库里基于 JPEG DCT 的 stegbreak 不同，本工具直接读像素、按比特序
提取，并自动从字节流里捞 flag。

特性：
  * 支持 RGB / RGBA 任意通道组合（--channels）
  * 支持任意起始偏移（--start，跳过头部若干 bit）
  * 支持 MSB-first 与 LSB-first 两种比特打包
  * 可选：先用 ZipCrypto 密码把加密 zip 里的 PNG 解出来（--zip/--password）
  * 自动识别 ISCC{...} / flag{...} / CTF{...} 形式的 flag

工作示例（ISCC「隐秘的信息 / LookThatExtraordinaryPicture」）：
  readme.txt 中 Base64 -> easy_to_find_the_flag 作为 zip 密码；
  解出 whereistheflag.png 后，用默认参数（RGB / 像素序 / 起始偏移 3bit / MSB-first）
  即可提取到 ISCC{...}。

用法：
  python steglsb.py                                  # 自动找 png；找不到则从 zip 用默认密码解
  python steglsb.py --png whereistheflag.png
  python steglsb.py --zip misc-LookThatExtraordinaryPicture-1.zip --password easy_to_find_the_flag
  python steglsb.py --start 3 --msb --channels 0,1,2
  python steglsb.py --out payload.bin                # 把完整字节流也落盘，便于二次分析

依赖：numpy, Pillow
"""
import argparse
import os
import re
import zipfile
import numpy as np
from PIL import Image

DEFAULT_PNG = "whereistheflag.png"
DEFAULT_ZIP = "misc-LookThatExtraordinaryPicture-1.zip"
DEFAULT_PW = "easy_to_find_the_flag"

FLAG_RE = re.compile(rb"ISCC\{[^}]*\}|flag\{[^}]*\}|CTF\{[^}]*\}")


def load_pixels(png_path):
    """读取 PNG，返回 (H, W, 4) 的 uint8 数组（RGBA）。"""
    im = Image.open(png_path).convert("RGBA")
    return np.array(im, dtype=np.uint8)


def extract_lsb(arr, channels=(0, 1, 2), start=3, msb=True):
    """
    从 RGBA 数组的指定通道提取 LSB 比特流，打包成字节。
      channels : 通道下标元组，0=R 1=G 2=B 3=A
      start    : 跳过前 N 个 bit（本题 = 3）
      msb      : True=MSB-first 打包；False=LSB-first
    """
    lsb = (arr & 1).astype(np.uint8)
    sub = lsb[:, :, list(channels)]          # (H, W, K)
    bits = sub.reshape(-1)                   # 像素序展平
    b = bits[start:]
    n = (len(b) // 8) * 8
    seg = b[:n].reshape(-1, 8)
    raw = np.packbits(seg, bitorder="big" if msb else "little").tobytes()
    return raw


def decode_payload(raw, max_head=200):
    """从字节流中找 flag，并 dump 头部可读内容。"""
    flags = [p.decode("latin1", "replace") for p in FLAG_RE.findall(raw)]
    # 头部可读预览
    head = raw[:max_head]
    preview = "".join(chr(c) if 32 <= c < 127 else "." for c in head)
    return flags, preview


def main():
    ap = argparse.ArgumentParser(description="PNG 像素 LSB 隐写提取工具 (steglsb)")
    ap.add_argument("--png", default=None, help="直接指定已解密的 PNG 路径")
    ap.add_argument("--zip", default=DEFAULT_ZIP, help="加密 zip 路径（找不到 png 时使用）")
    ap.add_argument("--password", default=DEFAULT_PW, help="zip 解压密码")
    ap.add_argument("--channels", default="0,1,2", help="使用的通道下标，逗号分隔，默认 0,1,2 (RGB)")
    ap.add_argument("--start", type=int, default=3, help="起始跳过的 bit 数，默认 3")
    ap.add_argument("--lsb", dest="msb", action="store_false", help="使用 LSB-first 打包（默认 MSB-first）")
    ap.add_argument("--out", default=None, help="可选：把完整提取字节流写到该文件")
    args = ap.parse_args()

    channels = tuple(int(x) for x in args.channels.split(",") if x.strip() != "")

    # 1) 定位 PNG
    png = args.png
    if png and os.path.exists(png):
        pass
    elif os.path.exists(DEFAULT_PNG):
        png = DEFAULT_PNG
    else:
        # 从 zip 解密
        if not os.path.exists(args.zip):
            print(f"[!] 未找到 PNG（{DEFAULT_PNG}）也未找到 zip（{args.zip}）")
            return
        print(f"[*] 从 zip 解密: {args.zip}")
        z = zipfile.ZipFile(args.zip)
        name = next((n for n in z.namelist() if n.lower().endswith(".png")), None)
        if name is None:
            print("[!] zip 内无 png 文件")
            return
        data = z.read(name, pwd=args.password.encode())
        png = DEFAULT_PNG
        with open(png, "wb") as f:
            f.write(data)
        print(f"[+] 解密写出 {png} ({len(data)} bytes)")

    print(f"[*] 读取像素: {png}")
    arr = load_pixels(png)
    print(f"[*] 图像尺寸: {arr.shape[1]}x{arr.shape[0]}, 通道序: {channels}, start={args.start}, {'MSB' if args.msb else 'LSB'}-first")

    raw = extract_lsb(arr, channels=channels, start=args.start, msb=args.msb)
    print(f"[+] 提取字节流长度: {len(raw)}")

    flags, preview = decode_payload(raw)
    print(f"[*] 头部预览: {preview}")
    if flags:
        print("\n[✅] 找到 flag:")
        for f in flags:
            print("   ", f)
    else:
        print("[!] 未找到 ISCC{/flag{/CTF{ 形式的 flag，可尝试调 --start / --channels / --lsb 重新扫描")

    if args.out:
        with open(args.out, "wb") as f:
            f.write(raw)
        print(f"[+] 完整字节流已写出: {args.out}")


if __name__ == "__main__":
    main()
