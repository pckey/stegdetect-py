[English](https://github.com/vu1Art1st/stegdetect-py/blob/main/README.md)| [简体中文](https://github.com/vu1Art1st/stegdetect-py/blob/main/README_CN.md)

# Stegdetect Python Rewrite

A pure Python rewrite of Stegdetect, designed for automated detection of steganographic content in JPEG images. Based on the algorithms from Niels Provos' original C implementation, it uses chi-square analysis and DCT coefficient statistical features for steganalysis.

## Features

- **Multi-Algorithm Steganalysis** — Detection support for five major JPEG steganography tools: JSteg, Outguess, JPHide, Invisible Secrets, and F5
- **Fisher LDA Classifier** — Classification-based detection using feature extraction transforms and linear discriminant analysis
- **Dictionary Attack Cracking** — Password dictionary attacks against JPHide (Blowfish), JSteg (ARC4), and Outguess (ARC4)
- **DCT Coefficient Extraction** — Full JPEG bitstream parsing with quantization table and DCT coefficient extraction
- **Image Comparison Analysis** — Compare DCT coefficient differences between original and stego images
- **JPHide Data Removal** — Remove steganographic data embedded by JPHide
- **Feature Extraction Transforms** — Spline transform, gradient transform, roughness transform, diffsquare transform
- **Pure Python Implementation** — No compilation required, runs cross-platform

## Quick Start

No installation needed — run directly with [uv](https://docs.astral.sh/uv/). uv automatically creates a virtual environment and installs dependencies based on `pyproject.toml`.

```bash
# Detect a single image (dependencies pulled automatically on first run)
uv run stegdetect.py image.jpg
```

> For password cracking support (JPHide Blowfish decryption), add `--extra crack` to install `pycryptodome`:
> ```bash
> uv run --extra crack stegbreak.py -f wordlist.txt image.jpg
> ```

## Command-Line Tools

### stegdetect — Steganalysis Detection

```bash
# Detect a single file (all detectors)
uv run stegdetect.py image.jpg

# Detect multiple files
uv run stegdetect.py image1.jpg image2.jpg image3.jpg

# Detect specific types only
uv run stegdetect.py -t j image.jpg          # JSteg only
uv run stegdetect.py -t o image.jpg          # Outguess only
uv run stegdetect.py -t p image.jpg          # JPHide only
uv run stegdetect.py -t i image.jpg          # Invisible Secrets only
uv run stegdetect.py -t f image.jpg          # F5 only
uv run stegdetect.py -t jopfi image.jpg      # Combined types

# Adjust sensitivity (default 1.0, higher = more sensitive)
uv run stegdetect.py -s 2.0 image.jpg

# Verbose output
uv run stegdetect.py -v image.jpg

# Show version
uv run stegdetect.py -V
```

Example output:

```
image.jpg: "jsteg" (probably, 4096 bits)
image.jpg: "outguess" (possibly)
```

### stegbreak — Password Dictionary Attack

```bash
# Auto-detect stego type and attempt cracking (requires --extra crack)
uv run --extra crack stegbreak.py -f wordlist.txt image.jpg

# Specify stego type
uv run --extra crack stegbreak.py -f wordlist.txt -t jphide image.jpg
uv run --extra crack stegbreak.py -f wordlist.txt -t jsteg image.jpg
uv run --extra crack stegbreak.py -f wordlist.txt -t outguess image.jpg

# Verbose output
uv run --extra crack stegbreak.py -f wordlist.txt -v image.jpg
```

Example output:

```
image.jpg: password="secret123" (jphide, v5, length=2048)
```

### stegcompare — Image Comparison

```bash
# Compare original image with stego image
uv run stegcompare.py original.jpg stego.jpg

# Show difference histogram
uv run stegcompare.py -v original.jpg stego.jpg
```

Example output:

```
Total coefficients: 524288
Changed coefficients: 12345 (2.36%)

Difference histogram:
   -1: 6172
   +1: 6173
```

### stegdeimage — JPHide Data Removal

```bash
# Remove JPHide steganographic data
uv run stegdeimage.py stego.jpg output.jpg
```

Example output:

```
Modified 8432 coefficients
Saved to output.jpg
Note: Full lossless DCT restoration requires JPEG re-encoding infrastructure
```

## Python API

### Steganalysis Detection

```python
from stegdetect.detectors import detect, FLAG_JSTEG, FLAG_OUTGUESS, FLAG_JPHIDE

# Detect an image
results = detect("image.jpg")

for r in results:
    print(f"Type: {r.stego_type}")
    print(f"Confidence: {r.confidence:.4f}")
    print(f"Estimated embedded bits: {r.estimated_bits}")
    print(f"Details: {r.details}")

# Detect specific types only
results = detect("image.jpg", flags=FLAG_JSTEG | FLAG_JPHIDE)

# Adjust sensitivity
results = detect("image.jpg", scale=2.0)
```

### Password Cracking

```python
from stegdetect.stegbreak import crack_file

result = crack_file("image.jpg", "wordlist.txt", stego_type="auto")

if result and result.success:
    print(f"Password: {result.password}")
    print(f"Stego type: {result.stego_type}")
    print(f"Details: {result.details}")
```

#### OutGuess (direct `crack_outguess`)

For OutGuess 0.2 you can also probe a single candidate password directly with
`crack_outguess`. The target is built with `OutguessTarget.prepare`, which decodes
the JPEG and extracts the DCT coefficient stream internally — no manual setup
required. (The bundled OutGuess cracker was fixed to derive the two ARC4 streams
as `MD5("Encryption"+pw)` / `MD5("Seeding"+pw)` and to read the admin header via
the Seeding iterator, fixing earlier false-positives on clean images.)

```python
from stegdetect.stegbreak import OutguessTarget, crack_outguess

target = OutguessTarget.prepare("image.jpg")
result = crack_outguess(target, "secret123")

if result and result.success:
    print(f"Password: {result.password}")
    print(f"Recovered data: {result.data}")
else:
    print("password not matched (or image is clean)")
```

### DCT Coefficient Extraction

```python
from stegdetect.jpeg_dct import read_jpeg, coefficient_stats

# Read JPEG file and extract DCT coefficients
info = read_jpeg("image.jpg")

print(f"Image size: {info.width}x{info.height}")
print(f"Number of components: {info.num_components}")

# View coefficient statistics
stats = coefficient_stats(info)
print(f"Total blocks: {stats['total_blocks']}")
print(f"Non-zero AC coefficients: {stats['nonzero_ac']}")
print(f"JSteg capacity: {stats['capacity_jsteg']} bits")

# Access individual block coefficients
comp = info.components[0]  # Luminance component
print(f"Blocks: {comp.blocks_x}x{comp.blocks_y}")
print(f"DC coefficient of first block: {comp.coeffs[0, 0, 0]}")
print(f"AC coefficients of first block: {comp.coeffs[0, 0, 1:10]}")
```

### Feature Extraction

```python
from stegdetect.extraction import extract_features

# Extract features using spline transform (72-dim feature vector)
features = extract_features("image.jpg", transform_name="spline")

# Available transforms: spline, gradient, roughness, diffsquare
gradient_features = extract_features("image.jpg", transform_name="gradient")
roughness_features = extract_features("image.jpg", transform_name="roughness")
diffsquare_features = extract_features("image.jpg", transform_name="diffsquare")
```

### Fisher LDA Classifier

```python
from stegdetect.discrimination import fisher_lda_train, fisher_lda_classify
from stegdetect.extraction import extract_features

# Prepare training data
stego_features = [extract_features(f, "spline") for f in stego_files]
clean_features = [extract_features(f, "spline") for f in clean_files]

# Train classifier
classifier = fisher_lda_train(stego_features, clean_features)

# Classify
features = extract_features("unknown.jpg", "spline")
is_stego, confidence = fisher_lda_classify(classifier, features)
print(f"Contains stego: {is_stego}, Confidence: {confidence:.4f}")
```

## Outguess Detection Optimization

Specialized optimization for Outguess detection on small images (low resolution, few DCT coefficients). The original algorithm fails when the histogram is sparse due to excessive pair discarding. The optimized version uses **multi-strategy fusion detection**:

### Optimization Strategies

1. **Relaxed Pair Unification** (`unify_outguess_relaxed`) — Only discards pairs where both sides are zero, preventing empty `theo` in sparse histograms
2. **LSB Chi-Square Analysis** — Multi-window sliding detection of LSB distribution anomalies
3. **Pair Asymmetry Analysis** — Detects adjacent histogram bin imbalances caused by embedding
4. **RS (Regular-Singular) Analysis** — Detects the impact of LSB flipping on coefficient smoothness
5. **LSB Randomness Test** — Runs test for randomness of LSB sequences

### Adaptive Weights

Strategy weights are dynamically adjusted based on available coefficient count:
- **Small images** (< 5000 coefficients): chi2 20% + LSB chi2 25% + asymmetry 25% + RS 20% + randomness 10%
- **Large images** (>= 5000 coefficients): chi2 35% + LSB chi2 20% + asymmetry 20% + RS 15% + randomness 10%

### chi2 Fixes

Fixed NaN issues in `chi2_test`:
- `discard` parameter type compatibility for ratios and counts
- Clamp discard values to `[0, 1]`, preventing negative/NaN
- Added NaN/Inf guards returning `0.0`

## Supported Steganography Types

| Stego Tool | Detection | Cracking | Removal | Description |
|-----------|-----------|----------|---------|-------------|
| JSteg   | Yes   | Yes   | -    | LSB replacement of non-zero AC coefficients |
| Outguess| Yes   | Yes   | -    | LSB replacement preserving statistical properties (optimized for small images) |
| JPHide  | Yes   | Yes   | Yes   | Blowfish encryption with LSB embedding |
| Invisible Secrets | Yes | - | - | LSB embedding similar to JSteg |
| F5      | Yes   | -    | -    | Matrix encoding + permutation |

## Project Structure

```
stegdetect-py/
├── pyproject.toml              # Project config and dependencies (uv-managed, no packaging)
├── README.md
├── stegdetect.py               # Detection entry-point (uv run stegdetect.py)
├── stegbreak.py                # Cracking entry-point (uv run stegbreak.py)
├── stegcompare.py              # Comparison entry-point (uv run stegcompare.py)
├── stegdeimage.py              # Removal entry-point (uv run stegdeimage.py)
└── stegdetect/
    ├── __init__.py              # Package initialization
    ├── cli.py                   # stegdetect CLI logic
    ├── jpeg_dct.py              # JPEG parsing and DCT coefficient extraction
    ├── dct.py                   # DCT/IDCT transforms
    ├── detectors.py             # Steganalysis detection algorithms (chi-square analysis)
    ├── histogram.py             # DCT coefficient histograms and unification
    ├── chi2.py                  # Chi-square test
    ├── discrimination.py         # Fisher LDA classifier
    ├── extraction.py            # Feature extraction transforms
    ├── f5.py                    # F5 detection (recompression analysis)
    ├── stegbreak.py             # Password dictionary attack core
    ├── stegbreak_cli.py         # stegbreak CLI logic
    ├── stegcompare_cli.py       # stegcompare CLI logic
    ├── stegdeimage_cli.py       # stegdeimage CLI logic
    └── crypto/
        ├── __init__.py
        ├── arc4.py              # ARC4 (RC4) stream cipher
        ├── blowfish.py          # Blowfish block cipher
        └── md5.py               # MD5 hash
```

## Dependencies

Dependencies are automatically managed by uv based on `pyproject.toml` — no manual installation needed.

**Core dependencies:**

- Python >= 3.9
- Pillow >= 9.0
- NumPy >= 1.22
- SciPy >= 1.8

**Optional dependencies:**

- pycryptodome >= 3.15 (password cracking, enabled via `uv run --extra crack`)

## License

BSD-3-Clause

Based on a rewrite of Niels Provos' original stegdetect implementation (C language version).
