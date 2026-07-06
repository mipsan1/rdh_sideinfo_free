# Side-Information-Free Reversible Image Data Hiding — Reproducibility Package

This archive contains the final source code, test data, figures, and manuscript for
the paper *"Side-Information-Free Reversible Image Data Hiding via Sorted
Prediction-Error Expansion."*

All experiments are strictly reversible (BER = 0, MSE = 0) and were run with
Python 3.13 + NumPy (SSIM uses OpenCV `cv2`).

## Directory layout

```
code/         Final pipeline and experiment scripts
manuscript/   Final LaTeX source (MDPI format) + bibliography
figures/      Figures used in the manuscript (PNG)
data/img/     Five standard 512x512 8-bit grayscale test images (BMP)
```

## Code files

| File | Purpose |
|------|---------|
| `rdh_v4.py`        | Base utilities: image I/O, PSNR, checkerboard parity, context-complexity, decoder-recoverable sorting |
| `rdh_final.py`     | **Final scheme**: checkerboard double layer, median-of-4 predictor, shifting PEE, single/double-layer best-of search |
| `rdh_mapfree_v2.py`| Earlier map-free variant (dependency of the multi-predictor ablation) |
| `rdh_multipred.py` | Multi-predictor selection ablation (Section: "Multi-Predictor Selection as an Ablation") |
| `exp1_netbpp.py`   | Net embedding rate: map-based (bz2/zlib compressed position map) vs. map-free |
| `exp2_curves.py`   | Capacity-PSNR curves (Table/Figure: capacity-PSNR) |
| `exp2_one.py`      | Single image/payload helper for the capacity-PSNR sweep |
| `exp3_pairwise.py` | Pairwise-PEE comparison |
| `exp4_predictors.py`| Predictor comparison (median-of-4 vs. inverse-gradient vs. mean rhombus) |
| `exp_ssim.py`      | SSIM + encode/decode runtime at 10,000-bit payload |
| `make_figs.py`     | Regenerates result figures fig1-fig5 |
| `make_schematic.py`| Regenerates the method schematic (fig_method, 600 dpi) |
| `audit_tex.py`     | Structural audit of the LaTeX (labels/refs/cites/bibitems/figures) |
| `check_refs.py`    | Reference integrity check (cited vs. defined) |

## How to reproduce

Run scripts from inside the `code/` directory, with the test images reachable at
`img/` (copy or symlink `../data/img` to `code/img`):

```bash
cd code
ln -s ../data/img img        # or: cp -r ../data/img .
python3 exp2_curves.py       # capacity-PSNR
python3 exp1_netbpp.py       # net embedding rate
python3 exp4_predictors.py   # predictor comparison
python3 exp_ssim.py          # SSIM + runtime (requires cv2)
python3 exp3_pairwise.py     # pairwise PEE
python3 make_figs.py         # result figures
python3 make_schematic.py    # method schematic (600 dpi)
```

## Key locked results (reference)

Capacity-PSNR (dB), best of single/double layer, all strictly reversible
(payloads 5k/10k/20k/30k/40k/50k bits):

| Image | 5k | 10k | 20k | 30k | 40k | 50k |
|-------|----|-----|-----|-----|-----|-----|
| Lena     | 60.39 | 56.80 | 53.31 | 51.28 | 49.77 | 48.55 |
| Boat     | 60.07 | 56.89 | 53.68 | 51.69 | 50.16 | 48.77 |
| Airplane | 65.56 | 61.87 | 57.97 | 55.66 | 53.93 | 52.52 |
| Barbara  | 62.86 | 58.93 | 54.80 | 51.88 | 49.67 | 47.93 |
| Baboon   | 54.61 | 50.99 | 46.86 | 44.24 | 42.22 | 40.58 |

SSIM / encode / decode @ 10,000 bits: Lena 0.9982/171/173 ms (single),
Boat 0.9981/180/177 (single), Airplane 0.9994/177/189 (single),
Barbara 0.9989/175/172 (single), Baboon 0.9965/376/361 (double).

## Notes

- Test images are the standard USC-SIPI / classic grayscale set at 512x512, 8-bit.
- Side information at the reported low-to-moderate payloads is ~80 bits (overflow list empty).
- No LaTeX toolchain is bundled; compile `manuscript/template_revised.tex` with a
  single `pdflatex` pass (bibliography is embedded via `thebibliography`).
