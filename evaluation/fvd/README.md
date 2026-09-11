# FVD implementation

`calculate_fvd.py` is the SPVC command-line wrapper for the two bundled FVD
implementations:

- `fvd/videogpt/`: VideoGPT I3D implementation and
  `i3d_pretrained_400.pt`;
- `fvd/styleganv/`: StyleGAN-V I3D implementation and
  `i3d_torchscript.pt`.

The original implementation notes and attribution are preserved in
`README-upstream.md`. Normally, invoke FVD through `../evaluate.sh` so CLIP, FID,
and FVD use the same dataset paths:

```bash
bash ../evaluate.sh --dataset nuscenes --scale 4.0
```

For an FVD-only direct invocation:

```bash
python calculate_fvd.py \
  --generated-videos /path/to/generated/videos \
  --gt-videos /path/to/gt/videos \
  --methods videogpt styleganv \
  --max-frames 40
```

Use at least two videos for a covariance-based FVD estimate and videos longer
than ten frames for the I3D temporal downsampling path. The evaluator is
single-process and loads the selected videos into memory.
