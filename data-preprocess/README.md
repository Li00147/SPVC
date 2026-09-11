# SPVC Data Preprocessing

This directory contains the two preprocessing workflows used to construct SPVC
training and evaluation conditions:

```text
data-preprocess/
├── degraded-video-generation/
│   ├── README.md
│   └── render_degradations.py
└── structured-condition-generation/
    ├── README.md
    ├── proj_combined.py
    ├── proj_map.py
    ├── proj_bbox.py
    └── requirements.txt
```

## Degraded-video generation

[`degraded-video-generation`](degraded-video-generation/) renders
underfitting, cross-view, and random-mask degradations from DriveStudio/OmniRe
scene reconstructions. Its outputs provide the degraded control videos used in
Stage I and Stage II training.

See the
[degraded-video generation guide](degraded-video-generation/README.md) for the
required DriveStudio environment, checkpoint conventions, commands, and output
layout.

## Structured-condition generation

[`structured-condition-generation`](structured-condition-generation/) creates
the nuScenes structured videos used by `reference_combined_video`. Each frame
contains projected HD-map layers and vehicle 3D bounding boxes without the RGB
camera background.

See the
[structured-condition generation guide](structured-condition-generation/README.md)
for dataset requirements, output naming, camera indices, and parallel rendering
commands.

The raw driving datasets, DriveStudio checkpoints, and nuScenes annotations are
not distributed in this source directory. Prepare them separately and follow
their respective licenses.
