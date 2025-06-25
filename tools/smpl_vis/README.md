# SMPL Motions Visualization

This folder describes how to visualize the SMPL format motion data. We provide two methods: `Blender` and `PyTorch3D`.

## Blender Vis

[Blender](https://www.blender.org/) is an open source software for 3D CG. It's also a powerful tool for motion visualization.

Import `npz` motion data into blender for visualization.

1. Download version `2.9.0` of blender and the [SMPL add-on](https://smpl.is.tue.mpg.de/index.html).

2. Add the SMPL object to the blender scene and run the `import_motion_blender.py` script in the `scripting` bar to bind the motion data to the SMPL object.

## PyTorch3D Vis

This implementation is adapted from the `GVHMR` code, so please refer to its [installation](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md) process.

`smpl_neutral_J_regressor.pt` and `smplx2smpl_sparse.pt` must be put in `./body_model`. Download [SMPLX] parameters and place it in the `../smpl_retarget/smpl_model/smplx/` folder. Rename the file to `SMPLX_NEUTRAL.npz`. The folder structure of `../smpl_retarget/smpl_model` should be organized like this:
```
smpl_model/
├── smpl/
|    └── SMPL_FEMALE.pkl
|    └── SMPL_MALE.pkl
|    └── SMPL_NEUTRAL.pkl
└── smplx/
     └── SMPLX_NEUTRAL.npz
```

Run the following command to visualize motion:

```
python smpl_render.py --filepath <PATH_TO_MOTION>
```

- `<PATH_TO_MOTION>` contains `.npz`