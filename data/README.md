# DP Dataset
You can download the DP dataset from [Google Drive](https://drive.google.com/drive/folders/1W_49HId_2FLFH0X9Ry8QiTTyaVt2Y0ks).
The dataset is distributed under the [Creative Commons Attribution-Noncommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## About
The dataset is compressed as seven zip files. Their md5sum values are:
```
6233f6c1f82c961bf9a8a14bf2a2c166  2023-01-17.zip
0821b5fb7551a05e0a304131c135a208  2023-01-18.zip
f7326341e0a2fd55063fa5cadb0bacb7  2023-01-19.zip
7e00a62efb57e04b3d19ffe40f4a80a9  2023-01-24.zip
77a71a36d688fa93cb543c6012e3bf9d  2023-01-25.zip
e1e0f881ac95a0b38f12c006ab7517a7  labels.zip
d1e46d1c7e7d5a0bd19b44de145e7655  keypoints.zip
```

In order to train or test the DP Dataset with the code of this repository, you need to decompress and place them under this `data` directory like below:
```
deepoint
├── data
│   ├── frames_squashed
│   │   ├── 2023-01-17-livingroom
│   │   └── ...
│   ├── labels
│   │   ├── 2023-01-17-livingroom
│   │   └── ...
│   └── keypoints
│       ├── collected_json.pickle
│       └── triangulation.pickle
└── ...
```

- The first five zip files named with `2023-01-%d.zip` each contains two squashed files. Place them under `frames_squashed` directory.
- `labels.zip` and `keypoints.zip` each contains a directory with the same name.

Since the dataset consists of many JPEG images, they are put together with `squashfs`.
You need to mount each files in `frames_squashed` in the `frames` directory.
You can do this by running `./mount_frames.sh`.

### Demo
You can visualize triangulation results using [`visualization.ipynb`](./visualization.ipynb) to better understand the data structure of the DP dataset.

## Structure
### frames_squashed
Each directory in `frames` contains files below:
```
2023-01-17-livingroom
├── venue-info
│   ├── marker_corners.npz
│   └── params.pickle
├── take1
│   ├── 00
│   │   ├── 0000000001.jpg
│   │   └── ...
│   ├── 01
│   ├── ...
│   └── 14
├── take2
└── ...
```
They each contain files with the same `venue`, which means their camera extrinsics and marker positions are the same.
- Files in `venue-info` directory contains calibration results.
	- `venue-info/params.pickle` contains camera parameters.
	- `venue-info/marker_corners.npz` contains 3D coordinates of triangulated marker corners.
- `take*` directories contain 15 directories, each of which contains a synchronized sequential jpg files of the DP Dataset.

### labels
This directory contains files below:
```
labels
├── 2023-01-17-livingroom
│   ├── take1.txt
│   ├── take2.txt
│   ├── take3.txt
│   ├── take4.txt
│   ├── take5.txt
│   ├── take6.txt
│   └── timeinfo.yaml
├── 2023-01-17-openoffice
└── ...
```
Each `*.txt` file contains annotations for the corresponding pointing video, and looks like below:
```
0,  18318
#take3
19, -2,    65,    r
33, 126,   189,   r
39, 244,   307,   r
32, 365,   426,   r
```
- The very first line shows the range of valid frames.
- The second line and later each contains four items separated by a comma:
	- marker id
	- start frame of pointing
	- end frame of pointing
	- pointing arm (`l` or `r`)
- From the second line, lines that start with `#` can appear. They are comments and should be ignored.

### keypoints
This directory contains two files: `collected_json.pickle` and `triangulation.pickle`.
- `collected_json.pickle` contains 2D keypoints detected by [OpenPifPaf](https://github.com/openpifpaf/openpifpaf).
- `triangulation.pickle` contains 3D keypoints calculated with DLT from 2D keypoints and camera parameters above.
