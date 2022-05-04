
# Learning Interactive Multi-Object Segmentation

## 1. Paper

**"Yan Gui, Bingqing Zhou, Jianming Zhang, Cheng Sun, Lingyun Xiang, Jin Zhang. Learning Interactive Multi-Object Segmentation through Appearance Embedding and Spatial Attention, Submitted to IET Image Processing, 2021."**

Overall architecture of Interactive Multi-Object Segmentation Network, as shown below.

![example](fig1.png)

experimental result data (.xlsx) in our paper：[experimental_results.zip](https://github.com/BingqiangZhou/Learning-Interactive-Multi-Object-Segmentation/releases/tag/experimental-results)

## 2. Run Demo App

### 2.1 Download [model file](https://github.com/BingqiangZhou/Learning-Interactive-Multi-Object-Segmentation/releases/download/model/best_mean_iou_epoch.pkl), and put it to `models` folder

### 2.2 config python env，install dependent packages and run demo

```bash
## 1. create conda virtual env.
conda create -n mos python=3.6

## 2. activate conda virtual env.
conda activate mos

## 3. install pytorch, reference url: https://pytorch.org.
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

## 4. install other dependent packages.
conda install matplotlib=3.2 opencv-python scikit-learn yacs loguru tensorboard

## 5. select GPU or CPU to inference.
modify the value of "device_num" in file "./demo/args.py", for CPU，set "device_num" to number less than 0, for GPU, set "device_num" to the Number of GPU.

## 6. into the workspaces of demo.
cd ./demo

## 7. run demo app by python file. (if you use ssh connect linux server to run deme app, you can skip this step, see 7-th step).
python demo.py

## 8. run demo app by jupyter notebook (you need run `conda install -c conda-forge notebook` to install jupyter notebook), and then run the last cell of `Demo.ipynb`.
```

How to Segmentation, you can see **chapter 2.3**

### 2.3 **Segmentation Demo**

**operation:**

- mouse:
  - [left button]：interacte
  - [right button]：cancel last interactation
- keyboard:
  - [number key, include 1-9]: n-th object mark
  - ['p' key]: predict result when not in "auto predict" mode
  - ['ctrl' + 'alt' + 's' key]：save result inlcude predict label, embedding map(random projection), visual attention map
  - ['c' key]: change mode, 'auto predict' or 'press 'p' to predict'
  - ['b' key]: change to before image
  - ['a' key]: change to after image
  - ['r' key]:  reset interaction

![example](SegDemo.gif)

## 3. Train

### 3.1 Download [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset and extract files

```bash
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

tar -xvf VOCtrainval_11-May-2012.tar
```

### 3.2 modify config file

setting the path of voc2012 and whether use GPU in `./config/MOS.yaml`. (Tips: On this file, you can change the configuration of dataset, loss, network and so on.)

```bash
# line 8-9
BASIC:
  USE_GPU: True
  GPU: 0

# line 21
DATASET:
  DATASET_ROOT_FOLDER:
```

### 3.3 run main.py

```bash
python main.py -c ./config/MOS.yaml 
```

## 4. Test

### 4.1 generate interactive pairs

```bash
# 1. install refernece package
pip install tqdm pandas openpyxl

# 2. setting the path of VOC2012 to 'voc_root_dir' on 'generate_interactive_pairs_voc.py'

# 3. run generate_interactive_pairs_voc.py
python generate_interactive_pairs_voc.py
```

### 4.2 test Our-S and Our-M on VOC2012 dataset

```bash
# 1. setting the path of VOC2012 to 'voc_root_dir' on 'test_voc_singal_object.py'(Our-S) or 'test_voc_mutli_object.py'(Our-M)

# 2. setting the path of interaction generated to 'interactives_dir' on 'test_voc_singal_object.py'(Our-S) or 'test_voc_mutli_object.py'(Our-M)

# 3. test Our-S
python test_voc_singal_object.py

# 4. test Our-M
python test_voc_mutli_object.py
```

### 4.3 test Our-S on GrabCut dataset

```bash
# 1. download GrabCut dataset and extract files
wget -c https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip

unzip GrabCut.zip

# 2. modify folder name of image and gt
mv data_GT images
mv boundary_GT masks

# 3. setting the path of GrabCut to 'datasets_root_dir' on 'generate_interactive_pairs_grabcut_berkeley.py'

# 4. setting the dataset name to 'datasets' list on 'generate_interactive_pairs_grabcut_berkeley.py'.

# 5. run 'generate_interactive_pairs_grabcut_berkeley.py' to genrate interaction.
python generate_interactive_pairs_grabcut_berkeley.py

# 6. setting the path of interaction generated to 'interactives_dir' on 'test_grabcut_berkeley.py'

# 7. setting the path of GrabCut to 'dataset_root_dir' on 'test_grabcut_berkeley.py'

# 8. setting the path of result to 'out_dir' on 'test_grabcut_berkeley.py'

# 9. run 'test_grabcut_berkeley.py' to test
python test_grabcut_berkeley.py
```

### 4.4 test Our-S on Berkeley dataset

```bash
# download GrabCut dataset and extract files
wget -c https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip

unzip Berkeley.zip

# 2. setting the path of Berkeley to 'datasets_root_dir' on 'generate_interactive_pairs_grabcut_berkeley.py'

# 3. setting the dataset name to 'datasets' list on 'generate_interactive_pairs_grabcut_berkeley.py'.

# 4. run 'generate_interactive_pairs_grabcut_berkeley.py' to genrate interaction.
python generate_interactive_pairs_grabcut_berkeley.py

# 5. setting the path of interaction generated to 'interactives_dir' on 'test_grabcut_berkeley.py'

# 6. setting the path of Berkeley to 'dataset_root_dir' on 'test_grabcut_berkeley.py'

# 7. setting the path of result to 'out_dir' on 'test_grabcut_berkeley.py'

# 8. run 'test_grabcut_berkeley.py' to test
python test_grabcut_berkeley.py
```

## 5. Other Datasets for Interactive Segmentation

The following table contains some datasets for interactive image segmentation，and the table is derived from [saic-vul/ritm_interactive_segmentation](https://github.com/saic-vul/ritm_interactive_segmentation).

| Dataset   |                      Description             |           Download Link              |
|-----------|----------------------------------------------|:------------------------------------:|
|ADE20k     |  22k images with 434k instances (total)      |  [official site][ADE20k]             |
|OpenImages |  944k images with 2.6M instances (total)     |  [official site][OpenImages]         |
|MS COCO    |  118k images with 1.2M instances (train)     |  [official site][MSCOCO]             |
|LVIS v1.0  |  100k images with 1.2M instances (total)     |  [official site][LVIS]               |
|COCO+LVIS* |  99k images with 1.5M instances (train)      |  [original LVIS images][LVIS] + [our combined annotations][COCOLVIS_annotation] |
|SBD        |  8498 images with 20172 instances for (train) 2857 images with 6671 instances for (test) |[official site][SBD]|
|Grab Cut   |  50 images with one object each (test)       |  [GrabCut.zip (11 MB)][GrabCut]      |
|Berkeley   |  96 images with 100 instances (test)         |  [Berkeley.zip (7 MB)][Berkeley]     |
|DAVIS      |  345 images with one object each (test)      |  [DAVIS.zip (43 MB)][DAVIS]          |
|Pascal VOC |  1449 images with 3417 instances (validation)|  [official site][PascalVOC]          |
|COCO_MVal  |  800 images with 800 instances (test)        |  [COCO_MVal.zip (127 MB)][COCO_MVal] |

[ADE20k]: http://sceneparsing.csail.mit.edu/
[OpenImages]: https://storage.googleapis.com/openimages/web/download.html
[MSCOCO]: https://cocodataset.org/#download
[LVIS]: https://www.lvisdataset.org/dataset
[SBD]: http://home.bharathh.info/pubs/codes/SBD/download.html
[GrabCut]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip
[Berkeley]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip
[DAVIS]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip
[PascalVOC]: http://host.robots.ox.ac.uk/pascal/VOC/
[COCOLVIS_annotation]: https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/cocolvis_annotation.tar.gz
[COCO_MVal]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/COCO_MVal.zip

====================================================

Email: bingqiangzhou@qq.com (Bingqiang Zhou)
