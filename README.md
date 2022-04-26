# Learning Interactive Multi-Object Segmentation

## 1. Paper

**"Yan Gui, Bingqing Zhou, Jianming Zhang, Cheng Sun, Lingyun Xiang, Jin Zhang. Learning Interactive Multi-Object Segmentation through Appearance Embedding and Spatial Attention, Submitted to IET Image Processing, 2021."**

Overall architecture of Interactive Multi-Object Segmentation Network, as shown below.

![example](fig1.png)

experimental result data (.xlsx) in our paper：[experimental_results.zip](https://github.com/BingqiangZhou/Learning-Interactive-Multi-Object-Segmentation/releases/tag/experimental-results)

## 2. Run Demo App

### 2.1 Download [model file](https://github.com/BingqiangZhou/Learning-Interactive-Multi-Object-Segmentation/releases/download/model/best_mean_iou_epoch.pkl), and put it to `models` folder.

### 2.2 config python env，install dependent packages and run deom .

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

====================================================

Email: bingqiangzhou@qq.com (Bingqiang Zhou)
