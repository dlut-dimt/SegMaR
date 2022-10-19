# Segment, Magnify and Reiterate Detecting Camouflaged Objects the Hard Way (CVPR2022)

![image](https://github.com/YAOSL98/Segment-Magnify-and-Reiterate-Detecting-Camouflaged-Objects-the-Hard-Way/blob/main/Images/overview.jpg)


Segment, Magnify and Reiterate: Detecting Camouflaged Objects the Hard Way. Jia Qi and Yao Shuilian and Liu Yu and Fan Xin and Liu Risheng and Luo Zhongxuan. CVPR2022.

[paper download](https://openaccess.thecvf.com/content/CVPR2022/papers/Jia_Segment_Magnify_and_Reiterate_Detecting_Camouflaged_Objects_the_Hard_Way_CVPR_2022_paper.pdf)

## Usage
> The training and testing experiments are conducted using PyTorch with a single Tesla V100 GPU of 36 GB Memory.

### 1. Prerequisites
> Note that SegMaR is only tested on Ubuntu OS with the following environments. 

- Creating a virtual environment in terminal: `conda create -n SegMaR python=3.6`.

- Installing necessary packages: `pip install -r requirements.txt`.

- Installing NVIDIA-Apex (Under CUDA-10.0 and Cudnn-7.4).

- Installing [MobulaOP](https://github.com/wkcn/mobulaop) for Sampler operation.
  ```
  # Clone the project
  git clone https://github.com/wkcn/MobulaOP
  
  # Enter the directory
  cd MobulaOP
  
  # Install MobulaOP
  pip install -v -e .

### 2. Downloading Training and Testing Datasets
> You can use [discriminative mask](https://drive.google.com/file/d/1q3kTtBUu2WJB67V5S0wiqSr0--H1nwBC/view?usp=sharing) or run `./OurSampler/DiscriminativeMask.py` to generate your discriminative mask.

- Downloading training dataset (COD10K-train) and move it into `./OurModule/datasets/train/`.

- Downloading testing dataset (COD10K-test + CAMO-test + CHAMELEON) and move it into `./OurModule/datasets/test/`.

### 3. Training Configuration
- After you download all the training datasets, just run `./OurModule/train.py` to generate the model (you can replace discriminative mask with binary groundtruth if you use a dataset without this mask).

- For iterative training: `generator.load_state_dict(torch.load('./OurModule/models/xxx.pth'))`.

- For the first stage [pretrained model](https://pan.baidu.com/s/1JqwWxxCJAA6HgTeb6n1MeQ), code `y4v3`, or [google drive link](https://drive.google.com/file/d/1UIFZTeMETIg9ZendbHNWc39dnpNn-xWl/view?usp=sharing)

### 4. Testing Configuration
- After you download all the pre-trained model and testing datasets, just run `./OurModule/test.py` to generate the prediction map. Your save directory is `./OurModule/results.py`.

- [Test results](https://pan.baidu.com/s/1I1OqBvDahJpzPdG72h6QdA), code `pxu7`

- [New NC4K results](https://pan.baidu.com/s/1wXy3YKM-d5YNRf5mQ-4gHg), code `pdas`(for the 4th stage)
### 5. Sampler Operation
- Make sure that you have installed MobulaOP in your virtual environment.

- For sampler operation, just run `./OurSampler/Sampler_Distort.py`.

- For restoration operation, just run `./OurSampler/Sampler_Restort.py`.

- For the directory of original prediction or restoration prediction, please see our codes details.

### 6. Evaluation

- One-key evaluation is written in [MATLAB](https://github.com/DengPingFan/CODToolbox) code, please follow this the instructions in` main.m` and just run it to generate the evaluation results.

## Citation
```
@InProceedings{Jia_2022_CVPR,
    author    = {Jia, Qi and Yao, Shuilian and Liu, Yu and Fan, Xin and Liu, Risheng and Luo, Zhongxuan},
    title     = {Segment, Magnify and Reiterate: Detecting Camouflaged Objects the Hard Way},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4713-4722}
}
```
