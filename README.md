# SegMaR
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
  pip install -v -e .```
### 2. Downloading Training and Testing Datasets
> The discriminative mask will be released soon.

- Downloading training dataset (COD10K-train) and move it into `./OurModule/datasets/train/`.

- Downloading testing dataset (COD10K-test + CAMO-test + CHAMELEON) and move it into `./OurModule/datasets/test/`.

### 3. Training Configuration
- After you download all the training datasets, just run `./OurModule/train.py` to generate the model (you can replace discriminative mask with binary groundtruth is necessary).

- For iterative training: `generator.load_state_dict(torch.load('./OurModule/models/xxx.pth'))`.
