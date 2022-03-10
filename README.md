# Segment, Magnify and Reiterate Detecting Camouflaged Objects the Hard Way
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
> The discriminative mask will be released soon. Or run `./OurSampler/DiscriminativeMask.py` to generate your discriminative mask.

- Downloading training dataset (COD10K-train) and move it into `./OurModule/datasets/train/`.

- Downloading testing dataset (COD10K-test + CAMO-test + CHAMELEON) and move it into `./OurModule/datasets/test/`.

### 3. Training Configuration
- After you download all the training datasets, just run `./OurModule/train.py` to generate the model (you can replace discriminative mask with binary groundtruth if necessary).

- For iterative training: `generator.load_state_dict(torch.load('./OurModule/models/xxx.pth'))`.

### 4. Testing Configuration
- After you download all the pre-trained model and testing datasets, just run `./OurModule/test.py` to generate the prediction map. Your save directory is `./OurModule/results.py`.

### 5. Sampler Operation
- Make sure that you have installed MobulaOP in your virtual environment.

- For sampler operation, just run `./OurSampler/Sampler_Distort.py`.

- For restoration operation, just run `./OurSampler/Sampler_Restort.py`.

- For the directory of original prediction or restoration prediction, please see our codes details.

### 6. Evaluation

- One-key evaluation is written in [MATLAB](https://github.com/DengPingFan/CODToolbox) code, please follow this the instructions in` main.m` and just run it to generate the evaluation results.
