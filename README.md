# SegMaR
## Usage
> The training and testing experiments are conducted using PyTorch with a single Tesla V100 GPU of 36 GB Memory.
### Prerequisites
> Note that SegMaR is only tested on Ubuntu OS with the following environments. 
- Creating a virtual environment in terminal: `conda create -n SegMaR python=3.6`.
- Installing necessary packages: `pip install -r requirements.txt`.
- Installing NVIDIA-Apex (Under CUDA-10.0 and Cudnn-7.4).
- Installing [MobulaOP](https://github.com/wkcn/mobulaop)
```
# Clone the project
git clone https://github.com/wkcn/MobulaOP

# Enter the directory
cd MobulaOP

# Install MobulaOP
pip install -v -e .
