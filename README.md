<p align="center">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/CyFeng16/MVIMP" />
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/CyFeng16/MVIMP" />
    <img alt="GitHub License" src="https://img.shields.io/github/license/cyfeng16/MVIMP" />
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg" />
</p>

# MVIMP

**M**ixed **V**ideo and **I**mage **M**anipulation **P**rogram

I realize that training a good-performance AI model is kind of just one side of the story, make it easy to use for others is the other thing. So, this repository tries to embrace out-of-the-box AI ability to manipulate multimedia, also, I wish you have fun!

[中文文档请移步](https://cyfeng.science/2020/05/05/introduce-to-MVIMP/)

| Parallel |  Input | Output |        Parallel        |
|:--------:|:------:|:------:|:----------------------:|
| AnimeGAN | Images | Images |          True          |
|   DAIN   |  Video |  Video |          False         |
|  Photo3D | Images | Videos | True(not recommmended) |
| DeOldify | Images | Images |          True          |

## AnimeGAN

Original repository: [TachibanaYoshino/AnimeGAN](https://github.com/TachibanaYoshino/AnimeGAN)

This is the Open source of the paper <AnimeGAN: a novel lightweight GAN for photo animation>, which uses the GAN framwork to transform real-world photos into anime images.

Requirements:
- TensorFLow 1.15.2
- CUDA 10.0(tested locally) / 10.1(colab)
- Python 3.6.8(3.6+/3.7+/3.8+)
- opencv
- tqdm
- numpy
- glob
- argparse

Usage:

1. `Local`

    ```shell
    # Step 1: Prepare
    git clone https://github.com/CyFeng16/MVIMP.git
    cd MVIMP
    python3 preparation.py -f animegan 
    # Step 2: Put your photos into ./Data/Input/
    # Step 3: Infernece
    python3 inference_animegan.py
    ```

2. `Colab`

    Or you can try following shared colab in playground mode:

    https://colab.research.google.com/drive/1bpwUFcr5i38_P3a0r3Qm9Dvkl-MS_Y1y?usp=sharing

## Photo3D

Original repository: [vt-vl-lab/3d-photo-inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting)

a method for converting a single RGB-D input image into a 3D photo, i.e., a multi-layer representation for novel view synthesis that contains hallucinated color and depth structures in regions occluded in the original view. 

Requirements:
- PyTroch 1.5.0
- CUDA 10.1(tested locally/colab)
- Python 3.6.8(3.6+/3.7+/3.8+)
- Other Python dependencies listed in requirements.txt (will be auto prepared through running `preparation.py`)

Usage:

1. `Local`

    ```shell
    # Step 1: Prepare
    git clone https://github.com/CyFeng16/MVIMP.git
    cd MVIMP
    python3 preparation.py -f photo3d
    # Step 2: Put your photos into ./Data/Input/
    # Step 3: Infernece
    python3 inference_photo3d.py -f 40 -n 240 -l 960
    ```

2. `Colab`

    Or you can try following shared colab in playground mode:

    https://colab.research.google.com/drive/1VAFCN8Wh4DAY_HDcwI-miNIBomx_MZc5?usp=sharing

    P.S. Massive memory is occupied during operation(grows with `-l`). `Higher memory` runtime helps if you are Colab Pro user.

3. Description of Parameters

   - `--fps`or`-f`: setup the FPS of output video.
   - `--frames`or`-n`: setup the number of frames of output video.
   - `--longer_side_len`or`-l`: set the longer side of output video(either height or width).

## DAIN

Original repository: [baowenbo/DAIN](https://github.com/baowenbo/DAIN)

Depth-Aware video frame INterpolation (DAIN) model explicitly detect the occlusion by exploring the depth cue. We develop a depth-aware flow projection layer to synthesize intermediate flows that preferably sample closer objects than farther ones. 

Requirements:
- FFmpeg
- PyTroch 1.4.0
- CUDA 10.0(tested locally/colab)
- Python 3.6.8(3.6+/3.7+/3.8+)
- GCC 7.5 (Compiling PyTorch 1.4.0 extension files (.c/.cu))

P.S. Make sure your virtual env has torch-1.4.0+cu100 and torchvision-0.5.0+cu100.
You can use the following [command](https://github.com/baowenbo/DAIN/issues/44#issuecomment-624025613):

```shell
# Install PyTorch 1.4.0 with CUDA 10.0
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy==1.1.0
# Then set the softlink to CUDA 10.0
sudo ln -snf /usr/local/cuda-10.0 /usr/local/cuda
# After that we can perform a complete compilation.
```

Usage:

1. `Local`

    ```shell
    # Step 1: Prepare
    git clone https://github.com/CyFeng16/MVIMP.git
    cd MVIMP
    python3 preparation.py -f dain
    # Step 2: Put a single video file into ./Data/Input/
    # Step 3: Infernece
    python3 inference_dain.py -input your_input.mp4 -ts 0.5 -hr False
    ```

2. `Colab`

    Or you can try following shared colab in playground mode:

    https://colab.research.google.com/drive/1pIPHQAu7z4Z3LXztCUXiDyBaIlOqy4Me?usp=sharing

3. Description of Parameters

   - `--input_video`or`-input`: set the input video name.
   - `--time_step`or`-ts`: set the frame multiplier, 0.5 corresponds to 2X, 0.25 corresponds to 4X, and 0.125 corresponds to 8X.
   - `--high_resolution`or`-hr`: Default is False. Pascal V100 has not enough memory to run DAIN for the FHD video, set `-hr` True to split a single frame into four blocks and process them separately in order to reduce GPU memory usage.

## DeOldify

Original repository: [jantic/DeOldify](https://github.com/jantic/DeOldify)

DeOldify is a Deep Learning based project for colorizing and restoring old images and video! We currently try the easiest way to colorize images using DeOldify, which is using SaaS service provided by DeepAI(**For Now**). You must sign up DeepAI.

Requirements:
- None

Usage:

1. `Local`

    ```shell
    # Step 1: Prepare
    git clone https://github.com/CyFeng16/MVIMP.git
    cd MVIMP
    python3 preparation.py
    # Step 2: Infernece
    python3 inference_deoldify.py -key quickstart-QUdJIGlzIGNvbWluZy4uLi4K
    # This trial code (`quickstart-QUdJIGlzIGNvbWluZy4uLi4K`) will be invalid after a few times. If you wanna keep doing this, sign up a DeepAI account and you will get $5 credit.
    ```

2. Description of Parameters

   - `--api_key`or`-key`: the API key of DeepAI, please sign up first.

# TODO
- [x] Chinese Document
- [x] DeOldify for colorizing and restoring old images and videos
- [x] tqdm instead of print loop
- [ ] Original DeOldify local as well as Colab
- [ ] Dockerized deployment.
- [ ] https://roxanneluo.github.io/Consistent-Video-Depth-Estimation/
- [ ] MMSR for image and video super-resolution

You are welcomed to discuss future features in [this issue](https://github.com/CyFeng16/MVIMP/issues/2).

# Acknowledgment

This code is based on the [TachibanaYoshino/AnimeGAN](https://github.com/TachibanaYoshino/AnimeGAN), [vt-vl-lab/3d-photo-inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting), [baowenbo/DAIN](https://github.com/baowenbo/DAIN) and [jantic/DeOldify](https://github.com/jantic/DeOldify). Thanks to the contributors of those project.