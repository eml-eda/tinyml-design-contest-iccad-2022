# 2022 ACM/IEEE TinyML Design Contest at ICCAD - M.A.D. AI @ Politecnico di Torino 

## What's in this repository?

This repository contains four folders:
1. `python code`: folder with the pyhton training code for the proposed classifier.
2. `c_code_l432kc`: folder with the c code generated using STM32CubeMX with X-Cube-AI and Keil uVision for the nucleo-l432kc board.
3. `c_code_l4a6zg`: folder with the c code generated using STM32CubeMX with X-Cube-AI and Keil uVision for the nucleo-l4a6zg board.
4. `trained_model`: folder with the trained and deployed onnx model.

## How do I run python code?

** **

**The code has been tested on Ubuntu 18.04.06 LTS with Bash 4.4 Shell**

You can run the code contained in `python_code` by first installing the python requirements:

    pip install requirements.txt

Then, downloading data:

    chmod +x download_data.sh
    ./download_data.sh

​	**N.B., you may require to install `unrar` within your system (e.g., `sudo apt install unrar`) to run the `download_data.sh` script**

and then, running:

    python train.py DiscIEGMNet_cbr --epoch 30 --lr 1e-2 --augment

The outputs of this execution will be saved in the path: `saved_models/DiscIEGMNet_cbr/<timestamp>`, where `<timestamp>` contains information about the date and time of when the code is executed.
In particular, the trained model's weights will be saved as `model.pkl` within this path.

## How do I deploy the model on the board?

You can firstly convert the model to onnx format by running

    python pkl2onnx.py DiscIEGMNet_cbr --model-path <path-where-model.pkl-is-stored>

Once we obtain the onnx model file, we could deploy the model on the board by following the provided instructions using XCube-AI and Keil uVision.

**N.B., we propose two different C code folders (`c_code_l432kc` and `c_code_l4a6zg`) because we tested our code on a slightly different nucleo-L4 board. The main difference between the two implementations regards the UART pheripheral. In the evaluation phase please refer to the contained in `c_code_l432kc`.**