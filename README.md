# IMC_innopam

## Dataset Directory

- **NAS4**
  - Path: `/volume1/1_InternalCompany/Image_matching_challenge/image-matching-challenge-2023/train/`

## Pretrained Model Directory

- **NAS4**
  - Path: `/volume1/1_InternalCompany/Image_matching_challenge/weights/`

## Prerequisites

- This project uses Python 3.8. Make sure your python version is 3.8. If not, you can download it from [here](https://www.python.org/downloads/release/python-380/).

## Installation

**Clone the repository**
```bash
git clone https://github.com/innopam/IMC_innopam.git
```
**Move into the cloned directory**
```
cd IMC_innopam
```
**Install the required dependencies:**
```
pip install -r requirements.txt
```


**Check before you start:**

Before starting, set the path of ```config.py``` and ```mAA_eval.py``` to your local environment path.

You can use multi-resolution and model ensemble using ```config.py```. 

You can also experiment by adjusting each parameter.

**Inference:**
```
cd submit

python3 submission_innopam.py
```

**mAA eval:**
```
cd module

python3 mAA_eval.py
```
