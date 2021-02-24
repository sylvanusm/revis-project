## Object recognition and computer vision 2020/2021

This repo contains my work for the object recognition class project MVA 2020-2021

### Assignment 3: Image classification 

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

#### Training and validating your model
Download the zip file and place it in inputs/  
Before going forward open and run notebooks/Data_augmentation.ipynb

To train a model you can run the script:

```
main.py --model [model_name]
```
:
- By default the images are loaded and resized to 64x64 pixels and normalized to zero-mean and standard deviation of 1. See data.py for the `data_transforms`.
The best model will be automatically saved in outputs/

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_name] --weights [checkpoint_path]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.
