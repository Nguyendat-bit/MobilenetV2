# MobilenetV2
My project is using the TensorFlow framework to implement the [MobilenetV2 model](https://arxiv.org/abs/1801.04381v4) to classify flower images. Give me a star :star2: if you like this repo.

### Model Architecture
The MobileNetV2 network architecture is shown below. The image below is excerpted from the author's original article
<p align = "center"> 
<img src = "image\mobilenetv2_architecture.jpg">
</p>

MobileNet V2 still uses depthwise separable convolutions, but its main building block now looks like this:
<p align = "center"><image src = "image\ResidualBlock.png"> </p>

There are two type of Convolution layers in MobileNet V2 architecture:
- 1x1 Convolution
- 3x3 Depthwise Convolution

These are the `two different components` in MobileNet V2 model:
<p align = "center"><image src = "image\MobileNetV2-6-uses-a-3-3-convolution-for-the-depth-wise-phase-of-the-convolution-and.png"></p>

`Stride 1` Block:
- Input
- 1x1 Convolution with Relu6
- Depthwise Convolution with Relu6
- 1x1 Convolution without any linearity
- Add

`Stride 2` Block:
- Input
- 1x1 Convolution with Relu6
- Depthwise Convolution with stride=2 and Relu6
- 1x1 Convolution without any linearity

### Author
<ul>
    <li>Github: <a href = "https://github.com/Nguyendat-bit">Nguyendat-bit</a> </li>
    <li>Email: <a href = "nduc0231@gmai.com">nduc0231@gmail</a></li>
    <li>Facebook: <a href = "https://www.facebook.com/dat.ng48/">Nguyễn Đạt</a></li>
    <li>Linkedin: <a href = "https://www.linkedin.com/in/nguyendat4801">Đạt Nguyễn Tiến</a></li>
</ul>

## I.  Set up environment
- Step 1: Make sure you have installed Miniconda. If not yet, see the setup document <a href="https://docs.conda.io/en/latest/miniconda.html">here</a>


- Step 2: `cd` into `MobilenetV2` and use command line
```
conda env create -f environment.yml
```

- Step 3: Run conda environment using the command

```
conda activate MobilenetV2
``` 

## II.  Set up your dataset

<!-- - Guide user how to download your data and set the data pipeline  -->
1. Download the data:
- Download dataset [here](http://download.tensorflow.org/example_images/flower_photos.tgz)
2. Extract file and put folder ```train``` and ```validation``` to ```./data``` by using [splitfolders](https://pypi.org/project/split-folders/)
- train folder was used for the training process
- validation folder was used for validating training result after each epoch

This library use ImageDataGenerator API from Tensorflow 2.0 to load images. Make sure you have some understanding of how it works via [its document](https://keras.io/api/preprocessing/image/)
Structure of these folders in ```./data```

```
train/
...daisy/
......daisy0.jpg
......daisy1.jpg
...dandelion/
......dandelion0.jpg
......dandelion1.jpg
...roses/
......roses0.jpg
......roses1.jpg
...sunflowers/
......sunflowers0.jpg
......sunflowers1.jpg
...tulips/
......tulips0.jpg
......tulips1.jpg
```

```
validation/
...daisy/
......daisy2000.jpg
......daisy2001.jpg
...dandelion/
......dandelion2000.jpg
......dandelion2001.jpg
...roses/
......roses2000.jpg
......roses2001.jpg
...sunflowers/
......sunflowers2000.jpg
......sunflowers2001.jpg
...tulips/
......tulips2000.jpg
......tulips2001.jpg
```

## III. Train your model by running this command line

Review training on colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JioKrr2GkUIAFWXvPSdoSOeqfp9Sl8hs?usp=sharing)


Training script:


```python

python train.py --train-folder ${link_to_train_folder} --valid-folder ${link_to_valid_folder} --classes ${num_classes} --epochs ${epochs}

```


Example:

```python

python train.py  --train-folder ./data/train --valid-folder ./data/val --classes 5 --epochs 100 

``` 

There are some important arguments for the script you should consider when running it:

- `train-folder`: The folder of training data
- `valid-folder`: The folder of validation data
- `Mobilenetv2-save`: Where the model after training saved
- `classes`: The number of your problem classes.
- `batch-size`: The batch size of the dataset
- `lr`: The learning rate
- `droppout`: The droppout 
- `label-smoothing`: The label smoothing
- `expansion`: The expansion factor
- `image-size`: The image size of the dataset
- `alpha`: Width Multiplier. It was mentioned in the paper on [page 4 - Mobilenet V1](https://arxiv.org/pdf/1704.04861.pdf)
- `rho`: Resolution Multiplier, It was mentioned in the paper on [page 4 - Mobilenet V1](https://arxiv.org/pdf/1704.04861.pdf)
## IV. Predict Process
If you want to test your single image, please run this code:
```bash
python predict.py --test-file ${link_to_test_image}
```


## V. Result and Comparision


My implementation
```
Epoch 88/90
207/207 [==============================] - 78s 377ms/step - loss: 0.2962 - acc: 0.9082 - val_loss: 0.3822 - val_acc: 0.8726

Epoch 00088: val_acc did not improve from 0.88889
Epoch 89/90
207/207 [==============================] - 78s 376ms/step - loss: 0.2930 - acc: 0.9115 - val_loss: 0.3681 - val_acc: 0.8780

Epoch 00089: val_acc did not improve from 0.88889
Epoch 90/90
207/207 [==============================] - 78s 375ms/step - loss: 0.3002 - acc: 0.9103 - val_loss: 0.3803 - val_acc: 0.8862

Epoch 00090: val_acc did not improve from 0.88889
```



## VI. Feedback
If you meet any issues when using this library, please let us know via the issues submission tab.



