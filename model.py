from os import name
from numpy.lib.arraypad import pad
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.python.keras.backend import shape


class Mobilenet_V2():
    def __init__(self, *, inp_shape = (224,224,3), rho = 1.0 , alpha = 1.0, expansion = 6.0, classes = 2, droppout = 0.0):
        assert alpha > 0 and alpha <= 1 ,'Error, my Mobilenet_V2 can only accept  alpha > 0 and alpha <= 1'
        assert rho > 0 and rho <= 1 ,'Error, my Mobilenet_V2 can only accept  rho > 0 and rho <= 1'
        self._inp_shape = inp_shape
        self._rho = rho
        self._alpha = alpha
        self._expansion = expansion
        self._classes = classes
        self._droppout = droppout
    def _depthwiseconv(self, *, strides: int):
        return models.Sequential([
            DepthwiseConv2D(kernel_size= (3,3), strides= strides, padding= 'same' if strides == 1 else 'valid', use_bias= False),
            BatchNormalization(),
            ReLU(max_value= 6.)
        ])
    def _pointwiseconv(self, *, filters: int, linear: bool):
        layer = models.Sequential([
            Conv2D(filters= int(filters * self._alpha), kernel_size= (1,1), strides= (1,1), padding= 'same', use_bias= False),
            BatchNormalization(),
        ])
        if linear == False:
            layer.add(ReLU(max_value= 6.))
        return layer
    def _standardconv(self):
        return models.Sequential([
            Conv2D(filters= 32, kernel_size= (3,3), strides= (2,2), use_bias= False),
            BatchNormalization(),
            ReLU(max_value= 6.)
        ])
    def _inverted_residual_block_(self, x, *, strides_depthwise: int, filter_pointwise: int, expansion: int):
        filter = int(filter_pointwise * self._alpha)
        fx = self._pointwiseconv(filters= filter * expansion, linear= False)(x)
        fx = self._depthwiseconv(strides= strides_depthwise)(fx)
        fx = self._pointwiseconv(filters= filter , linear= True)(fx)
        if strides_depthwise == 1 and x.shape[-1] == filter_pointwise:
            return add([fx,x])
        else:
            return fx
    def _bottleneck_block_(self, x, *,  s: int, c: int, t: int, n: int):
        '''
            s : strides
            c : output channels
            t : expansion factor
            n : repeat
        '''
        x = self._inverted_residual_block_(x, strides_depthwise= s, filter_pointwise= c, expansion= t)
        for i in range(n-1):
            x = self._inverted_residual_block_(x, strides_depthwise= 1, filter_pointwise= c, expansion= t)
        return x 
    def build(self):
        feature_map = int(self._rho * self._inp_shape[0])
        img_inp = Input(shape= (feature_map, feature_map,3))
        # standardconv 
        x = self._standardconv()(img_inp)
        # block bottleneck 1
        x = self._bottleneck_block_(x, s= 1, c= 16, t= 1, n= 1)
        # block bottleneck 2
        x = self._bottleneck_block_(x, s= 2, c= 24, t= self._expansion, n= 2)
        # block bottleneck 3
        x = self._bottleneck_block_(x, s= 2, c= 32, t= self._expansion, n= 3)
        # block bottleneck 4
        x = self._bottleneck_block_(x, s= 2, c= 64, t= self._expansion, n= 4)
        # block bottleneck 5
        x = self._bottleneck_block_(x, s= 1, c= 96, t= self._expansion, n= 3)
        # block bottleneck 6
        x = self._bottleneck_block_(x, s= 2, c= 160, t= self._expansion, n= 3)
        # block bottleneck 7
        x = self._bottleneck_block_(x, s= 1, c= 320, t= self._expansion, n= 1)
        # conv2d 1x1 
        x = self._pointwiseconv(filters= 1280, linear= False)(x)
        # fully connect
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self._droppout)(x)
        x = Dense(self._classes, activation='softmax')(x)
        return models.Model(img_inp, x, name= 'mobilenetv2')


