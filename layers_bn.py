__author__ = 'krishnch'

import tensorflow as tf
import numpy as np
import math

class layersObj:

    #define functions like conv, deconv, upsample, dropout etc inside this class

    def __init__(self):
        print('init')
        self.batch_size=20
        #self.batch_size=cfg.batch_size
        #self.

    def conv2d_layer(self,ip_layer,  # The previous layer.
                 name,               # Name of the conv layer
                 #num_input_channels, # Num. channels in prev. layer.
                 bias_init=0,          # Constant bias value for initialization
                 kernel_size=(3,3),  # Width and height of each filter.
                 strides=(1,1),      # stride value of the filter
                 num_filters=32,     # Number of output filters.
                 padding='SAME',     # Padding - SAME for zero padding - i/p and o/p of conv. have same dimensions
                 use_bias=True,      #to use bias or not
                 use_relu=True,      # Use relu as activation function
                 use_batch_norm=False,# use batch norm on layer before passing to activation function
                 use_conv_stride=False,  # Use 2x2 max-pooling - obtained by convolution with stride 2.
                 training_phase=True,   # Training Phase
                 scope_name=None,     #scope name for batch norm
                 acti_type='xavier',  #weight and bias variable initializer type
                 dilated_conv=False,  #dilated convolution enbale/disable
                 dilation_factor=1):  #dilation factor
        '''
        Standard 2D convolutional layer
        '''
        prev_layer_no_filters = ip_layer.get_shape().as_list()[-1]

        weight_shape = [kernel_size[0], kernel_size[1], prev_layer_no_filters, num_filters]
        bias_shape = [num_filters]

        strides_augm = [1, strides[0], strides[1], 1]

        if(scope_name==None):
            scope_name=str(name)+'_bn'

        with tf.variable_scope(name):

            weights = self.get_weight_variable(weight_shape, name='W',acti_type=acti_type)
            if(use_bias==True):
                biases = self.get_bias_variable(bias_shape, name='b', init_bias_val=bias_init)

            if(dilated_conv==True):
                op_layer = tf.nn.atrous_conv2d(ip_layer, filters=weights, rate=dilation_factor, padding=padding, name=name)
            else:
                if(use_conv_stride==False):
                    op_layer = tf.nn.conv2d(ip_layer, filter=weights, strides=strides_augm, padding=padding)
                else:
                    op_layer = tf.nn.conv2d(input=ip_layer, filter=weights, strides=[1, 2, 2, 1], padding=padding)

            #Add bias
            if(use_bias==True):
                op_layer = tf.nn.bias_add(op_layer, biases)

            if(use_batch_norm==True):
                op_layer = self.batch_norm_layer(ip_layer=op_layer,name=scope_name,training=training_phase)
            if(use_relu==True):
                op_layer = tf.nn.relu(op_layer)

            # Add Tensorboard summaries
            #_add_summaries(op_layer, weights, biases)

        #return op_layer,weights,biases
        return op_layer

    def deconv2d_layer(self,ip_layer,         # The previous layer.
                 name,               # Name of the conv layer
                 #num_input_channels, # Num. channels in prev. layer.
                 bias_init=0,           # Constant bias value for initialization
                 kernel_size=(3,3),  # Width and height of each filter.
                 strides=(2,2),      # stride value of the filter
                 num_filters=32,     # Number of filters.
                 padding='SAME',     # Padding - SAME for zero padding - i/p and o/p of conv. have same dimensions
                 output_shape=None,  # output shape of deconv. layer
                 use_bias=True,      #to use bias or not
                 use_relu=True,      # Use relu as activation function
                 use_batch_norm=False,# use batch norm on layer before passing to activation function
                 #use_conv_stride=False, # Use 2x2 max-pooling - obtained by convolution with stride 2.
                 training_phase=True,   # Training Phase
                 scope_name=None,       #scope name for batch norm
                 acti_type='xavier',dim_list=None):   #scope name for batch norm

        '''
        Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a factor of 2.
        '''
        if(dim_list==None):
            prev_layer_shape = ip_layer.get_shape().as_list()
        else:
            prev_layer_shape = dim_list
        batch_size_val=tf.shape(ip_layer)[0]
        #print("0 prev layer shape",batch_size_val,prev_layer_shape[0],prev_layer_shape[1],prev_layer_shape[2])
        if output_shape is None:
            #output_shape = tf.stack([prev_layer_shape[0], prev_layer_shape[1]*strides[0], prev_layer_shape[2]*strides[1], num_filters])
            #output_shape = [batch_size_val, prev_layer_shape[1]*strides[0], prev_layer_shape[2]*strides[1], num_filters]
            output_shape = tf.stack([tf.shape(ip_layer)[0], tf.shape(ip_layer)[1] * strides[0], tf.shape(ip_layer)[2] * strides[1], num_filters])

        prev_layer_no_filters = prev_layer_shape[3]
        #print("prev layer shape",output_shape,prev_layer_shape[3])

        weight_shape = [kernel_size[0], kernel_size[1], num_filters, prev_layer_no_filters]
        bias_shape = [num_filters]
        strides_augm = [1, strides[0], strides[1], 1]

        if(scope_name==None):
            scope_name=str(name)+'_bn'

        with tf.variable_scope(name):

            weights = self.get_weight_variable(weight_shape, name='W',acti_type=acti_type)
            if(use_bias==True):
                biases = self.get_bias_variable(bias_shape, name='b', init_bias_val=bias_init)

            op_layer = tf.nn.conv2d_transpose(ip_layer,
                                        filter=weights,
                                        output_shape=output_shape,
                                        strides=strides_augm,
                                        padding=padding)

            #Add bias
            if(use_bias==True):
                op_layer = tf.nn.bias_add(op_layer, biases)

            if(use_batch_norm==True):
                op_layer = self.batch_norm_layer(ip_layer=op_layer,name=scope_name,training=training_phase)
            if(use_relu==True):
                op_layer = tf.nn.relu(op_layer)

            # Add Tensorboard summaries
            #_add_summaries(op_layer, weights, biases)

        #return op_layer,weights,biases
        return op_layer

    def lrelu(self, x, leak=0.2, name='lrelu'):
        return tf.maximum(x, leak*x)
#################################################################################
    # Attention unit layers
#################################################################################
    #Attention layer
    def attn_layer(self, ip_layer, ch, sn=False, scope='attention', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            f = self.conv(ip_layer, ch // 8, kernel=1, stride=1, sn=sn, scope='f_conv') # [bs, h, w, c']
            g = self.conv(ip_layer, ch // 8, kernel=1, stride=1, sn=sn, scope='g_conv') # [bs, h, w, c']
            h = self.conv(ip_layer, ch, kernel=1, stride=1, sn=sn, scope='h_conv') # [bs, h, w, c]

            #print('ip layer',ip_layer,f,g,h)
            # N = h * w
            s = tf.matmul(self.hw_flatten(g), self.hw_flatten(f), transpose_b=True) # # [bs, N, N]
            #print('s',s)
            beta = tf.nn.softmax(s, axis=-1)  # attention map
            #print('beta',beta)
            o = tf.matmul(beta, self.hw_flatten(h)) # [bs, N, C]
            #print('o',o)
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            #o = tf.reshape(o, shape=ip_layer.shape) # [bs, h, w, C]
            o = tf.reshape(o, shape=[self.batch_size,ip_layer.shape[1],ip_layer.shape[2],ip_layer.shape[3]]) # [bs, h, w, C]
            #print('re o',o)
            ip_layer = gamma * o + ip_layer

        return ip_layer

    # conv layer for attention layer
    #def conv(self, x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='attn_conv_0'):
    def conv(self, x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='attn_conv_0'):
        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        weight_regularizer = None
        with tf.variable_scope(scope):
            # Removed this from orig code - want VALID padding
            #if pad_type == 'zero' :
            #    x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
            #if pad_type == 'reflect' :
            #    x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

            if sn :
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],\
                                    initializer=weight_init, regularizer=weight_regularizer)
                x = tf.nn.conv2d(input=x, filter=self.spectral_norm(w), strides=[1, stride, stride, 1], padding='SAME')
                if use_bias :
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                    x = tf.nn.bias_add(x, bias)

            else :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=weight_init,\
                                     kernel_regularizer=weight_regularizer, strides=stride, use_bias=use_bias)
        return x

    # flatten layer for attention layer
    def hw_flatten(self,x):
        #if(x.shape[0]==None):
        bs=self.batch_size
        return tf.reshape(x, shape=[bs, -1, x.shape[-1]])

    # spectral norm layer
    def spectral_norm(self, w, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = self.l2_norm(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = self.l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm

    # L2 norm layer
    def l2_norm(self, v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

#################################################################################

    def dropout_conv_layer(self,ip_layer, name, training, keep_prob=0.5):
        '''
        Performs dropout on the activations of an input
        '''
        return tf.nn.dropout(ip_layer, keep_prob=keep_prob, name=name, seed=1)

    def upsample_layer(self,ip_layer, method=0, scale_factor=2, dim_list=None):
        '''
        2D upsampling layer with default image scale factor of 2.
        method = 0 --> Bilinear Interpolation
                 1 --> Nearest Neighbour
                 2 --> Bicubic Interpolation
        '''
        if(dim_list!=None):
            prev_height = dim_list[1]
            prev_width = dim_list[2]
        else:
            prev_height = ip_layer.get_shape().as_list()[1]
            prev_width = ip_layer.get_shape().as_list()[2]
        #print("prev dims",prev_height,prev_width)
        new_height = int(round(prev_height * scale_factor))
        new_width = int(round(prev_width * scale_factor))
        #print("new dims",new_height,new_width)
        op = tf.image.resize_images(images=ip_layer,size=[new_height,new_width],method=method)

        return op

    def max_pool_layer2d(self,ip_layer, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name=None):
        '''
        2D max pooling layer with standard 2x2 pooling with stride 2 as default
        '''

        kernel_size_aug = [1, kernel_size[0], kernel_size[1], 1]
        strides_aug = [1, strides[0], strides[1], 1]

        op = tf.nn.max_pool(ip_layer, ksize=kernel_size_aug, strides=strides_aug, padding=padding, name=name)

        return op

    #exp1
    def batch_norm_layer(self,ip_layer, name, training, moving_average_decay=0.99, epsilon=1e-3):
        '''
        Batch normalisation layer (Adapted from https://github.com/tensorflow/tensorflow/issues/1122)
        :param ip_layer: Input layer (should be before activation)
        :param name: A name for the computational graph
        :param training: A tf.bool specifying if the layer is executed at training or testing time
        :return: Batch normalised activation
        '''

        with tf.variable_scope(name):

            n_out = ip_layer.get_shape().as_list()[-1]
            tensor_dim = len(ip_layer.get_shape().as_list())

            if tensor_dim == 2:
                # must be a dense layer
                moments_over_axes = [0]
            elif tensor_dim == 4:
                # must be a 2D conv layer
                moments_over_axes = [0, 1, 2]
            elif tensor_dim == 5:
                # must be a 3D conv layer
                moments_over_axes = [0, 1, 2, 3]
            else:
                # is not likely to be something reasonable
                raise ValueError('Tensor dim %d is not supported by this batch_norm layer' % tensor_dim)

            init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
            init_gamma = tf.constant(1.0, shape=[n_out], dtype=tf.float32)
            beta = tf.get_variable(name='beta', dtype=tf.float32, initializer=init_beta, regularizer=None,
                                   trainable=True)
            gamma = tf.get_variable(name='gamma', dtype=tf.float32, initializer=init_gamma, regularizer=None,
                                    trainable=True)

            batch_mean, batch_var = tf.nn.moments(ip_layer, moments_over_axes, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=moving_average_decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(training, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normalised = tf.nn.batch_normalization(ip_layer, mean, var, beta, gamma, epsilon)

        return normalised


    ### VARIABLE INITIALISERS ####################################################################################

    def get_weight_variable(self,shape,name=None,acti_type='xavier',fs=3):
        """
        Initializes the weights/convolutional kernels based on Xavier's method. Dimensions of filters are determined as per the input shape. Xavier's method initializes values of filters randomly around zero with the standard deviation computed as per Xavier's method.
        Args:
            shape : provides the shape of convolutional filter.
                    shape[0], shape[1] denotes the dimensions (height and width) of filters.
                    shape[2] denotes the depth of each filter. This also denotes the depth of each feature.
                    shape[3] denotes the number of filters which will determine the number of features that have to be computed.
        Returns:
            Weights/convolutional filters initialized as per Xavier's method. The dimensions of the filters are set as per input shape variable.
        """
        nInputUnits=shape[0]*shape[1]*shape[2]
        stddev_val = 1. / math.sqrt( nInputUnits/2 )
        #http://cs231n.github.io/neural-networks-2/#init
        #print("w init",nInputUnits,stddev_val)
        #return tf.Variable(tf.truncated_normal(shape, stddev=stddev_val, seed=1))
        if(acti_type=='identity_std0' or acti_type=='identity_stdnz'):
            #arr=np.identity(shape[0])#,dtype=float32)
            #init_arr=np.tile(arr[:,:,np.newaxis,np.newaxis],[1,1,shape[2],shape[3]])
            init_arr=np.zeros((shape[0],shape[1],shape[2],shape[3]))
            for index in range(shape[2]):
                init_arr[:,:,index,index]=np.identity(shape[0])
            #print("0",arr.shape,init_arr.shape)
        elif(acti_type=='centerone_std0' or acti_type=='centerone_stdnz'):
            arr=np.zeros((shape[0],shape[1]))
            arr[int(shape[0]/2),int(shape[1]/2)]=1
            #arr[1,1]=1
            #init_arr=np.tile(arr[:,:,np.newaxis,np.newaxis],[1,1,shape[2],shape[3]])
            init_arr=np.zeros((shape[0],shape[1],shape[2],shape[3]))
            for index in range(shape[2]):
                init_arr[:,:,index,index]=arr
            #print("0",arr.shape,init_arr.shape)
        #print("kc",shape)#,arr.shape,init_arr.shape)
        if(acti_type=='xavier'):
            return tf.Variable(tf.random_normal(shape, stddev=stddev_val, seed=1),name=name)
        elif(acti_type=='identity_std0' or acti_type=='centerone_std0'):
            #return tf.Variable(tf.random_normal(shape, mean=1.0, stddev=0.0, seed=1,name=name))
            return tf.Variable(initial_value=init_arr,name=name,dtype=tf.float32)#+tf.random_normal(shape, mean=0.0, stddev=0.0, seed=1,name=name))
        elif(acti_type=='identity_stdnz' or acti_type=='centerone_stdnz'):
            #return tf.Variable(tf.random_normal(shape, mean=1.0, stddev=stddev_val, seed=1,name=name))
            return tf.Variable(initial_value=init_arr,name=name,dtype=tf.float32)+tf.random_normal(shape, mean=0.0, stddev=stddev_val, seed=1)#,name=name)
        #elif(acti_type=='centerone_std0'):
        #    a = tf.zeros(shape)+[[0,0,0],[0,1,0],[0,0,0]]
        #    return tf.Variable(a,name=name)
        #elif(acti_type=='centerone_stdnz'):
        #    a = tf.zeros(shape)+[[0,0,0],[0,1,0],[0,0,0]]
        #    return tf.Variable(a,name=name)+tf.random_normal(shape, mean=0.0, stddev=stddev_val, seed=1)#,name=name)


        #return tf.Variable(tf.random_uniform(shape, minval=0, maxval=2*stddev_val, seed=1))

    def get_bias_variable(self,shape, name=None, init_bias_val=0.0):
        """
        Initializes the biases as per input bias_val. The initial value is equal to zero + bias_val. No of such bias values required are determined by the number of filters which is input as length variable.
        Args:
            shape : provides us the number of filters.
            init_bias_val : provides the base bias_val to initialize all bias values with.
        Returns:
            biases initilialized as per the bias_val.
        """
        return tf.Variable(tf.zeros(shape=shape)+init_bias_val,name=name)

