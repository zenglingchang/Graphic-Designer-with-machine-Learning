import os,sys,time
import numpy as np
from ImgDeal import *
import tensorflow as tf
from skimage import exposure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

#-------------------------- Args -----------------------
BN_EPSILON              = 0.005
L2_EPSILON              = 0.001
MOVING_AVERAGE_DECAY    = 0.9997
DEAFULT_LEARNING_RATE   = 0.005
BN_DECAY = MOVING_AVERAGE_DECAY
LEARNING_RATE_DECAY = 0.98
DRNETWORK_VARIABLES = "DRNETWORK_VARIABLES"
UPDATE_OPS_COLLECTION = "DRNETWORK_UPDATE_OPS"

#----------------------- Function ----------------------
def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, DRNETWORK_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)
                           
def GetCurTime():
    return time.strftime('%Y_%m_%d-%H_%M_%S',time.localtime(time.time()))
    
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
def unconv(x,W,Shape):
    return tf.nn.conv2d_transpose(x, W, Shape, strides=[1,1,1,1], padding='SAME')
    
def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')
                        
def max_pool_with_argmax(net,stride):
    _, mask = tf.nn.max_pool_with_argmax( net,ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],padding='SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME') 
    return net,mask

def un_max_pool(net,mask,stride):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret
    
def weight_variable( shape, stddev = 0.1):
    initial = tf.truncated_normal(shape, stddev = stddev)
    return initial

def bias_variable( shape , value = 0.1):
    initial = tf.constant(value, shape=shape)
    return initial
    
def leaky_relu(input):
    return tf.nn.leaky_relu(input, alpha=0.2)

def batch_normal(layer, is_training):
    return tf.layers.batch_normalization(layer, training=is_training)

def moments(input):
    return tf.nn.moments(input,axes=list(range(len(input.get_shape()))))
    
#-----------------------  NetWork Class ----------------------

class DRN:
    def __init__(self):
    
        self.lr = DEAFULT_LEARNING_RATE
        self._build_network()
        self.sess = tf.Session()
            
        if os.path.isfile(os.path.join(sys.path[0], r'my_net/checkpoint')):
            self._load_data()
       # self.sess.run(tf.global_variables_initializer())
        print('NetWork init!')
        
    def _load_data(self):
        Saver = tf.train.Saver()
        Saver.restore(self.sess, os.path.join(sys.path[0], r'my_net/save_net.ckpt'))
        print('Model restored.')
        
    def _write_data(self):
        '''
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        Saver = tf.train.Saver(var_list=var_list, max_to_keep=5)'''
        Saver=tf.train.Saver()
        save_path = Saver.save(self.sess, os.path.join(sys.path[0], r"my_net/save_net.ckpt"))
        print("Model saved in file: %s" % save_path)
        
    def _build_network(self):
        weight_decay = 0.1
        self.l2_loss_rate = L2_EPSILON
        self.LayerOutput = {}
        l2_reg = tf.contrib.layers.l2_regularizer(weight_decay) 
        self.keep_prob = tf.placeholder(tf.float32, name='Keep_Prob')   
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        global_steps = tf.Variable(0, trainable=False)
        
        #--------------------------------- Network Input----------------------------
        
        # feature network input
        self.Imgs = tf.placeholder(tf.float32, [None, None], name='Imgs')
        self.Labels = tf.placeholder(tf.float32, [None, 5], name='Labels')
        
        # re-design network input
        self.backgroud = tf.placeholder(tf.float32, [192*256*3], name='backgroud')
        self.ElementList = tf.placeholder(tf.float32, [None, 192*256*4], name='ElementList')
        self.Tags = tf.placeholder(tf.float32,[None, 5], name='Tags')
        
        # unpool and unconv network input
        self.GetFeatureIndex = tf.placeholder(tf.int64, name='FeatureIndex')   
        #--------------------------------- Re-Design Network ----------------------------
        def Conv(Input, shape, MaxPool = True):
            w = tf.get_variable('w', initializer=weight_variable(shape))
            output = conv2d(Input, w)
            output = leaky_relu(output)
            if MaxPool:
                output = max_pool_4x4(output)
            return output
            
        def FN(Input, Shape, UseDropOut = False):
            w = tf.get_variable('w', regularizer=l2_reg, initializer=weight_variable(Shape), collections=COLLECTIONS)
            b = tf.get_variable('b', regularizer=l2_reg, initializer=bias_variable([Shape[1]], 0.5), collections=COLLECTIONS)
            if UseDropOut:
                return tf.nn.dropout(leaky_relu(tf.matmul(Input, w) + b), self.keep_prob)
            else:
                return leaky_relu(tf.matmul(Input, w) + b)
            
        def ElementNetwork(Input):
            # input shape:[1,256,192,4], output shape: [1,256]
            Input = tf.reshape(Input, [-1,256,192,4])
            with tf.variable_scope('Element_feature_network'):
                #(NameScope, Shape, use_4x4_Pooling)
                for ConvLayer in [('conv1', [3,3,4,64], False), ('conv2', [3,3,64,64], True), ('conv3', [3,3,64,64], True), ('conv4', [3,3,64,64], True)]:
                    with tf.variable_scope(ConvLayer[0]) as variable_scope:
                        Input = Conv(Input, ConvLayer[1], ConvLayer[2])
                Input = tf.reshape(Input, [-1,3*4*64])
                with tf.variable_scope('Output') as variable_scope:
                    output = FN(Input, [3*4*64, 256],True)
            return output
            
        def BackgroundNetwork(Input):
            # input shape:[1,256,192,3], output shape: [1,256]
            Input = tf.reshape(Input, [-1,256,192,3])
            with tf.variable_scope('Background_feature_network'):
                #(NameScope, Shape, use_4x4_Pooling)
                for ConvLayer in [('conv1', [3,3,3,64], False), ('conv2', [3,3,64,64], True), ('conv3', [3,3,64,64], True), ('conv4', [3,3,64,64], True)]:
                    with tf.variable_scope(ConvLayer[0]) as variable_scope:
                        Input = Conv(Input, ConvLayer[1], ConvLayer[2])
                Input = tf.reshape(Input, [-1,3*4*64])
                with tf.variable_scope('Output') as variable_scope:
                    output = FN(Input, [3*4*64, 256],True)
            return output
            
        def LabelNetwork(Input):
            # input shape:[1,5], output shape: [1,64]
            Input = tf.reshape(Input, [-1,5])
            with tf.variable_scope('Label_segmantic_network'):
                for FnLayer in [('input', [5,64], False), ('Output', [64,64], True)]:
                    with tf.variable_scope(FnLayer[0]) as variable_scope:
                        Input = FN(Input, FnLayer[1],  FnLayer[2])
            return Input
        
        def DesignNetwork(Label, Element, Background):
            # input shape:[1,256],[1,256],[1,64], output shape: [1,4]
            input = tf.concat([Label, Element, Background], 1)
            with tf.variable_scope('Design_network'):
                for FnLayer in [('FN1', [576,1024], True), ('FN2', [1024,256], True), ('Output', [256,4], True)]:
                    with tf.variable_scope(FnLayer[0]) as variable_scope:
                        Input = FN(Input, FnLayer[1],  FnLayer[2])
                output = tf.math.sigmoid(Input)[0]
            return output
            
        def paste(backgroud, element, Design):
            _element = tf.image.resize_images( element, [tf.cast((tf.ceil(Design[0]*256)), tf.int32), tf.cast(tf.ceil(Design[1]*192), tf.int32)])
            print(_element.get_shape())
            ElementSize = tf.shape(_element)
            print(ElementSize)
            dy =  tf.cond(tf.cast(tf.ceil(Design[2]*256), tf.int32) + ElementSize[0] < 256, lambda: tf.cast((tf.ceil(Design[2]*256)), tf.int32) , lambda: 256 - ElementSize[0])
            dx = tf.cond(tf.cast(tf.ceil(Design[3]*192), tf.int32) + ElementSize[1] < 192, lambda: tf.cast(tf.ceil(Design[3]*192), tf.int32) ,lambda: 192 - ElementSize[1])
            print(dx.get_shape(),dy.get_shape())
            _element = tf.pad(_element,[[dy, 256 - dy - ElementSize[0]],[dx,192-dx-ElementSize[1]],[0,0]], "CONSTANT")
            mask = tf.tile(_element[:,:,2:3]/255, [1,1,3])
            backmask = 1 - mask
            _backgroud = tf.multiply(_element[:,:,0:3], mask) + tf.multiply(backgroud, backmask)
            return _backgroud
            
        
        def con(index, backgroud, ElementList, DesignList, Tag):
            return index<tf.shape(ElementList)[0]
        
        def body(index, backgroud, ElementList, DesignList, Tag):
            
            Design = DesignNetwork(LabelNetwork(Tag), ElementNetwork(ElementList[index]), BackgroundNetwork(backgroud))
            _backgroud = paste(backgroud, ElementList[index], Design)
            _DesignList = tf.concat([_DesignList, Design], 0)
            return index+1, _backgroud, ElementList, _DesignList, Tag
            

        #--------------------------------- Design Feature Network ----------------------------
    
        with tf.variable_scope('design_feature_network'):
            COLLECTIONS = ['COV_NETWORK_VARIABLES', tf.GraphKeys.GLOBAL_VARIABLES]
            
            # Hidden layer 1 conv : [256 x 192]x3 --> [256 x 192]x64
            with tf.variable_scope('Cov1') as variable_scope:
                w_conv1 = tf.get_variable('w1', regularizer=l2_reg, initializer=weight_variable([3,3,3,64]), collections=COLLECTIONS)
                Input = tf.reshape(self.Imgs, [-1,256,192,3])
                temp = conv2d(Input, w_conv1)
                temp = batch_normal(temp, self.is_training)
                op_conv1 = leaky_relu(temp)
                self.LayerOutput[variable_scope] = tf.transpose(op_conv1, perm=[3,0,1,2])
                
                mean, variance = moments(w_conv1)
                tf.summary.scalar("mean", mean)
                tf.summary.scalar("variance", variance)
                
            # Hidden layer 2 conv : [256 x 192]x64 --> [64 x 48]x64
            with tf.variable_scope('Cov2') as variable_scope:
                w_conv2 = tf.get_variable('w2', regularizer=l2_reg, initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                
                temp = conv2d(op_conv1, w_conv2)
                temp = batch_normal(temp, self.is_training)
                op_conv2 = leaky_relu(temp)
                op_pool2, pool_mask2 = max_pool_with_argmax(op_conv2, 4)
                
                self.LayerOutput[variable_scope] = tf.transpose(op_pool2, perm=[3,0,1,2])
                mean, variance = moments(w_conv2)
                tf.summary.scalar("mean", mean)
                tf.summary.scalar("variance", variance)
                
            # Hidden layer 3 conv : [64 x 48]x64 --> [16 x 12]x64
            with tf.variable_scope('Cov3') as variable_scope:
                w_conv3 = tf.get_variable('w3', regularizer=l2_reg, initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                
                temp = conv2d(op_pool2, w_conv3)
                temp = batch_normal(temp, self.is_training)
                op_conv3 = leaky_relu(temp)
                op_pool3, pool_mask3 = max_pool_with_argmax(op_conv3, 4)
                
                mean, variance = moments(w_conv3)
                tf.summary.scalar("mean", mean)
                tf.summary.scalar("variance", variance)
                
            # Hidden layer 4 conv : [16 x 12]x64 --> [4 x 3]x64
            with tf.variable_scope('Cov4') as variable_scope:
                w_conv4 = tf.get_variable('w4', regularizer=l2_reg, initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                
                temp = leaky_relu(conv2d(op_pool3, w_conv4))
                temp = batch_normal(temp, self.is_training)
                op_conv4 = leaky_relu(temp)
                op_pool4, pool_mask4= max_pool_with_argmax(op_conv4, 4)

                mean, variance = moments(w_conv4)
                tf.summary.scalar("mean", mean)
                tf.summary.scalar("variance", variance)
                
            # Fully connected layer 2_2: [3 x 4]x64 --> Vector[256]
            with tf.variable_scope('fc1') as variable_scope: 
                w_fc1 = tf.get_variable('w5', regularizer=l2_reg, initializer=weight_variable([3*4*64, 256]), collections=COLLECTIONS)
                b_fc1 = tf.get_variable('b5', regularizer=l2_reg, initializer=bias_variable([256], 0.5), collections=COLLECTIONS)
                
                h_pool4_flat = tf.reshape(op_pool4, [-1,3*4*64])
                h_fc1 = tf.nn.dropout(leaky_relu(tf.matmul(h_pool4_flat, w_fc1) + b_fc1), self.keep_prob)
        
        with tf.variable_scope('unconv_unpool_network'):
            self.LayerFeatureShow = []
            LayerFeature = tf.reshape(op_pool4[self.GetFeatureIndex,:,:,:,], [1,4,3,64])
            with tf.variable_scope('unconv3') as variable_scope:
                unpool4 = un_max_pool(LayerFeature,  tf.reshape(pool_mask4[self.GetFeatureIndex,:,:,:,], [1,4,3,64]), 4)
                temp = tf.nn.relu(unpool4)
                unconv4 = unconv(temp, w_conv4, [1, 16, 12, 64])
                self.LayerFeatureShow.append(tf.reshape(unconv4, [16,12,64]))
                
            with tf.variable_scope('unconv2') as variable_scope:
                unpool2 = un_max_pool(unconv4, tf.reshape(pool_mask3[self.GetFeatureIndex,:,:,:,], [1,16,12,64]), 4)
                temp = tf.nn.relu(unpool2)
                unconv3 = unconv(temp, w_conv3, [1, 64, 48, 64])
                self.LayerFeatureShow.append(tf.reshape(unconv3, [64,48,64]))
                
            with tf.variable_scope('unconv1') as variable_scope:
                unpool3 = un_max_pool(unconv3, tf.reshape(pool_mask2[self.GetFeatureIndex,:,:,:,], [1,64,48,64]), 4)
                temp = tf.nn.relu(unpool3)
                unconv2 = unconv(temp, w_conv2, [1, 256, 192, 64])  
                self.LayerFeatureShow.append(tf.reshape(unconv2, [256,192,64]))
                
            with tf.variable_scope('output') as variable_scope:
                temp = tf.nn.relu(unconv2)
                self.FeatureMap = tf.reshape(unconv(temp, w_conv1, [1, 256, 192, 3]), [256,192,3])  
            
            
        with tf.variable_scope('semantic_scoring_network'):
            COLLECTIONS = ['SEM_SCORING_NETWORK', tf.GraphKeys.GLOBAL_VARIABLES]
            
            #Fully Connected layer 3 : Vector[320] --> Vector[256]
            with tf.variable_scope('fc2') as variable_scope: 
                w_fc2 = tf.get_variable('w3', regularizer=l2_reg, initializer=weight_variable([256, 256]), collections=COLLECTIONS)
                b_fc2 = tf.get_variable('b3', regularizer=l2_reg, initializer=bias_variable([256]), collections=COLLECTIONS)
                h_fc2 = tf.nn.dropout(leaky_relu(tf.matmul(h_fc1, w_fc2) + b_fc2), self.keep_prob)
                

                
            #Fully Connected layer 4 : Vector[256] --> Vector[128]
            with tf.variable_scope('fc3') as variable_scope: 
                w_fc3 = tf.get_variable('w4', regularizer=l2_reg, initializer=weight_variable([256, 128]), collections=COLLECTIONS)
                b_fc3 = tf.get_variable('b4', regularizer=l2_reg, initializer=bias_variable([128]), collections=COLLECTIONS)
                h_fc3 = tf.nn.dropout(leaky_relu(tf.matmul(h_fc2, w_fc3) + b_fc3), self.keep_prob)
                

                
        #----------------------------------------    Output    -----------------------------------  
        with tf.variable_scope('Output'):
            w_op = tf.get_variable('w_op', regularizer=l2_reg, initializer=weight_variable([128, 5]), collections=COLLECTIONS)
            self.Score = tf.nn.softmax(tf.matmul(h_fc3, w_op))

            
        #----------------------------------------    Loss    -----------------------------------    
        with tf.variable_scope('loss') as variable_scope:
            self.loss = tf.losses.mean_squared_error( self.Labels, self.Score)
            self.l2_loss = self.l2_loss_rate*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            

            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.variable_scope('tarin'):
            with tf.control_dependencies(update_ops):
                learning_rate = tf.train.exponential_decay(self.lr, global_steps, 25, LEARNING_RATE_DECAY, staircase=False)  
                self._train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5).minimize(self.loss + self.l2_loss, global_step=global_steps)
                tf.summary.scalar("learning_rate", learning_rate)
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("l2Loss", self.l2_loss)
        self.merged_summary = tf.summary.merge_all()
        
    def Set_Learning_Rate(self, LearningRate):
        self.lr = LearningRate
        
    def _GetLayerOut(self, Imgs, Labels, is_training = False, use_shuffle = True):
        im=Image.fromarray(Imgs[30].reshape(256,192,3))
        im.show()
        for LayerName in self.LayerOutput:
            output = self.sess.run(self.LayerOutput[LayerName], feed_dict={ 
                                                        self.Imgs: Imgs[30].reshape([-1,256*192*3]),
                                                        self.Labels : Labels.reshape([-1,5]),   
                                                        self.keep_prob: 1.0,
                                                        self.is_training: is_training
                                                       })
            #for i in range(output.shape[0]):
            for index in range(output.shape[0]):
                feature = output[index][0]
                length,min = feature.max() - feature.min(),feature.min()
                for i in range(len(feature)):
                    for j in range(len(feature[i])):
                        feature[i][j] = int((feature[i][j] - min)/length*255)
                plt.subplot(4,4,(index)%16 + 1)
                plt.imshow(exposure.equalize_hist(feature), plt.cm.gray)
                if index%16 == 0 and index != 0:
                    plt.show()
            print(LayerName.name, output.shape)
            
    def GetFeatureMap(self, Imgs, Index = 0):
        Imgs = Imgs.reshape([-1,256*192*3])
        FeatureMap, LayerFeatureList = self.sess.run([self.FeatureMap, self.LayerFeatureShow], feed_dict = {
                                                    self.Imgs: Imgs,
                                                    self.keep_prob: 1.0,
                                                    self.is_training: False,
                                                    self.GetFeatureIndex: Index
                                                    })
        for LayerFeature in LayerFeatureList:
            for index in range(LayerFeature.shape[2]):
                feature = LayerFeature[:,:,index]
                length,min = feature.max() - feature.min(),feature.min()
                for i in range(len(feature)):
                    for j in range(len(feature[i])):
                        feature[i][j] = int((feature[i][j] - min)/length*255)
                plt.subplot(6,11,index+1)
                plt.imshow(feature, plt.cm.gray)
                plt.xticks([])
                plt.yticks([])
            plt.show()
            
        # hist remove some point
        _FeatureMap = np.mean(FeatureMap, axis = -1)
        hist = np.histogram(_FeatureMap, bins=100)
        num = 0
        min = None
        max = None
        for i in range(len(hist[0])):
            num += hist[0][i]
            if num > 300:
                min = hist[1][i]
                break
        num = 0
        for i in range(len(hist[0])):
            num += hist[0][len(hist[0]) - i - 1]
            if num > 300:
                max = hist[1][len(hist[0]) - i - 1]
                break
        _FeatureMap = np.maximum(_FeatureMap, min)
        _FeatureMap = np.minimum(_FeatureMap, max)
        _FeatureMap += np.min(_FeatureMap)
        _FeatureMap /= np.max(_FeatureMap)
        plt.matshow(_FeatureMap, cmap='YlOrRd',interpolation='nearest')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.show()
        return FeatureMap
        
    def DrawSensetiveMap(self, img):
        def square_error(A, B):
            return np.sum(np.square(A-B))
                    
        SensetiveMap = np.array([[0.0 for col in range(192) ] for row in range(256)])
        Score = self.sess.run(self.Score, feed_dict={ 
                                    self.Imgs: img.reshape([-1,256*192*3]),
                                    self.keep_prob: 1.0,
                                    self.is_training: False
                                   })[0]
        img = img.reshape([256,192,3])
        for i in np.arange(0,256,8):
            _Imgs = []
            locations = []
            for j in range(0,192,8):
                Y0,Y1,X0,X1 = [max(i-24,0), min(i+24,256), max(j-24,0), min(j+24,256)]
                _Img = copy.deepcopy(img)
                _Img[Y0:Y1, X0:X1, :] = 100
                locations.append([Y0,Y1,X0,X1])
                _Imgs.append(_Img.reshape([256*192*3]))
            Scores = self.sess.run(self.Score, feed_dict={ 
                                    self.Imgs: _Imgs,
                                    self.keep_prob: 1.0,
                                    self.is_training: False
                                   })
            for index in range(len(Scores)):
                SensetiveMap[locations[index][0]:locations[index][1], locations[index][2]:locations[index][3]] += square_error(Scores[index],Score)
            print("\rGenerate:%.1f%%"%((i+8)*100/256),end= " ")
        print(' ')
        SensetiveMap /= np.max(SensetiveMap)
        #plt.matshow(SensetiveMap, cmap='YlOrRd',interpolation='nearest')
        #plt.colorbar()
        #plt.show()
        return SensetiveMap
        
    def GetScore(self, img):
        Imgs = img.reshape([-1,256*192*3])
        return np.array(self.sess.run(self.Score, feed_dict={ 
                                                        self.Imgs: Imgs,
                                                        self.keep_prob: 1.0,
                                                        self.is_training: False
                                                       }))
    
    def Train(self, Times, SaveTimes = 10, keep_prob = 0.5, is_training = True):
        LearnStepCounter = 0
        DataSet = LoadingTrainingData()
        Writer = tf.summary.FileWriter(os.path.join(sys.path[0], 'logs/', GetCurTime() + r'/'), self.sess.graph)
        
        while LearnStepCounter < Times:
        
            print('Learn Times:', LearnStepCounter)
            Tag = [0 if i != PersonDict['cute'] else 1 for i in range(0,16)]
            _, Loss, summary  = self.sess.run([self._train_op,self.loss, self.merged_summary],
                        feed_dict={ 
                                    self.Imgs: DataSet['Imgs'].reshape([-1,256*192*3]),
                                    self.Labels : DataSet['Labels'].reshape([-1,5]),   
                                    self.keep_prob: keep_prob,
                                    self.is_training: is_training
                                   })
            Writer.add_summary(summary, LearnStepCounter)
            LearnStepCounter += 1
            print('Train Loss:',Loss)
            
            if (LearnStepCounter + 1)  % SaveTimes == 0:
                self._write_data()
                print('TestLoss:', self.TestLoss())
        return
        
    def ReDesign(self, ElementList):
        DesignList = [[0.0,0.0,1.0,1.0]]
        for i in range(1,len(ElementList)):
            DesignList.append([0,0,0.4,0.5])
        Z = []
        for i in np.arange(0,1,0.02):
            temp = []
            for j in np.arange(0,1,0.02):
                DesignList[1][0] = i
                DesignList[1][1] = j
                Img = GetDesginImg(ElementList, DesignList)        
                temp.append(self.GetScore(Img2Array(Img))[0])
            Z.append(temp)
            print(i)
        Z=np.array(Z)
        print(Z.shape)
        X = np.arange(0,1,0.02)
        Y = np.arange(0,1,0.02)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure()
        for i in range(5):
            ax = fig.add_subplot(2,3,i+1, projection='3d')
            _Z = Z[:,:,i]
            ax.plot_surface(X, Y, _Z, rstride=1, cstride=1, cmap='rainbow')
        plt.show()
        return DesignList
        
    def TestLoss(self, is_training = False):
        DataSet = LoadingTestingData()
        Loss = self.sess.run(self.loss,
                   feed_dict={ 
                            self.Imgs: DataSet['Imgs'].reshape([-1,256*192*3]),
                            self.Labels : DataSet['Labels'].reshape([-1,5]),    
                            self.keep_prob: 1.0,
                            self.is_training: is_training
                           })
        return Loss
        
#---------------------------- Main -----------------------------

if __name__ == '__main__':
    drNetWork = DRN()
    while 1:
        Input = input('NetWork Manager:').strip()
        if Input == 'Exit':
            break
        elif Input == 'Train':
            times = int(input('Train Times:'))
            keep_prob = float(input('keep_prob:'))
            drNetWork.Train(Times = times,keep_prob=keep_prob )
        elif Input == 'Test':
            print("Loss: ", drNetWork.TestLoss())
        elif Input == 'Clear':
            ClearDataSet()
        elif Input == 'GetLayer':
            DataSet = LoadingTrainingData()
            drNetWork._GetLayerOut(DataSet['Imgs'], DataSet['Labels'])
        elif Input == 'GetScore':
            DataSet = LoadingTempData()
            #Scores = drNetWork.GetScore(DataSet['Imgs'][ShuffleList[:10],:])    
            Scores = drNetWork.GetScore(DataSet['Imgs'])              
            for i in range(len(DataSet['Imgs'])):        
                print('index:',i,Scores[i])
        elif Input == 'GetFeatureMap':
            DataSet = LoadingTrainingData()
            im=Image.fromarray(DataSet['Imgs'][35].reshape(256,192,3))
            im.show()
            drNetWork.GetFeatureMap(DataSet['Imgs'][35])
        elif Input == 'GetSensetiveMap':
            DataSet = LoadingTempData()
            for i in range(len(DataSet['Imgs'])):
                print(i)
                SensetiveMap = drNetWork.DrawSensetiveMap(DataSet['Imgs'][i])
                fig = plt.figure()
                ax = fig.add_subplot(121)
                ax.imshow(DataSet['Imgs'][i].reshape(256,192,3))
                plt.xticks([])
                plt.yticks([])
                bx = fig.add_subplot(122)
                temp = bx.matshow(SensetiveMap, cmap='YlOrRd',interpolation='nearest')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(r'D:\graduation_word\Img',str(i)+"x_.png"))
                plt.close()
        else:
            print('Can\'t find Command: %s ' % Input)