import os,sys,time
import numpy as np
from ImgDeal import *
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

#-------------------------- Args -----------------------
BN_EPSILON              = 0.005
MOVING_AVERAGE_DECAY    = 0.9997
DEAFULT_LEARNING_RATE   = 0.0003
BN_DECAY = MOVING_AVERAGE_DECAY
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

def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')
                        
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
    
#-----------------------  NetWork Class ----------------------

class DRN:
    def __init__(self):
    
        self.lr = DEAFULT_LEARNING_RATE
        self.is_training = tf.convert_to_tensor(True,dtype='bool',name='is_training')
        self.use_shuffle = tf.convert_to_tensor(False,dtype='bool',name='use_shuffle')
        self._build_network()
        self.sess = tf.Session()
            
        if os.path.isfile(os.path.join(sys.path[0], r'my_net/checkpoint')):
            self._load_data()
        else:
            self.sess.run(tf.global_variables_initializer())
            print('NetWork init!')
        
    def _load_data(self):
        Saver = tf.train.Saver()
        Saver.restore(self.sess, os.path.join(sys.path[0], r'my_net/save_net.ckpt'))
        print('Model restored.')
        
    def _write_data(self):
        Saver = tf.train.Saver()
        save_path = Saver.save(self.sess, os.path.join(sys.path[0], r"my_net/save_net.ckpt"))
        print("Model saved in file: %s" % save_path)
    
    def _build_network(self):
        class Tester:
            def __init__(self, name):
                self.name = name
        weight_decay = 0.1
        self.LayerOutput = {}
        l2_reg = tf.contrib.layers.l2_regularizer(weight_decay) 
        self.keep_prob = tf.placeholder(tf.float32, name='Keep_Prob')        
        self.margin = tf.placeholder(tf.float32, name='Margin')        
        self.AllLoss = tf.placeholder(tf.float32, name='AllLoss')
        
        #---------------------------------Design Feature Network----------------------------
        self.PositiveImgs = tf.placeholder(tf.float32, [None, None], name='PositiveImgs')
        self.NegativeImgs = tf.placeholder(tf.float32, [None, None], name='NegativeImgs')
        self.ShuffleList = tf.placeholder(tf.int32, [None], name="ShuffleList")
        self.ReShuffleList = tf.placeholder(tf.int32, [None], name='ReShuffleList')
        
        PosBatchSize = tf.shape(self.PositiveImgs)[0]
        NegBatchSize = tf.shape(self.NegativeImgs)[0]
        Batch = control_flow_ops.cond(
            tf.equal(tf.shape(self.NegativeImgs)[1], 0), lambda: self.PositiveImgs,
            lambda: tf.concat([self.PositiveImgs, self.NegativeImgs], axis = 0))
        Batch = control_flow_ops.cond(
            self.use_shuffle, lambda: tf.gather(Batch, self.ShuffleList),
            lambda: Batch)
        with tf.variable_scope('design_feature_network'):
            COLLECTIONS = ['COV_NETWORK_VARIABLES', tf.GraphKeys.GLOBAL_VARIABLES]
            
            # Hidden layer 1 conv : [192 x 256]x3 --> [192 x 256]x64
            with tf.variable_scope('Cov1') as variable_scope:
                w_conv1 = tf.get_variable('w1', regularizer=l2_reg, initializer=weight_variable([3,3,3,64]), collections=COLLECTIONS)
                Input = tf.reshape(Batch, [-1,192,256,3])
                temp = conv2d(Input, w_conv1)
                temp = batch_normal(temp, self.is_training)
                op_conv1 = leaky_relu(temp)
                self.LayerOutput[variable_scope] = tf.reshape(op_conv1,[-1])
                
            # Hidden layer 2 conv : [192 x 256]x64 --> [48 x 64]x64
            with tf.variable_scope('Cov2') as variable_scope:
                w_conv2 = tf.get_variable('w2', regularizer=l2_reg, initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                
                temp = conv2d(op_conv1, w_conv2)
                temp = batch_normal(temp, self.is_training)
                op_conv2 = leaky_relu(temp)
                op_pool2 = max_pool_4x4(op_conv2)
                
                self.LayerOutput[variable_scope] = tf.reshape(op_pool2,[-1])
                
            # Hidden layer 3 conv : [48 x 64]x64 --> [12 x 16]x64
            with tf.variable_scope('Cov3') as variable_scope:
                w_conv3 = tf.get_variable('w3', regularizer=l2_reg, initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                
                temp = conv2d(op_pool2, w_conv3)
                temp = batch_normal(temp, self.is_training)
                op_conv3 = leaky_relu(temp)
                op_pool3 = max_pool_4x4(op_conv3)
                
                self.LayerOutput[variable_scope] = tf.reshape(op_pool3,[-1])
                
            # Hidden layer 4 conv : [12 x 16]x64 --> [3 x 4]x64
            with tf.variable_scope('Cov4') as variable_scope:
                w_conv4 = tf.get_variable('w4', regularizer=l2_reg, initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                
                temp = leaky_relu(conv2d(op_pool3, w_conv4))
                temp = batch_normal(temp, self.is_training)
                op_conv4 = leaky_relu(temp)
                op_pool4= max_pool_4x4(op_conv4)

                self.LayerOutput[variable_scope] = tf.reshape(op_pool4,[-1])
                
            # Fully connected layer 2_2: [3 x 4]x64 --> Vector[256]
            with tf.variable_scope('Fc2_2') as variable_scope: 
                w_fc2_2 = tf.get_variable('w5', regularizer=l2_reg, initializer=weight_variable([3*4*64, 256]), collections=COLLECTIONS)
                b_fc2_2 = tf.get_variable('b5', regularizer=l2_reg, initializer=bias_variable([256], 0.5), collections=COLLECTIONS)
                
                h_pool4_flat = tf.reshape(op_pool4, [-1,3*4*64])
                h_fc2_2 = tf.nn.dropout(leaky_relu(tf.matmul(h_pool4_flat, w_fc2_2) + b_fc2_2), self.keep_prob)

                self.LayerOutput[variable_scope] = tf.reshape(h_fc2_2,[-1])
                
        #---------------------------------Semantic embedding network----------------------------
        self.Tag = tf.placeholder(tf.float32, [16], name='Tag')
        Tags = control_flow_ops.cond(
            tf.equal(tf.shape(self.NegativeImgs)[1],0), lambda:  tf.tile(tf.reshape(self.Tag, [1,16]), [PosBatchSize, 1]),
            lambda: tf.tile(tf.reshape(self.Tag, [1,16]), [tf.add(PosBatchSize,NegBatchSize), 1]))
            
        with tf.variable_scope('semantic_embedding_network'):
            COLLECTIONS = ['SEM_EMBEDDING_NETWORK', tf.GraphKeys.GLOBAL_VARIABLES]
            
            # Fully connected layer 1 : Vector[16] --> Vector[64]
            with tf.variable_scope('fc1') as variable_scope: 
                Input = tf.reshape(Tags, [-1,16])
                
                w_fc1 = tf.get_variable('w1', regularizer=l2_reg, initializer=weight_variable([16, 64], 0.5), collections=COLLECTIONS)
                b_fc1 = tf.get_variable('b1', regularizer=l2_reg, initializer=bias_variable([64], 0.5), collections=COLLECTIONS)
                
                temp = tf.matmul(Input, w_fc1) + b_fc1
                h_fc1 = leaky_relu(temp)
                
                self.LayerOutput[variable_scope] = tf.reshape(h_fc1,[-1])
                
            # Fully connected layer 2_1 : Vector[16] --> Vector[64]
            with tf.variable_scope('fc2_1') as variable_scope: 
                w_fc2_1 = tf.get_variable('w2', regularizer=l2_reg, initializer=weight_variable([64, 64], 0.5), collections=COLLECTIONS)
                b_fc2_1 = tf.get_variable('b2', regularizer=l2_reg, initializer=bias_variable([64], 0.5), collections=COLLECTIONS)
                
                temp = tf.matmul(h_fc1, w_fc2_1) + b_fc2_1
                h_fc2_1 = leaky_relu(temp)
                
                self.LayerOutput[variable_scope] = tf.reshape(h_fc2_1,[-1])
                
        #---------------------------------Semantic scoring network----------------------------
        
        # concat (Vector[64] Vector[256]) --> Vector[320] 
        input_fc3 = tf.concat([h_fc2_1,h_fc2_2], axis = 1)
        
        with tf.variable_scope('semantic_scoring_network'):
            COLLECTIONS = ['SEM_SCORING_NETWORK', tf.GraphKeys.GLOBAL_VARIABLES]
            
            #Fully Connected layer 3 : Vector[320] --> Vector[256]
            with tf.variable_scope('fc3') as variable_scope: 
                w_fc3 = tf.get_variable('w3', regularizer=l2_reg, initializer=weight_variable([320, 256]), collections=COLLECTIONS)
                b_fc3 = tf.get_variable('b3', regularizer=l2_reg, initializer=bias_variable([256]), collections=COLLECTIONS)
                h_fc3 = tf.nn.dropout(leaky_relu(tf.matmul(input_fc3, w_fc3) + b_fc3), self.keep_prob)
                
                self.LayerOutput[variable_scope] = tf.reshape(h_fc3,[-1])
                
            #Fully Connected layer 4 : Vector[256] --> Vector[128]
            with tf.variable_scope('fc4') as variable_scope: 
                w_fc4 = tf.get_variable('w4', regularizer=l2_reg, initializer=weight_variable([256, 128]), collections=COLLECTIONS)
                b_fc4 = tf.get_variable('b4', regularizer=l2_reg, initializer=bias_variable([128]), collections=COLLECTIONS)
                h_fc4 = tf.nn.dropout(leaky_relu(tf.matmul(h_fc3, w_fc4) + b_fc4), self.keep_prob)
                
                self.LayerOutput[variable_scope] = tf.reshape(h_fc4,[-1])
                
        #----------------------------------------    Output    -----------------------------------  
        with tf.variable_scope('Output'):
            w_op = tf.get_variable('w_op', regularizer=l2_reg, initializer=weight_variable([128, 1]), collections=COLLECTIONS)
            self.Score = tf.matmul(h_fc4, w_op)
            Output = control_flow_ops.cond(
            self.use_shuffle, lambda: tf.gather(self.Score, self.ReShuffleList),
            lambda: self.Score)
            print(Output.get_shape())
            self.LayerOutput[variable_scope] = tf.reshape(self.Score ,[-1])  
        #----------------------------------------    Loss    -----------------------------------    
        with tf.variable_scope('loss') as variable_scope:
            h_op = tf.reshape(Output, [-1])
            op_posi = tf.slice(h_op, [0], [PosBatchSize])
            op_neg = tf.slice(h_op, [PosBatchSize], [NegBatchSize])
            D = tf.nn.relu(tf.reshape(op_neg, [-1]) - tf.reshape(op_posi, [-1,1]) + self.margin)
            self.loss = tf.reduce_mean(D) 
            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            
            self.LayerOutput[variable_scope] = [op_posi,op_neg,D]
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            self._train_op = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.5).minimize(self.loss + l2_loss)
            
        with tf.variable_scope('write'):
            tf.summary.scalar("loss", self.AllLoss)
            
        self.merged_summary = tf.summary.merge_all()
        
    def Set_Learning_Rate(self, LearningRate):
        self.lr = LearningRate
        
    def Set_Is_Training(self, is_training):
        self.is_training = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
    
    def Set_Use_Shuffle(self, use_shuffle):
        self.use_shuffle = tf.convert_to_tensor(False,dtype='bool',name='use_shuffle')
        
    def _GetLayerOut(self, img, NegImgs, Tag, is_training = False, use_shuffle = True):
        Imgs = img.reshape([-1,256*192*3])
        self.Set_Is_Training(is_training)
        self.Set_Use_Shuffle(use_shuffle)
        print('*************************:',len(Imgs),len(NegImgs))
        ShuffleList = np.random.permutation(len(Imgs)+len(NegImgs))
        ReShuffleList = [0 for i in range(0,len(ShuffleList))]
        for i in range(0,len(ShuffleList)):
            ReShuffleList[ShuffleList[i]] = i
        for LayerName in self.LayerOutput:
            output = self.sess.run(self.LayerOutput[LayerName], feed_dict={ 
                                                        self.PositiveImgs: Imgs[np.random.permutation(int(len(Imgs)*0.6)),:],
                                                        self.NegativeImgs: NegImgs[np.random.permutation(int(len(NegImgs)*0.6)),:],
                                                        self.ShuffleList: ShuffleList,
                                                        self.ReShuffleList: ReShuffleList,          
                                                        self.Tag : [0 if i != PersonDict[Tag] else 1 for i in range(0,16)], 
                                                        self.keep_prob: 1.0,
                                                        self.margin: 5
                                                       })
            print(LayerName.name, output)
        
    def GetScore(self, img, Tag, is_training = False, use_shuffle = False):
        Imgs = img.reshape([-1,256*192*3])
        self.Set_Is_Training(is_training)
        self.Set_Use_Shuffle(use_shuffle)
        print(len(Imgs))
        return np.array(self.sess.run(self.Score, feed_dict={ 
                                                        self.PositiveImgs: Imgs,
                                                        self.NegativeImgs: [[]],
                                                        self.Tag: [0 if i != PersonDict[Tag] else 1 for i in range(0,16)], 
                                                        self.keep_prob: 1.0
                                                       })).reshape([-1])
    
    def Train(self, Times, SaveTimes = 10, keep_prob = 0.5, is_training = True, use_shuffle = True):
        LearnStepCounter = 0
        self.Set_Is_Training(is_training)
        self.Set_Use_Shuffle(use_shuffle)
        DataSet = LoadingTrainingData()
        Writer = tf.summary.FileWriter(os.path.join(sys.path[0], 'logs/', GetCurTime() + r'/'), self.sess.graph)
        while LearnStepCounter < Times:
            if (LearnStepCounter + 1)  % SaveTimes == 0:
                self._write_data()
                print(self.GetScore(DataSet['cute'], 'cute'))
                print(self.GetScore(DataSet['terror'], 'cute'))
            print('Learn Times:', LearnStepCounter)
            Loss = 0.0
            for Positive in DataSet:
                for Negative in DataSet:
                    if Positive == Negative:
                        continue
                    Imgs = DataSet[Positive]
                    NegImgs = DataSet[Negative]
                    ShuffleList = np.random.permutation(len(Imgs)+len(NegImgs))
                    ReShuffleList = [0 for i in range(0,len(ShuffleList))]
                    for i in range(0,len(ShuffleList)):
                        ReShuffleList[ShuffleList[i]] = i
                    Tag = [0 if i != PersonDict['cute'] else 1 for i in range(0,16)]
                    _, cost  = self.sess.run([self._train_op,self.loss],
                                    feed_dict={
                                        self.PositiveImgs: Imgs,
                                        self.NegativeImgs: NegImgs,
                                        self.ShuffleList: ShuffleList,
                                        self.ReShuffleList: ReShuffleList,
                                        self.Tag: Tag, 
                                        self.keep_prob: keep_prob,
                                        self.margin: 5
                                        })
                    Loss += cost
            print('Train Loss:',Loss)
            summary = self.sess.run(self.merged_summary,
                                feed_dict={
                                    self.AllLoss:Loss,
                                    })
            Writer.add_summary(summary, LearnStepCounter)
            LearnStepCounter += 1
            print('TestLoss:', self.TestLoss())
        self.Set_Use_Shuffle(False)
        return
        
    def TestLoss(self, is_training = False, use_shuffle = False):
        self.Set_Is_Training(is_training)
        self.Set_Use_Shuffle(use_shuffle)
        DataSet = LoadingTestingData()
        Loss = 0
        ''' for Positive in DataSet:
            for Negative in DataSet:
                if Positive == Negative:
                    continue'''
        Tag = [0 if i != PersonDict['cute'] else 1 for i in range(0,16)]
        cost = self.sess.run(self.loss,
                    feed_dict={
                        self.PositiveImgs: DataSet['cute'],
                        self.NegativeImgs: DataSet['terror'],
                        self.Tag: Tag, 
                        self.keep_prob: 1.0,
                        self.margin: 5.0,
                        })
        Loss += cost
                
        return Loss#/((len(DataSet)-1)*len(DataSet))
        
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
            Tag = input('Tag:')
            NegTag = input('NegTag:')
            DataSet = LoadingTrainingData()
            if Tag not in DataSet:
                print('%s not in DataSet'% Tag)
                continue
            UseBn = input('Use Batch Normal?(Y/N)') == 'Y'
            drNetWork._GetLayerOut(DataSet[Tag], DataSet[NegTag], Tag, UseBn)
        elif Input == 'GetScore':
            Tag = input('Tag:')
            DataSet = LoadingTrainingData()
            if Tag not in DataSet:
                print('%s not in DataSet'% Tag)
                continue
            for Label in DataSet:
                Scores = drNetWork.GetScore(DataSet[Label], Tag)
                print(Label+'\n', Scores)
        else:
            print('Can\'t find Command: %s ' % Input)
            help()