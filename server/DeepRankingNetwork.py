import os,sys
import numpy as np
import tensorflow as tf

#-------------------------- Args -----------------------

LEARNING_RATE = 0.05


#----------------------- Function ----------------------

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')
                        
def weight_variable( shape ):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return initial

def bias_variable( shape ):
    initial = tf.constant(0.1, shape=shape)
    return initial
    
class DRN:
    def __init__(self):
        self.lr = LEARNING_RATE
        
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
    
        weight_decay = 0.1
        l2_reg = tf.contrib.layers.l2_regularizer(weight_decay) 
        self.keep_prob = tf.placeholder(tf.float32, name='Keep_Prob')        
        self.margin = tf.placeholder(tf.float32, name='Margin')        
        
        #---------------------------------Design Feature Network----------------------------
        self.PositiveImgs = tf.placeholder(tf.float32, [None, 192*256*4], name='PositiveImgs')
        self.NegativeImgs = tf.placeholder(tf.float32, [None, 192*256*4], name='NegativeImgs')

        
        with tf.variable_scope('design_feature_network'):
            COLLECTIONS = ['COV_NETWORK_VARIABLES', tf.GraphKeys.GLOBAL_VARIABLES]
            
            # Hidden layer 1 conv : [192 x 256]x3 --> [192 x 256]x64
            with tf.variable_scope('Cov1'):
                w_conv1 = tf.get_variable('w1', regularizer=l2_reg, initializer=weight_variable([3,3,4,64]), collections=COLLECTIONS)
                b_conv1 = tf.get_variable('b1', regularizer=l2_reg, initializer=bias_variable([64]), collections=COLLECTIONS)
                l_positive1 = tf.reshape(self.PositiveImgs, [-1,192,256,4])
                l_negative1 = tf.reshape(self.NegativeImgs, [-1,192,256,4])
                h_conv1_posi = tf.nn.relu(conv2d(l_positive1, w_conv1) + b_conv1)
                h_conv1_neg = tf.nn.relu(conv2d(l_negative1, w_conv1) + b_conv1)
                
            # Hidden layer 2 conv : [192 x 256]x64 --> [48 x 64]x64
            with tf.variable_scope('Cov2'):
                w_conv2 = tf.get_variable('w2', regularizer=l2_reg, initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                b_conv2 = tf.get_variable('b2', regularizer=l2_reg, initializer=bias_variable([64]), collections=COLLECTIONS)
                h_conv2_posi = tf.nn.relu(conv2d(h_conv1_posi, w_conv2) + b_conv2)
                h_pool2_posi = max_pool_4x4(h_conv2_posi)
                h_conv2_neg = tf.nn.relu(conv2d(h_conv1_neg, w_conv2) + b_conv2)
                h_pool2_neg = max_pool_4x4(h_conv2_neg)
            
            # Hidden layer 3 conv : [48 x 64]x64 --> [12 x 16]x64
            with tf.variable_scope('Cov3'):
                w_conv3 = tf.get_variable('w3', regularizer=l2_reg, initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                b_conv3 = tf.get_variable('b3', regularizer=l2_reg, initializer=bias_variable([64]), collections=COLLECTIONS)
                h_conv3_posi = tf.nn.relu(conv2d(h_pool2_posi, w_conv3) + b_conv3)
                h_pool3_posi = max_pool_4x4(h_conv3_posi)
                h_conv3_neg = tf.nn.relu(conv2d(h_pool2_neg, w_conv3) + b_conv3)
                h_pool3_neg = max_pool_4x4(h_conv3_neg)
                
            # Hidden layer 4 conv : [12 x 16]x64 --> [3 x 4]x64
            with tf.variable_scope('Cov4'):
                w_conv4 = tf.get_variable('w4', regularizer=l2_reg, initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                b_conv4 = tf.get_variable('b4', regularizer=l2_reg, initializer=bias_variable([64]), collections=COLLECTIONS)
                h_conv4_posi = tf.nn.relu(conv2d(h_pool3_posi, w_conv4) + b_conv4)
                h_pool4_posi = max_pool_4x4(h_conv4_posi)
                h_conv4_neg = tf.nn.relu(conv2d(h_pool3_neg, w_conv4) + b_conv4)
                h_pool4_neg = max_pool_4x4(h_conv4_neg)
                
            # Fully connected layer 2_2: [3 x 4]x64 --> Vector[256]
            with tf.variable_scope('Fc2_2'):
                w_fc2_2 = tf.get_variable('w5', regularizer=l2_reg, initializer=weight_variable([3*4*64, 256]), collections=COLLECTIONS)
                b_fc2_2 = tf.get_variable('b5', regularizer=l2_reg, initializer=bias_variable([256]), collections=COLLECTIONS)
                h_pool4_flat_posi = tf.reshape(h_pool4_posi, [-1,3*4*64])
                h_fc2_2_posi = tf.nn.dropout(tf.nn.relu(tf.matmul(h_pool4_flat_posi, w_fc2_2) + b_fc2_2), self.keep_prob)
                h_pool4_flat_neg = tf.reshape(h_pool4_neg, [-1,3*4*64])
                h_fc2_2_neg = tf.nn.dropout(tf.nn.relu(tf.matmul(h_pool4_flat_neg, w_fc2_2) + b_fc2_2), self.keep_prob)
                
        #---------------------------------Semantic embedding network----------------------------
        self.PositiveTags = tf.placeholder(tf.float32, [None, 16], name='PositiveTags')
        self.NegativeTags = tf.placeholder(tf.float32, [None, 16], name='NegativeTags')
        with tf.variable_scope('semantic_embedding_network'):
            COLLECTIONS = ['SEM_EMBEDDING_NETWORK', tf.GraphKeys.GLOBAL_VARIABLES]
            
            # Fully connected layer 1 : Vector[16] --> Vector[64]
            with tf.variable_scope('fc1'):
                w_fc1 = tf.get_variable('w1', regularizer=l2_reg, initializer=weight_variable([16, 64]), collections=COLLECTIONS)
                b_fc1 = tf.get_variable('b1', regularizer=l2_reg, initializer=bias_variable([64]), collections=COLLECTIONS)
                l_positiveTags = tf.reshape(self.PositiveTags, [-1,16])
                l_negativeTags = tf.reshape(self.NegativeTags, [-1,16])
                h_fc1_posi = tf.nn.relu(tf.matmul(l_positiveTags, w_fc1) + b_fc1)
                h_fc1_neg = tf.nn.relu(tf.matmul(l_negativeTags, w_fc1) + b_fc1)
                
            # Fully connected layer 2_1 : Vector[16] --> Vector[64]
            with tf.variable_scope('fc2_1'):
                w_fc2_1 = tf.get_variable('w2', regularizer=l2_reg, initializer=weight_variable([64, 64]), collections=COLLECTIONS)
                b_fc2_1 = tf.get_variable('b2', regularizer=l2_reg, initializer=bias_variable([64]), collections=COLLECTIONS)
                h_fc2_1_posi = tf.nn.relu(tf.matmul(h_fc1_posi, w_fc2_1) + b_fc2_1)
                h_fc2_1_neg = tf.nn.relu(tf.matmul(h_fc1_neg, w_fc2_1) + b_fc2_1)
        
        #---------------------------------Semantic scoring network----------------------------
        
        # concat (Vector[64] Vector[256]) --> Vector[320] 
        input_fc3_posi = tf.concat([h_fc2_1_posi,h_fc2_2_posi], axis = 1)
        input_fc3_neg  = tf.concat([h_fc2_1_neg,h_fc2_2_neg], axis = 1)
        
        with tf.variable_scope('semantic_scoring_network'):
            COLLECTIONS = ['SEM_SCORING_NETWORK', tf.GraphKeys.GLOBAL_VARIABLES]
            
            #Fully Connected layer 3 : Vector[320] --> Vector[256]
            with tf.variable_scope('fc3'):
                w_fc3 = tf.get_variable('w3', regularizer=l2_reg, initializer=weight_variable([320, 256]), collections=COLLECTIONS)
                b_fc3 = tf.get_variable('b3', regularizer=l2_reg, initializer=bias_variable([256]), collections=COLLECTIONS)
                h_fc3_posi = tf.nn.dropout(tf.nn.relu(tf.matmul(input_fc3_posi, w_fc3) + b_fc3), self.keep_prob)
                h_fc3_neg = tf.nn.dropout(tf.nn.relu(tf.matmul(input_fc3_neg, w_fc3) + b_fc3), self.keep_prob)
            
            #Fully Connected layer 4 : Vector[256] --> Vector[128]
            with tf.variable_scope('fc4'):
                w_fc4 = tf.get_variable('w4', regularizer=l2_reg, initializer=weight_variable([256, 128]), collections=COLLECTIONS)
                b_fc4 = tf.get_variable('b4', regularizer=l2_reg, initializer=bias_variable([128]), collections=COLLECTIONS)
                h_fc4_posi = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc3_posi, w_fc4) + b_fc4), self.keep_prob)
                h_fc4_neg = tf.nn.dropout(tf.nn.relu(tf.matmul(h_fc3_neg, w_fc4) + b_fc4), self.keep_prob)
                
        #----------------------------------------    Output    -----------------------------------  
        with tf.variable_scope('Output'):
            w_op = tf.get_variable('w_op', regularizer=l2_reg, initializer=weight_variable([128, 1]), collections=COLLECTIONS)
            b_op = tf.get_variable('b_op', regularizer=l2_reg, initializer=bias_variable([1]), collections=COLLECTIONS)
            h_op_posi = tf.nn.relu(tf.matmul(h_fc4_posi, w_op) + b_op)
            h_op_neg = tf.nn.relu(tf.matmul(h_fc4_neg, w_op) + b_op)
            self.Score = h_op_posi
                
        #----------------------------------------    Loss    -----------------------------------    
        with tf.variable_scope('loss'):
            D = tf.nn.relu(tf.reshape(h_op_neg, [-1]) - tf.reshape(h_op_posi, [-1,1]) + self.margin)
            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = tf.reduce_sum(D) + l2_loss
        
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            
    def Draw_Graph(self):
        tf.summary.FileWriter(os.path.join(sys.path[0], 'logs/'), self.sess.graph)
        
    def Set_Learning_Rate(self, LearningRate):
        self.lr = LearningRate
        
    def Get_Score(self, img, Tag):
        TagData = [[0 if i != Tag else 1 for i in range(0,16)]]
        print(TagData)
        ImgData = img.reshape([-1,256*192*4])
        #somefunction ....
        
        Score = self.sess.run(self.Score, feed_dict={ 
                                                        self.PositiveImgs: ImgData,
                                                        self.PositiveTags: TagData, 
                                                        self.keep_prob: 1.0
                                                       })
        return Score
    
    def Train(self):
        
        return 