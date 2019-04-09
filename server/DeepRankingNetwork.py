import os,sys
import numpy as np
import tensorflow as tf

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')
                        
def weight_variable( shape ):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable( shape ):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
class DeepRankingNetwork:
    def __init__(self):
        self._build_network()
        self.sess = tf.Session()
        
    def _draw_graph(self):
        tf.summary.FileWriter(os.path.join(sys.path[0], 'logs/'), self.sess.graph)
        
    def _load_data(self):
        Saver = tf.train.Saver()
        Saver.restore(self.sess, os.path.join(sys.path[0], r'my_net/save_net.ckpt'))
    
    def _write_data(self):
        Saver = tf.train.Saver()
        Saver.save(self.sess, "my_net/save_net.ckpt")
        
    def _build_network(self):
        
        #---------------------------------Design Feature Network----------------------------
        self.PositiveImgs = tf.placeholder(tf.float32, [None, 384*512*3], name='PositiveImgs')
        self.NegativeImgs = tf.placeholder(tf.float32, [None, 384*512*3], name='NegativeImgs')
        self.Score = tf.placeholder(tf.float32, [None, 1])
        
        with tf.variable_scope('design_feature_network'):
            COLLECTIONS = ['COV_NETWORK_VARIABLES', tf.GraphKeys.GLOBAL_VARIABLES]
            
            # Hidden layer 1 conv : [384 x 512]x3 --> [384 x 512]x64
            with tf.variable_scope('Cov1'):
                w_conv1 = tf.get_variable('w1', initializer=weight_variable([3,3,3,64]), collections=COLLECTIONS)
                b_conv1 = tf.get_variable('b1', initializer=bias_variable([64]), collections=COLLECTIONS)
                l_positive1 = tf.reshape(self.PositiveImgs, [-1,288,384,3])
                l_negative1 = tf.reshape(self.NegativeImgs, [-1,288,384,3])
                h_conv1_posi = tf.nn.relu(conv2d(l_positive1, w_conv1) + b_conv1)
                h_conv1_neg = tf.nn.relu(conv2d(l_negative1, w_conv1) + b_conv1)
                
            # Hidden layer 2 conv : [384 x 512]x64 --> [96 x 128]x64
            with tf.variable_scope('Cov2'):
                w_conv2 = tf.get_variable('w2', initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                b_conv2 = tf.get_variable('b2', initializer=bias_variable([64]), collections=COLLECTIONS)
                h_conv2_posi = tf.nn.relu(conv2d(h_conv1_posi, w_conv2) + b_conv2)
                h_pool2_posi = max_pool_4x4(h_conv2)
                h_conv2_neg = tf.nn.relu(conv2d(l_negative1, w_conv2) + b_conv2)
                h_pool2_neg = max_pool_4x4(h_conv2)
            
            # Hidden layer 3 conv : [96 x 128]x64 --> [24 x 32]x64
            with tf.variable_scope('Cov3'):
                w_conv3 = tf.get_variable('w3', initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                b_conv3 = tf.get_variable('b3', initializer=bias_variable([64]), collections=COLLECTIONS)
                h_conv3_posi = tf.nn.relu(conv2d(h_pool2_posi, w_conv3) + b_conv3)
                h_pool3_posi = max_pool_4x4(h_conv3)
                h_conv3_neg = tf.nn.relu(conv2d(h_pool2_neg, w_conv3) + b_conv3)
                h_pool3_neg = max_pool_4x4(h_conv3)
                
            # Hidden layer 4 conv : [24 x 32]x64 --> [6 x 8]x64
            with tf.variable_scope('Cov4'):
                w_conv4 = tf.get_variable('w4', initializer=weight_variable([3,3,64,64]), collections=COLLECTIONS)
                b_conv4 = tf.get_variable('b4', initializer=bias_variable([64]), collections=COLLECTIONS)
                h_conv4_posi = tf.nn.relu(conv2d(h_pool3_posi, w_conv4) + b_conv4)
                h_pool4_posi = max_pool_4x4(h_conv4)
                h_conv4_neg = tf.nn.relu(conv2d(h_pool3_neg, w_conv4) + b_conv4)
                h_pool4_neg = max_pool_4x4(h_conv4)
                
            # Fully connected layer 2_2: [6 x 8]x64 --> Vector[256]
            with tf.variable_scope('Fc2_2'):
                w_fc2_2 = tf.get_variable('w5', initializer=weight_variable([6*8*64, 256]), collections=COLLECTIONS)
                b_fc2_2 = tf.get_variable('b5', initializer=bias_variable([256]), collections=COLLECTIONS)
                h_pool4_flat_posi = tf.reshape(h_pool4_posi, [-1,6*8*64])
                h_fc2_2_posi = tf.nn.relu(tf.matmul(h_pool4_flat_posi, w_fc2_2) + b_fc2_2)
                h_pool4_flat_neg = tf.reshape(h_pool4_neg, [-1,6*8*64])
                h_fc2_2_neg = tf.nn.relu(tf.matmul(h_pool4_flat_neg, w_fc2_2) + b_fc2_2)
                
        #---------------------------------Semantic embedding network----------------------------
        self.Tags = tf.placeholder(tf.float32, [None, 16], name='Tags')
        
        with tf.variable_scope('semantic_embedding_network'):
            COLLECTIONS = ['SEM_EMBEDDING_NETWORK', tf.GraphKeys.GLOBAL_VARIABLES]
            
            # Fully connected laer 1 : Vector[16] --> Vector[64]
            with tf.variable_scope('fc1'):
                w_fc1 = tf.get_variable('w1', initializer=weight_variable([16, 64]), collections=COLLECTIONS)
                b_fc1 = tf.get_variable('b5', initializer=bias_variable([256]), collections=COLLECTIONS)
                h_fc1 = tf.nn.relu(tf.matmul(self.Tags, w_fc1) + b_fc1)
                
            # Fully connected laer 1 : Vector[16] --> Vector[64]
            with tf.variable_scope(fc2_1):
            