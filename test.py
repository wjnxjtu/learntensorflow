#coding=utf-8
import tensorflow as tf 
#这里通过numpy工具包来生成模拟数据集 
from numpy.random import RandomState 
import numpy as np
import matplotlib.pyplot as plt
#init=tf.global_variables_initializer()
#a=tf.constant([[1.,5.],
#               [2.,3.],
#               [3.,4.]])
#print a              
#y=tf.constant([1,0,0])             
#sess=tf.Session()
#sess.run(init)
#print sess.run(tf.nn.softmax(a))
#b=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a,labels=y)
#print sess.run(b)

#定义训练数据batch的大小 
batch_size = 8 
#定义神经网络参数 
w1 = tf.Variable(tf.random_normal([2,10],stddev = 1,seed = 1)) 
w2 = tf.Variable(tf.random_normal([10,2],stddev = 1,seed = 1)) 
b1= tf.Variable(tf.random_normal([10],stddev = 1,seed = 1))
b2= tf.Variable(tf.random_normal([2],stddev = 1,seed = 1))
#在训练时需要把数据分成较小的batch 
x = tf.placeholder(tf.float32,shape=(None,2),name = 'x-input') 
y_ = tf.placeholder(tf.int32,shape=(None),name = 'y-input') 
#定义神经网络前向传播过程 
a =tf.matmul(x,w1)+b1
y=tf.matmul(a,w2)+b2

#定义损失函数和反向传播的算法 
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)
loss=tf.reduce_mean(loss)
print loss
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
#通过随机函数生成一个模拟数据集 
rdm = RandomState(1) 
dataset_size = 128
X = rdm.rand(dataset_size,2) 
Y = [int(x1+x2 < 1) for (x1,x2) in X] 
print Y
plt.scatter(X[:,0],X[:,1],15.0*np.array(Y)+1,15.0*np.array(Y)+1)
#plt.show()

correct = tf.nn.in_top_k(y, y_, 1)
correct = tf.cast(correct, tf.float16)
acc = tf.reduce_mean(correct)
with tf.Session() as sess: 
    init_op = tf.global_variables_initializer() 
    sess.run(init_op)  
    STEPS = 4000
    
    for i in range(STEPS): 
        start = (i * batch_size) % dataset_size 
        end = min(start+batch_size,dataset_size) 
        #通过选取的样本训练神经网络并更新参数 
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]}) 
        
        if i %100 == 0: 
            total_cross_entropy = sess.run(loss,feed_dict={x:X,y_:Y}) 
            print('after %d trainning step(s),cross entrop on all data is %g' %(i,total_cross_entropy)) 

    acc_=sess.run(correct,feed_dict={x:X,y_:Y}) 
    print acc_

