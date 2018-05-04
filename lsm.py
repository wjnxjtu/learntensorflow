#coding=utf-8                                                                                                                                                                                                                                                                 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

f=open('./data/period_trend.csv')  
df=pd.read_csv(f)     
data=np.array(df['value'])
plt.figure()
#plt.plot(data)
#plt.show()

#标准化
normalize_data=(data-np.mean(data))/np.std(data)  
#增加维度
normalize_data=normalize_data[:,np.newaxis] 

#----------------------形成训练集-------------------------#
#设置常量
time_step=20      #时间步
rnn_unit=10       #隐藏层神经单元
batch_size=60     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0006         #学习率

#生成训练集
train_x,train_y=[],[]   
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    print x
    train_x.append(x.tolist())
    train_y.append(y.tolist()) 

#每批次输入的tensor
X=tf.placeholder(tf.float32, [None,time_step,input_size])    
#每批次Tensor的对应的标签
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   

#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }
def lstm(batch):     #参数：输入网络批次数目    
    w_in=weights['in']
    b_in=biases['in']
    #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input=tf.reshape(X,[-1,input_size]) 
    input_rnn=tf.matmul(input,w_in)+b_in
    #将tensor转成3维，作为lstm cell的输入
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit]) 
    
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit,reuse=tf.get_variable_scope().reuse)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    print output
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states
def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(batch_size)
    #定义损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #训练1000次，可以增加次数
        for i in range(10):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #每10步保存一次参数
                if step%20==0:
                    print("Number of iterations:",i," loss:",loss_)
                    print("model_save",\
                          saver.save(sess,\
                                     './data/modle.ckpt'))
 #运行在windows 10,使用'model_save1\\modle.ckpt'
 #运行在Linux,使用 'model_save1/modle.ckpt'
                step+=1
        print("The train has finished")
def prediction():
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(1)    
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, './data/modle.ckpt')
        prev_seq=data[-1]
        predict=[]
        for i in range(100):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()
train_lstm()
#prediction()
