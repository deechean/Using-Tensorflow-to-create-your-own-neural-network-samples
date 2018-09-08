# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf

classify_labels = ['setosa','versicolor','virginica']
#定义常量
TRAIN_BATCH_SIZE = 3 #每次取一个数据点训练

def get_batch_data(file_queue, batch_size):    
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)   
    Id, Sepal_Length,Sepal_Width,Petal_Length,Petal_Width,label = tf.decode_csv(value,record_defaults=[[1],[1.0],[1.0],[1.0],[1.0],['null']])
    Y = tf.case({
        tf.equal(label, tf.constant('setosa')): lambda: tf.constant([1,0,0],dtype = tf.float32), \
        tf.equal(label, tf.constant('versicolor')): lambda: tf.constant([0,1,0],dtype = tf.float32), \
        tf.equal(label, tf.constant('virginica')): lambda: tf.constant([0,0,1],dtype = tf.float32),}, \
        lambda: tf.constant([0,0,0],dtype = tf.float32), exclusive=True)
    get_data, get_label = tf.train.batch([[Sepal_Length,Sepal_Width,Petal_Length,Petal_Width],Y], 
                                         batch_size = batch_size)
    return get_data, get_label
   
with tf.Graph().as_default() as g: 
    #生成训练数据文件队列,此处我们只有一个训练数据文件
    train_file_queue = tf.train.string_input_producer(['iris_train.csv'], num_epochs=None) 
  
    #从训练数据文件列表中获取数据和标签
    data,Y = get_batch_data(train_file_queue, TRAIN_BATCH_SIZE)
    #将数据reshape成[每批次训练样本数,每个训练样本包含的数据量]
    X =tf.reshape(data,[TRAIN_BATCH_SIZE,4])
    #随机初始化一个weight变量
    W = tf.Variable(tf.random_normal([4,3], mean=0.0, stddev=1.0, dtype=tf.float32), 
                    trainable=True, name='weight')
    #初始化bias变量为0
    b = tf.Variable(tf.zeros([3]),trainable=True, name='bias')
    #使用softmax作为激活函数    
    logits = tf.nn.softmax(tf.matmul(X,W) + b, name = 'softmax')
    #计算交叉熵
    cross_entropy = -tf.reduce_sum(Y*tf.log(logits))    
    #定义一个优化器
    Optimizer = tf.train.GradientDescentOptimizer(0.01)
    #计算梯度
    gradient = Optimizer.compute_gradients(cross_entropy)
    #更新W和B来最小化交叉熵
    train_op = Optimizer.apply_gradients(gradient)
    
    train_eval = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits,1),tf.arg_max(Y,1)),dtype=tf.float32))
    
    #local_variables_initializer则会报错
    init_local = tf.local_variables_initializer()
    #初始化所有全局变量，
    init_global = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        #初始化本地和全局变量
        init_local.run()
        init_global.run()
        #创建一个Coordinator用于训练结束后关闭队列
        coord = tf.train.Coordinator()
        #启动队列
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            i = 0
            while (not coord.should_stop()) and i < 2000:                
                loss, _,accuracy = sess.run([cross_entropy,train_op,train_eval])                       
                print('Train step:' + str(i) + '-----------------------------------')
                print('Loss: '+str(loss))
                print('Train accuracy:' + str(accuracy*100)+'%')
                i += 1
        except tf.errors.OutOfRangeError: 
            print("Training done")
        finally:
            coord.request_stop()
        coord.join(threads)   
    

        
                  
