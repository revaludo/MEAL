#coding=utf-8
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time
import os,math
import tensorflow as tf
# import numpy as np
# from sklearn.model_selection import train_test_split
from dataprep import generatebatch
os.environ["CUDA_VISIBLE_DEVICES"]='1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True

def _pairwise_distances(embeddings, squared=False):
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, axis=1) - 2.0 * dot_product + tf.expand_dims(square_norm, axis=0)
    distances = tf.maximum(distances, 0.0)
    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances

def _get_triplet_mask_my(labels,casenum):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal_u = tf.equal(tf.expand_dims(labels[:,0], 0), tf.expand_dims(labels[:,0], 1))
    label_equal_l = tf.equal(tf.expand_dims(labels[:, 1], 0), tf.expand_dims(labels[:, 1], 1))
    i_equal_j_u = tf.expand_dims(label_equal_u, 2)
    i_equal_k_u = tf.expand_dims(label_equal_u, 1)
    i_equal_j_l = tf.expand_dims(label_equal_l, 2)
    i_equal_k_l = tf.expand_dims(label_equal_l, 1)
    if casenum==1:   # same user,same poi;same user,same poi;dif user,same poi
        valid_labels = tf.logical_and( tf.logical_and(i_equal_j_u, i_equal_j_l), tf.logical_and(tf.logical_not(i_equal_k_u), i_equal_k_l))
    if casenum==2:   # same user,same poi;same user,same poi;same user,dif poi
        valid_labels = tf.logical_and( tf.logical_and(i_equal_j_u, i_equal_j_l), tf.logical_and(tf.logical_not(i_equal_k_l), i_equal_k_u))
    if casenum==3:   # same user,same poi;same user,same poi;dif user,dif poi
        valid_labels = tf.logical_and( tf.logical_and(i_equal_j_u, i_equal_j_l), tf.logical_and(tf.logical_not(i_equal_k_l), tf.logical_not(i_equal_k_u)))
    if casenum==4:   # same user,same poi;dif user,same poi;same user,dif poi
        valid_labels = tf.logical_and( tf.logical_and(tf.logical_not(i_equal_j_u), i_equal_j_l), tf.logical_and(tf.logical_not(i_equal_k_l), i_equal_k_u))
    if casenum == 5:  # same user,same poi;dif user,same poi;dif user,dif poi
        valid_labels = tf.logical_and(tf.logical_and(tf.logical_not(i_equal_j_u), i_equal_j_l),
                                      tf.logical_and(tf.logical_not(i_equal_k_l), tf.logical_not(i_equal_k_u)))
    if casenum == 6:  # same user,same poi;same user,dif poi;dif user,dif poi
        valid_labels = tf.logical_and(tf.logical_and(tf.logical_not(i_equal_j_l), i_equal_j_u),
                                      tf.logical_and(tf.logical_not(i_equal_k_l), tf.logical_not(i_equal_k_u)))

    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask

def batch_all_triplet_loss_my(labels, embeddings, casenum,margin1,margin2,margin3,margin4,margin5,margin6,squared=False):
    pairwise_dis = _pairwise_distances(embeddings, squared=squared)
    anchor_positive_dist = tf.expand_dims(pairwise_dis, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = tf.expand_dims(pairwise_dis, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    if casenum==1:   # same user,same poi;same user,same poi;dif user,same poi
        margin=margin1
    if casenum==2:   # same user,same poi;same user,same poi;same user,dif poi
        margin = margin2
    if casenum==3:   # same user,same poi;same user,same poi;dif user,dif poi
        margin = margin3
    if casenum==4:   # same user,same poi;dif user,same poi;same user,dif poi
        margin = margin4
    if casenum == 5:  # same user,same poi;dif user,same poi;dif user,dif poi
        margin = margin5
    if casenum == 6:  # same user,same poi;dif user,same poi;dif user,dif poi
        margin = margin6

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask_my(labels,casenum)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_postive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return mask,triplet_loss, fraction_postive_triplets

class MEAL():

    def __init__(self, rmargin1=0.1,rmargin2=0.2,rmargin3=0.3,rmargin4=0.1,rmargin5=0.2,rmargin6=0.1, batch_num=30, d_a=10,uphonum=100,vphonum=200,factor_dim=50,num_factors=4096, num_iterations=3,
                 reg=0.1):
        self.batch_num = batch_num
        self.d_a=d_a
        self.uphonum=uphonum
        self.vphonum = vphonum
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='placehold_x')
        self.labels = tf.placeholder(tf.int32, [None, 2], name='placehold_y')

        self.margin1 = rmargin1
        self.margin2 = rmargin2
        self.margin3 = rmargin3
        self.margin4 = rmargin4
        self.margin5 = rmargin5
        self.margin6 = rmargin6

        self.parameters = []
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68 / 255, 116.779 / 255, 103.939 / 255], dtype=tf.float32, shape=[1, 1, 1, 3],
                               name='img_mean')
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), trainable=True, name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        self.output = tf.nn.l2_normalize(fc1, dim=1, epsilon=1e-12, name='output')
        tf.identity(self.output, name="inference")


        mask1, triplet1, _ = batch_all_triplet_loss_my(self.labels, self.output, 1, self.margin1, self.margin2, self.margin3, self.margin4, self.margin5,
                                                       self.margin6)
        loss1 = tf.reduce_mean(triplet1, name='triplet1')
        mask2, triplet2, _ = batch_all_triplet_loss_my(self.labels, self.output, 2, self.margin1, self.margin2, self.margin3, self.margin4, self.margin5,
                                                       self.margin6)
        loss2 = tf.reduce_mean(triplet2, name='triplet2')
        # loss2 = tf.constant(0.0)
        mask3, triplet3, _ = batch_all_triplet_loss_my(self.labels, self.output, 3, self.margin1, self.margin2, self.margin3, self.margin4, self.margin5,
                                                       self.margin6)
        loss3 = tf.reduce_mean(triplet3, name='triplet3')
        mask4, triplet4, _ = batch_all_triplet_loss_my(self.labels, self.output, 4, self.margin1, self.margin2, self.margin3, self.margin4, self.margin5,
                                                       self.margin6)
        loss4 = tf.reduce_mean(triplet4, name='triplet4')
        mask5, triplet5, _ = batch_all_triplet_loss_my(self.labels, self.output, 5, self.margin1, self.margin2, self.margin3, self.margin4, self.margin5,
                                                       self.margin6)
        loss5 = tf.reduce_mean(triplet5, name='triplet5')
        mask6, triplet6, _ = batch_all_triplet_loss_my(self.labels, self.output, 6, self.margin1, self.margin2, self.margin3, self.margin4, self.margin5,
                                                       self.margin6)
        loss6 = tf.reduce_mean(triplet6, name='triplet6')
        self.lossp = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        tf.summary.scalar("lossp", self.lossp)


        self.upins = tf.placeholder(tf.float32, [None, 4096])
        self.vpins = tf.placeholder(tf.float32, [None, 4096])

        with tf.variable_scope("self_attention_layer_u"):
            uW_s1 = tf.get_variable('uW_s1', shape=[self.d_a, 4096])
            uW_s2 = tf.get_variable('uW_s2', shape=[1, self.d_a])
            # attention
            # shape = (r, batch_size*n)
            uA = tf.nn.softmax(
                tf.matmul(uW_s2,
                          tf.tanh(
                              tf.matmul(uW_s1, tf.reshape(self.upins, [4096, -1]))
                          )
                          )
            )
            # shape = (batch_size, r, n)
            uA = tf.reshape(uA, shape=[-1,  self.uphonum])
            # shape = (batch_size, r, 2*u)
            self.uM = tf.matmul(uA, self.upins)


        with tf.variable_scope("self_attention_layer_v"):
            vW_s1 = tf.get_variable('vW_s1', shape=[self.d_a, 4096])
            vW_s2 = tf.get_variable('vW_s2', shape=[1, self.d_a])
            # attention
            # shape = (r, batch_size*n)
            vA = tf.nn.softmax(
                tf.matmul(vW_s2,
                          tf.tanh(
                              tf.matmul(vW_s1, tf.reshape(self.vpins, [4096, -1]))
                          )
                          )
            )
            # shape = (batch_size, r, n)
            vA = tf.reshape(vA, shape=[-1, self.vphonum])
            # shape = (batch_size, r, 2*u)
            self.vM = tf.matmul(vA, self.vpins)




        self.num_factors = num_factors
        self.factor_dim = factor_dim
        self.num_iterations = num_iterations
        self.reg = reg
        self.mu= tf.placeholder(tf.float32, shape=[None,self.factor_dim], name='u_factor')
        self.mv= tf.placeholder(tf.float32, shape=[None,self.factor_dim], name='v_factor')

        self.u = tf.concat([self.mu, self.uM], 1)
        self.v = tf.concat([self.mv, self.vM], 1)
        x_input = tf.placeholder(tf.float32, shape=[None,self.num_factors+self.factor_dim], name='x_input')
        y_input = tf.placeholder(tf.int32, shape=[None], name='y_input')

        W = tf.Variable(tf.truncated_normal([self.num_factors+self.factor_dim, 2]), name='W')
        b = tf.Variable(tf.zeros([2]), name='b')

        logits = tf.sigmoid(tf.matmul(x_input, W) + b)
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=y_input)
        self.lossm = tf.reduce_mean(loss)

        self.loss = self.lossp+self.lossm

        optimizer = tf.train.AdamOptimizer(0.1)
        self.train_op1 = optimizer.minimize(self.lossp)
        self.train_op2 = optimizer.minimize(self.lossm)
        self.train_op = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def train_model(self,img_file,label_file,matrix_file):
        weights = np.load('data/vgg16_weights.npz')
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i < 28:
                print(i, k, np.shape(weights[k]))
                self.sess.run(self.parameters[i].assign(weights[k]))

        batchu,batchv,batchp,batchpl,batchuv,batchm,ufactors,vfactors=generatebatch(img_file,label_file,matrix_file,self.batch_num)

        for epoch in range(1000):
            for i in range(self.batch_num):
                if epoch<300:
                    _,loss1,ebds=self.sess.run([self.train_op1,self.lossp,self.output],
                                  feed_dict={self.imgs: batchp, self.labels: batchpl})
                    print("Epoch: [%2d] [%4d/%4d]  loss1: %.6f"
                          % (epoch + 1, i + 1, self.batch_num, loss1))
                else:
                    xin = []
                    for item in batchuv:
                        xin.append(tf.concat([self.u[item[0]], self.v[item[1]]], 1))

                    upos=[]
                    for user in batchu:
                        upo=[]
                        for i in range(len(batchpl)):
                            if batchpl[i][0]==user:
                                upo.append(ebds[i])
                    upos.append(upo)

                    vpos = []
                    for item in batchv:
                        vpo = []
                        for i in range(len(batchpl)):
                            if batchpl[i][1] == item:
                                vpo.append(ebds[i])
                    vpos.append(vpo)


                    if epoch < 600:
                        _,loss2=self.sess.run([self.train_op2,self.lossm],
                                      feed_dict={self.x_input: xin, self.y_input: batchm,self.upins: upos, self.vpins: vpos,self.mu: ufactors, self.mv:vfactors})
                        print("Epoch: [%2d] [%4d/%4d]  loss2: %.6f"
                              % (epoch + 1, i + 1, self.batch_num, loss2))
                    else:
                        _, loss,ebds = self.sess.run([self.train_op, self.loss, self.output],
                                                 feed_dict={self.imgs: batchp, self.labels: batchpl,self.x_input: xin, self.y_input: batchm,self.upins: upos, self.vpins: vpos,self.mu: ufactors, self.mv:vfactors})
                        print("Epoch: [%2d] [%4d/%4d]  total loss: %.6f"
                                          % (epoch + 1, i + 1, self.batch_num,  loss))



model = MEAL()
model.train_model('data/imgf.npy','data/imgl.npy','data/uvmatrix.txt')


