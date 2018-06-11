import tensorflow as tf

class TextCNN():
    def __init__(self, sequence_length, num_class,
                 vocab_size, emb_dim,
                 filter_size_list, num_filters,
                 l2_lambda):

        self.input_x = tf.placeholder(shape=(None,sequence_length),dtype=tf.int32, name="input_x")
        self.input_y = tf.placeholder(shape=(None,num_class),dtype=tf.int32,name="input_y")
        self.drop_prob = tf.placeholder(dtype=tf.float32,name="drop_keep_prob")
        self.pretrained_embedding = tf.Variable(tf.random_uniform((vocab_size,emb_dim),-1,1),dtype=tf.float32,name="emb")
        self.l2_lambda = l2_lambda
        l2_loss = 0
        with tf.name_scope('embedding'):
            self.embedding_tokens = tf.nn.embedding_lookup(self.pretrained_embedding,self.input_x)
            self.embedding_tokens_expand = tf.expand_dims(self.embedding_tokens,-1)


        pool_result = []
        for filter_size in filter_size_list:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = (filter_size,emb_dim,1,num_filters)
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1),name="W")
                b = tf.Variable(tf.constant(0.1, shape=(num_filters,)),name="b")
                # conv shape = (batch_size,sequence_length-filter_size+1,1,num_filters)
                conv = tf.nn.conv2d(self.embedding_tokens_expand, W, strides=[1,1,1,1],padding='VALID')+b
                conv = tf.nn.relu(conv)
                # pool shape = (batch_size,1,1,num_filters)
                pool = tf.nn.max_pool(conv,[1,sequence_length-filter_size+1,1,1],[1,1,1,1],padding='VALID')
                pool_result.append(pool)
                l2_loss += tf.nn.l2_loss(W)

        num_filter_total = num_filters*len(filter_size_list)
        # tf.stack(pool_result) -> (len(filter_size_list), pool_shape)
        pool_result_concat = tf.concat(pool_result, 3)
        pool_flat = tf.reshape(pool_result_concat,shape=(-1, num_filter_total))
        with tf.name_scope('dropout'):
            pool_flat_drop = tf.nn.dropout(pool_flat,keep_prob=self.drop_prob)

        with tf.name_scope('output'):
            #W_out = tf.Variable(tf.truncated_normal(shape=(num_filter_total,num_class),stddev=0.1),name="W_out")
            W_out = tf.get_variable(
                "W_out",
                shape=(num_filter_total,num_class),
                initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.Variable(tf.constant(0.1, shape=(num_class,)),name="b_out")
            logits = tf.nn.xw_plus_b(pool_flat_drop,W_out,b_out,name="logits")
            l2_loss += tf.nn.l2_loss(W_out)
            self.prob = tf.nn.softmax(logits=logits,name="prob")
            self.pred = tf.argmax(logits,dimension=1,name="pred")

        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=logits,name="loss")
            self.loss = tf.reduce_mean(loss) + self.l2_lambda*l2_loss
        with tf.name_scope('accuracy'):
            correct_num = tf.equal(self.pred,tf.argmax(self.input_y,dimension=1))
            self.acc = tf.reduce_mean(tf.cast(correct_num,dtype=tf.float32),name="acc")


