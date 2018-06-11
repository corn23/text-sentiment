# built-in library
import time
import os

# external packages
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from sklearn.model_selection import train_test_split
from gensim import models

# within the same project
from text_cnn import TextCNN
from data_helper import load_data_and_labels, batch_iter


# Parameters
# =================

# General Model Hyperparameters
tf.flags.DEFINE_integer('emb_dim', 50, 'Dimensionality of word embedding')
tf.flags.DEFINE_float('lr', 1e-2, 'learning rate for optimization')
tf.flags.DEFINE_float('l2_lambda', 0.0001, 'regularization coefficient')
tf.flags.DEFINE_float('drop_keep_prob', 0.95,'the dropout probability')

# CNN Parameters
tf.flags.DEFINE_string('filter_size_list', "3,4,5", 'filter_size')
tf.flags.DEFINE_integer('num_filters', 50, 'the number of filters')

# Training Parameters
tf.flags.DEFINE_integer('num_epoch', 1, 'number of epochs')
tf.flags.DEFINE_integer('batch_size', 64, 'number of samples in one batch')
tf.flags.DEFINE_integer('valid_interval', 100, 'validate after given number of training loop')
tf.flags.DEFINE_integer('ckpt_interval', 1000, 'save model after given number of training loop')

# Data Parameters
tf.flags.DEFINE_string('train_pos_file', 'twitter-datasets/train_pos.txt',"the path of positive training data")
tf.flags.DEFINE_string('train_neg_file', 'twitter-datasets/train_neg.txt',"the path of negative training data")
tf.flags.DEFINE_string('embedding_path', 'twitter-datasets/glove.6B.50d.txt',"the path for embeddings")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

for attr,value in FLAGS.__flags.items():
    print("{}={}".format(attr,value))

# Data Preparation
x_text, y = load_data_and_labels(FLAGS.train_pos_file,FLAGS.train_neg_file)

# build dict
max_length = max([len(text.strip().split(' ')) for text in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print("data prepared and dict built")

# split train and valid set
x_train, x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.01,random_state=10)

with tf.Session() as sess:
    model = TextCNN(sequence_length=max_length,
                    num_class=2,vocab_size=len(vocab_processor.vocabulary_),
                    emb_dim=FLAGS.emb_dim,
                    filter_size_list=list(map(int, FLAGS.filter_size_list.split(','))),
                    num_filters=FLAGS.num_filters,
                    l2_lambda=FLAGS.l2_lambda)
    global_step = tf.Variable(0, name="global_step",trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(model.loss,global_step=global_step)

    # create the dir for saving summary
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir,"run", timestamp))
    print("write summary to {}\n".format(out_dir))

    # add summary
    loss_summary = tf.summary.scalar(name="loss", tensor=model.loss)
    acc_summary = tf.summary.scalar(name="acc", tensor=model.acc)

    sess.run(tf.global_variables_initializer())

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary,acc_summary])
    train_summary_dir = os.path.join(out_dir,"summary", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph=sess.graph)

    # Valid summaries
    valid_summary_op = tf.summary.merge([loss_summary,acc_summary])
    valid_summary_dir = os.path.join(out_dir, "summary", "valid")
    valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, graph=sess.graph)

    # ckpt file
    checkpoint_dir = os.path.abspath(os.path.join(out_dir,"checkpoint"))
    checkpoint_prefix = os.path.join(checkpoint_dir,'model')
    if not os.path.exists(checkpoint_prefix):
        os.mkdir(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())

    # vocabulary dir
    vocab_path = os.path.join(os.path.curdir, "vocabulary")
    vocab_dir = os.path.abspath(vocab_path)
    if not os.path.exists(vocab_dir):
        os.mkdir(vocab_dir)
    vocab_processor.save(os.path.join(vocab_path,'vocab'))

    # prepare for the embeddings
    word_emb = models.KeyedVectors.load_word2vec_format(FLAGS.embedding_path)
    vocab_size = len(vocab_processor.vocabulary_)
    my_embedding = np.random.uniform(-0.25, 0.25, (vocab_size,FLAGS.emb_dim))
    for word in vocab_processor.vocabulary_._mapping:  # traverse through the keys
        idx = vocab_processor.vocabulary_._mapping[word]
        if word in word_emb.vocab:
            my_embedding[idx] = word_emb[word]

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None], name = "pretrained_embeddings")
    set_emb_op = model.pretrained_embedding.assign(pretrained_embeddings)
    sess.run(set_emb_op,feed_dict={pretrained_embeddings:my_embedding})

    def train_step(x_batch, y_batch, print_loss=True):
        """
        A single training step
        """
        feed_dict={
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.drop_prob: FLAGS.drop_keep_prob
        }
        _, step, train_summary, loss, acc = sess.run([train_op, global_step, train_summary_op, model.loss, model.acc],
                 feed_dict=feed_dict)
        if print_loss:
            print("training step:{} loss:{},acc:{}\n".format(step, loss, acc), flush=True)
        train_summary_writer.add_summary(train_summary)


    def valid_step(x_batch, y_batch):
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.drop_prob: 1
        }
        step, valid_summary, loss, acc = sess.run([global_step, valid_summary_op, model.loss, model.acc],
                                      feed_dict=feed_dict)
        print("validation step:{} loss:{},acc:{}\n".format(step, loss, acc), flush=True)
        valid_summary_writer.add_summary(valid_summary)

    batches = batch_iter(list(zip(x_train,y_train)),batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epoch)
    for i, batch in enumerate(batches):
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch, i%20 == 0)
        if i % FLAGS.valid_interval == 0:
            valid_step(x_valid,y_valid)
        if i % FLAGS.ckpt_interval == 0:
            path = saver.save(sess,checkpoint_prefix,global_step=i)



