import tensorflow as tf
from tensorflow.contrib import learn
from data_helper import load_test_data,create_submission_file
import numpy as np

class test_model():
    def __init__(self,ckpt_path):
        self._sess = tf.Session()
        saver = tf.train.import_meta_graph(ckpt_path)
        saver.restore(self._sess, ckpt_path.split('.')[0])
        graph = tf.get_default_graph()
        self._input_x = graph.get_tensor_by_name("input_x:0")
        self._input_y = graph.get_tensor_by_name("input_y:0")
        self._pred = graph.get_tensor_by_name("output/pred:0")
        self._acc = graph.get_tensor_by_name("accuracy/acc:0")
        self._drop_keep_prob = graph.get_tensor_by_name('drop_keep_prob:0')
        self._prob = graph.get_tensor_by_name("output/prob:0")

    def predict(self,x):
        prediction = self._sess.run([self._pred],feed_dict={self._input_x:x,
                                                            self._drop_keep_prob:1})
        return prediction
    def prob(self,x):
        prob = self._sess.run([self._prob],feed_dict={self._input_x:x,
                                                     self._drop_keep_prob:1})
        return prob
    def score(self,x,label):
        acc = self._sess.run([self._acc],feed_dict={self._input_x:x,
                                                    self._input_y:label,
                                                    self._drop_keep_prob:1})
        return acc


if __name__ == "main":
    x_text = load_test_data('twitter-datasets/test_data.txt')
    vocab = learn.preprocessing.VocabularyProcessor.restore('vocabulary/vocab')
    x_data = np.array(list(vocab.transform(x_text)))

    ckpt_path = 'run/1526114630/checkpoint/model-3000.meta'
    tmodel = test_model(ckpt_path=ckpt_path)
    result = tmodel.predict(x_data)
    create_submission_file(result[0],'submission.txt')


