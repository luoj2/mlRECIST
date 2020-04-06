import os
import sys
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import datetime
import data_helpers_predict_objective as data_helpers
from tensorflow.contrib import learn
from gensim import models
from model import Model

# np.random.seed(123)
# tf.set_random_seed(1)



# Parameters
# ==================================================

# Data loading params

tf.flags.DEFINE_string("data_source", "../New_training.xlsx", "Data source for the positive data.")
tf.flags.DEFINE_string("data_test", "../New_testing.xlsx", "Data test for the positive data.")
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("embedding", "/Users/tuanluu/Downloads/GoogleNews-vectors-negative300.bin", "Word2Vec embedding file")
tf.flags.DEFINE_string("embedding", "../glove_embeddings/glove.840B.300d.txt", "Glove embedding file")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 300, "Dimensionality of embedding")
tf.flags.DEFINE_integer("hidden_dim", 300, "Dimensionality of hidden layer")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate (default: 0.0001)")
tf.flags.DEFINE_float("init", 0.01, "Initial value (default: 0.01)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 2, "Batch Size (default: 2)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("record_step", 1, "Evaluate model on dev set after this many steps (default: 1)")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS(sys.argv)

out = open('../log_test/predict_objective/log_' + str(FLAGS.learning_rate) + "_" + str(FLAGS.init) + "_" + str(FLAGS.batch_size)+ "_" + str(FLAGS.num_epochs) + ".txt",'w')
out.write("\nParameters:\n")
print("\nParameters:")
for key in sorted(FLAGS.flag_values_dict()):
    print("{}={}".format(key.upper(), FLAGS.flag_values_dict()[key]))
    out.write("{}={}\n".format(key.upper(), FLAGS.flag_values_dict()[key]))
print("")
out.write("\n")



# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
out.write("Loading data...\n")
x_pre_train, x_post_train, y_train, _ = data_helpers.load_data_and_labels(FLAGS.data_source)
x_pre_test, x_post_test, y_test, ori_id = data_helpers.load_data_and_labels(FLAGS.data_test)

x_data = x_pre_train + x_post_train + x_pre_test + x_post_test
max_document_length = max([len(x.split(" ")) for x in x_data])

idt = 0
id_to_word = {}
word_to_id = {}
id_to_word[0] = '<pad>'
word_to_id['<pad'] = 0
idt += 1
id_to_word[1] = '<unk>'
word_to_id['<unk'] = 1
idt += 1

for x in x_data:
    for p in x.split(' '):
        if p in word_to_id:
            continue
        else:
            word_to_id[p] = idt
            id_to_word[idt] = p
            idt += 1

print("Vocabulary Size: {:d}".format(len(id_to_word)))
out.write("Vocabulary Size: {:d}\n".format(len(id_to_word)))

initW = np.random.uniform(-FLAGS.init,FLAGS.init,(len(word_to_id), FLAGS.embedding_size))

# Load W2V embedding
# ==================================================
# print('Start Loading Embedding!')
# model = models.KeyedVectors.load_word2vec_format(FLAGS.embedding, binary=True)
# for word in model.vocab:
#     if word in word_to_id:
#         initW[word_to_id[word]] = model[word]
# print('Finish Loading Embedding!')


# Load Glove embedding
# ==================================================
print('Start Loading Embedding!')
file = open(FLAGS.embedding,'r')
model = {}
for line in file.readlines():
    row = line.strip().split(' ')
    word = row[0]
    embedding = [float(val) for val in row[1:]]
    model[word] = embedding
print('Finish Loading Embedding!')
print('Length of embedding is: {:d}'.format(len(model)))
file.close()

for word in word_to_id:
    if word in model:
        initW[word_to_id[word]] = model[word]
 
initW[0] = [float(0)]*FLAGS.embedding_size


temp = []
for x in x_pre_train:
    t = []
    for e in x.split(' '):
        t.append(word_to_id[e])
    for i in range(max_document_length - len(x.split(' '))):
        t.append(0)
    t = np.array(t,dtype=np.int32)
    temp.append(t)
x_pre_train = np.array(list(temp),dtype=np.int32)

temp = []
for x in x_post_train:
    t = []
    for e in x.split(' '):
        t.append(word_to_id[e])
    for i in range(max_document_length - len(x.split(' '))):
        t.append(0)
    t = np.array(t,dtype=np.int32)
    temp.append(t)
x_post_train = np.array(list(temp),dtype=np.int32)

temp = []
for x in x_pre_test:
    t = []
    for e in x.split(' '):
        t.append(word_to_id[e])
    for i in range(max_document_length - len(x.split(' '))):
        t.append(0)
    t = np.array(t,dtype=np.int32)
    temp.append(t)
x_pre_test = np.array(list(temp),dtype=np.int32)

temp = []
for x in x_post_test:
    t = []
    for e in x.split(' '):
        t.append(word_to_id[e])
    for i in range(max_document_length - len(x.split(' '))):
        t.append(0)
    t = np.array(t,dtype=np.int32)
    temp.append(t)
x_post_test = np.array(list(temp),dtype=np.int32)

ori_id = np.array(list(ori_id))



# Training
# ==================================================
 
best_acc = 0
best_echo = 0
with tf.Graph().as_default():
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
#         np.random.seed(123)
#         tf.set_random_seed(1)
        model = Model(3,len(word_to_id),FLAGS.hidden_dim,FLAGS.learning_rate,FLAGS.init)
         
        print "Initialize variable"
        sess.run(tf.global_variables_initializer())
         
        print "Initialize embedding"
        sess.run(model.embedding_init, feed_dict={model.embedding_placeholder:initW})
         
        ################
        # Training batch
        ################
        def get_feed_dict(x_pre, x_post, y, batch_number, mode='training'):
            ''' This function returns a feed_dict for training
            Supports both training and testing mode. Just pass in different x,y sets
            batch_number extracts the correct batch, if set to None, returns entire set
            (For testing and evaluation normally we test the entire training set at once and no need to do batching)
         
            Input Arguments
            ---------------
            x : This is either x_train or x_dev
            y: This is either y_train or y_test
            batch_number: Integer. Indexing the batch, set to None for whole set
            mode: if Training, sets dropout to 0.5, else sets it to 1.0 (Testing cannot use dropout 0.5)
            '''
         
            if(batch_number is not None):
                # Get batch
                X_pre = x_pre[int(batch_number * FLAGS.batch_size):int(int(batch_number * FLAGS.batch_size) + FLAGS.batch_size)]
                X_post = x_post[int(batch_number * FLAGS.batch_size):int(int(batch_number * FLAGS.batch_size) + FLAGS.batch_size)]
                Y = y[int(batch_number * FLAGS.batch_size):int(int(batch_number * FLAGS.batch_size) + FLAGS.batch_size)]
                X_pre = np.array(X_pre,dtype=np.int32)
                X_post = np.array(X_post,dtype=np.int32)
                Y = np.array(Y,dtype=np.float32)
            else:
                # Get entire set as feed_dict
                X_pre = np.array(x_pre, dtype=np.int32)
                X_post = np.array(x_post, dtype=np.int32)
                Y = np.array(y, dtype=np.float32)
             
            if(mode=='training'):
                drop_val = FLAGS.dropout_keep_prob
            else:
                # Testing should use dropout 1.0
                drop_val = 1.0
         
            feed_dict = {
                        model.input_x_pre:X_pre, 
                        model.input_x_post:X_post, 
                        model.input_y:Y,
                        model.dropout_keep_prob:drop_val
                        }
            return feed_dict
         
         
        no_of_batches = int(len(y_train)/FLAGS.batch_size)
        print "Start training"
        time_str = datetime.datetime.now().isoformat()
        print(time_str)
        for t in range(FLAGS.num_epochs):
            shuffle_indices = np.random.permutation(np.arange(len(y_train)))
            x_pre_train_shuffle = x_pre_train[shuffle_indices]
            x_post_train_shuffle = x_post_train[shuffle_indices]
            y_train_shuffle = y_train[shuffle_indices]
            losses = []
#             for j in tqdm(range(no_of_batches)):
            for j in range(no_of_batches):
                # NOTE : Remove this tqdm is cannot install
                feed_dict = get_feed_dict(x_pre_train_shuffle, x_post_train_shuffle, y_train_shuffle, j, mode='training')
#                 _, loss = sess.run([model.train_step, model.cross_entropy], feed_dict)
                _, loss = sess.run([model.train_op, model.loss], feed_dict)
                losses.append(loss)
            print("[Epoch {}] Loss={}".format(t+1, np.mean(losses)))
            out.write("[Epoch {}] Loss={}\n".format(t+1, np.mean(losses)))
            if t % FLAGS.record_step == 0:
                feed_dict = get_feed_dict(x_pre_test, x_post_test, y_test, None, mode='testing')
                pre,acc = sess.run([model.predictions,model.accuracy], feed_dict)
                
                if best_acc < acc:
                    a = [0,0,0]
                    b = [0,0,0]
                    g = [0,0,0]
#                     print("Prediction",pre)
#                     print("Ground truth",y_test)
                    for i in range(len(pre)):
                        if y_test[i][pre[i]] == 1:
                            a[pre[i]] += 1
                    for i in range(len(pre)):
                        b[pre[i]] += 1
                    for i in range(len(y_test)):
                        g = [x+y for x,y in zip(g,y_test[i])] 
                    
                    print("Size of each class",g)
                    print("Predict of each class",b)
                    print("Correct prediction of each class",a)
                    out.write("Size of each class {}\n".format(g))
                    out.write("Predict of each class {}\n".format(b))
                    out.write("Correct prediction of each class {}\n".format(a))
                    
                    with open('../log_test/predict_objective/result_' + str(FLAGS.learning_rate) + "_" + str(FLAGS.init) + "_" + str(FLAGS.batch_size) + '.txt','w') as rec:
                        for i in range(len(pre)):
                            rec.write(ori_id[i] + '\t' + str(pre[i]) + '\n')
                            
#                     c = [float(x)/float(y) for x, y in zip(a,b)]
#                     print("Accuracy of each class:",c)
                    best_acc = acc
                    best_echo = t+1
                print("[Evaluate] Accuracy={}".format(acc))
                print("Best Accuracy={} at echo {}".format(best_acc,best_echo))
                out.write("[Evaluate] Accuracy={}\n".format(acc))
                out.write("Best Accuracy={} at echo {}\n".format(best_acc,best_echo))

out.close()