from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from  utils import *
from  models import GCN, MLP


import numpy as np 
import pickle as pkl
import sys
import matplotlib.pyplot as plt

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 3000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

flags.DEFINE_float('w1', 1, 'weight  for  reverse attention branch')
flags.DEFINE_float('w2', 1, 'weight  for  origin branch.')
flags.DEFINE_float('w3', 1, 'weight  for  reverse branch.')
# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

checkpt_file = './save/mod_'+FLAGS.dataset+'.ckpt'

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
saver = tf.train.Saver()

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

best_val_acc = 0
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    if acc > best_val_acc:
        best_val_acc = acc
        saver.save(sess, checkpt_file)


    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc),    "test_acc=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
saver.restore(sess, checkpt_file)
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print('test_acc: ', test_acc)



# mat = np.array(acc_test)
# print(np.max(mat))

# index_best =  np.argmax(mat)
# print('test:  index_best',np.argmax(mat))


# mat1 = np.array(acc_val)

# # if FLAGS.dataset == 'cora' or FLAGS.dataset == 'pubmed':
# #     trunc_index = 80
# # elif FLAGS.dataset == 'citeseer':
# #     trunc_index = 70
# val_index_best =  np.argmax(mat1)
# ans  = val_index_best
# for i in range(val_index_best, len(cost_val)):
#     if mat1[i] == mat1[val_index_best] and cost_val[i] < cost_val[val_index_best]:
#         ans = val_index_best

# print('val:  index_best',val_index_best)
# print('val:  index_best  regulate ',ans)
# print('test:  best result',mat[val_index_best])
# print('test:  best result regulate ',mat[ans])



# val_index_best_v2 =  np.argmax((mat1 - np.array(cost_val))[val_index_best-5:val_index_best+5])
# print('test:  best result v2',mat[val_index_best_v2 + val_index_best-5])



# val_index_best_v3 =  np.argmax((mat1 - np.array(cost_val)))
# print('test:  best result v3',mat[val_index_best_v3])
# ############################### 


# val_index_best_v4 =  np.argmin( np.array(cost_val))
# print('test:  best result v4',mat[val_index_best_v4])



