
# coding: utf-8

# In[7]:


import pandas as pd
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import roc_auc_score
import datetime
import csv


# In[8]:


chart_path_tr = '/data/dcjenkin/workspace/data/CHARTEVENTS_trn.csv'
chart_path_te = '/data/dcjenkin/workspace/data/CHARTEVENTS_tstn.csv'
adm_path = '/data/dcjenkin/workspace/data/ADMISSIONS.csv'

n_hidden = 80  # hidden layer num of features

n_classes = 61  # Mortality in hospital.  The network will put out n_classes values between 0 and 1 at each timestep indicating the prediction whether the patient has lived or died.  (0 for lived, 1 for died)

batch_size = 60

n_steps = 100  #100 timesteps per batch

learning_rate = 0.0001

item_list = [220045,220210,220277,211,742,646,618,220181,220179,220180,51,8368,52,8549,5815,5820,8554,8553,5819,5817,8551,581,455,8441,456,220052,220050,220051,113,223753,5813,8547,220739,223900,223901,223761,184,723,454,198,8548,5814,224641,224168,220074,678,677,8448,492,224054]

n_input = len(item_list) + 1   # (OH encoded vector of length len(item_list), value)


# In[9]:


def OH_F(ITID, items_l):
        oh_v = np.zeros(len(items_l))
        iidx = 0
        for item in items_l:
            if ITID == item:
                oh_v[iidx] = 1
            iidx += 1
        return (oh_v.tolist())

def csvgen(chart_path, item_list, adm_path):
    a_df = pd.read_csv(adm_path)
    ADM_DTH_Dict = {}
    
    for i, row in a_df.iterrows():
        ADM_DTH_Dict[row['HADM_ID']] = [row['ADMITTIME'], row['DISCHTIME'], row['DEATHTIME']]
        
    first_bool = True
    
    with open(chart_path, "r") as chcsv:
        reader = csv.reader(chcsv)
        next(reader)
        for row in reader: 
            if int(row[4]) in item_list and len(row[9]) > 0:
                ICVrow = [int(row[4]), row[5], float(row[9])]
                OH_V = OH_F(int(row[4]), item_list)
                ICVrow.extend(OH_V)
                if first_bool:
                    last_row_HADM_ID = row[2] #HADM_ID
                    first_bool = False
                    HADM_ID_chunk = [ICVrow]
                else:
                    next_row_HADM_ID = row[2]

                    if next_row_HADM_ID == last_row_HADM_ID:
                        HADM_ID_chunk.append(ICVrow)
                    else:
                        itarr = np.array([l[0] for l in HADM_ID_chunk])
                        CTarr = np.array([l[1] for l in HADM_ID_chunk])
                        VNarr = np.array([l[2:] for l in HADM_ID_chunk])

                        ADM_DTH_array = np.array(ADM_DTH_Dict[int(last_row_HADM_ID)])

                        M_array = np.zeros(shape= (len(CTarr), n_classes))

                        death_time = pd.Timestamp(ADM_DTH_array[2])

                        if not pd.isnull(death_time):
                            timetildeathlist = []

                            for ct in CTarr:
                                dt = death_time-pd.Timestamp(ct)
                                timetildeathlist.append(int(np.floor(dt.total_seconds()/60)))

                            i = 0
                            for ttd in timetildeathlist:
                                if ttd < n_classes-1:
                                    M_array[i,ttd:-1] = 1
                                i += 1


                            if pd.isnull(death_time):
                                M_array[:,-1] = 0
                            else: 
                                M_array[:,-1] = int(death_time <= pd.Timestamp(ADM_DTH_array[1]))


                        if pd.isnull(death_time) or (death_time > pd.Timestamp(ADM_DTH_array[0])):
                            yield (last_row_HADM_ID, itarr, CTarr, VNarr, M_array, ADM_DTH_array)

                        HADM_ID_chunk = [ICVrow]
                    last_row_HADM_ID = next_row_HADM_ID

def map_func1(HADM_ID, ITEM_IDs, Charttimes, Values, M_array, ADM_DSC_DTH):
    #Pre-Pad sequences so padded batches can be rounded up to lengths which are divisible by n_steps.
    
    #Record the number of measurements for each patient in num_meas
    num_meas = tf.shape(Values)[0]
    
    #Pad all sequences with n_steps zeros
    paddings = tf.constant([[0, n_steps,], [0, 0]])
    Values = tf.pad(Values, paddings, "CONSTANT")
    M_array = tf.pad(M_array, paddings, "CONSTANT")
    return HADM_ID, ITEM_IDs, Charttimes, Values, M_array, ADM_DSC_DTH, num_meas

def map_func2(HADM_ID, ITEM_IDs, Charttimes, Values, M_array, ADM_DSC_DTH, num_meas):
    #find the max number of measurements for patients in the batch.
    mlen = tf.reduce_max(num_meas)
    #round up to nearest multiple of n_steps and take only the values up to this index.  (All values after rmlen should be zeros from the padding)
    rmlen = tf.to_int32(tf.ceil(mlen/n_steps)*n_steps)
    Values = Values[:,0:rmlen]
    M_array = M_array[:,0:rmlen]
    #split up into groups of n_steps measurements each
    Values = tf.reshape(Values, [batch_size,-1,n_steps,n_input])
    M_array = tf.reshape(M_array, [batch_size,-1,n_steps,n_classes])
    #transpose to match format for rnn.
    Values = tf.transpose(Values, perm=[1, 0, 2, 3]) #(number of groups, batch size, n_steps, number of inputs)
    M_array = tf.transpose(M_array, perm=[1, 0, 2, 3]) #(number of groups, batch size, n_steps, number of classes)

    return HADM_ID, ITEM_IDs, Charttimes, Values, M_array, ADM_DSC_DTH, num_meas

def length(sequence):
    #returns a vector of sequence lengths for each patient within the batch segment
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


# In[10]:


tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_input], name='X')

y = tf.placeholder(tf.float32, [None, n_steps, n_classes], name='y')

#Weights and Biases to map hidden state to n_classes predictions
w1 = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='w1')
b1 = tf.Variable(tf.random_normal([n_classes]), name='w2')

# Define a lstm cell with tensorflow
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

state = lstm_cell.zero_state(batch_size, tf.float32)

c_v = tf.Variable(state[0], trainable=False)
h_v = tf.Variable(state[1], trainable=False)

vstate = tf.contrib.rnn.LSTMStateTuple(c_v, h_v)

outputs, states = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        dtype=tf.float32,
        sequence_length=length(X),
        inputs=X,
        initial_state=vstate)

state_op = tf.tuple([tf.assign(c_v, states[0]), tf.assign(h_v, states[1])])


out = tf.reshape(outputs, shape=[-1, n_hidden])
out = tf.matmul(out, w1)
out = tf.reshape(out, shape = [batch_size, n_steps, n_classes])
out = tf.add(out,b1)

#This creates a mask (X_m) indicating which values are not zero padded.
X_m = tf.sign(tf.reduce_max(tf.abs(X), 2))

#This removes outputs and labels corresponding to zero padding inputs.
yout = tf.boolean_mask(out, X_m)
ylab = tf.boolean_mask(y, X_m)
X_num = tf.reduce_sum(X_m, [0,1])

#This creates the set of mortality scores ysig which ranges from zero to 1. (Shape = [batch_size, n_steps, n_classes])
ysig = tf.sigmoid(yout)

#This defines the cost as cross entropy with logits.
cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets = ylab,logits = yout, pos_weight= 1), name='cost')

#The optimizer is AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, name='optimizer')

#Creates a saver whic will be used to save and load models.
saver = tf.train.Saver()


# In[11]:


ds_train = tf.data.Dataset.from_generator(
    lambda : csvgen(chart_path_tr, item_list, adm_path),
    (tf.int32, tf.int32, tf.string, tf.float32, tf.int32, tf.string),
    (tf.TensorShape([]), tf.TensorShape([None]),tf.TensorShape([None]), tf.TensorShape([None,n_input]),
     tf.TensorShape([None, n_classes]), tf.TensorShape([3]))
)

# train_dataset = ds_train.repeat()

ds_train = ds_train.map(map_func1, num_parallel_calls=10)

train_dataset = ds_train.shuffle(buffer_size=100)


ds_train = ds_train.padded_batch(
    batch_size, 
    padded_shapes=([],[None],[None],[None,n_input],[None, n_classes], [3], [])
)

ds_train = ds_train.filter(
    lambda HADM_ID, ITEM_IDs, Charttimes, Values, M_array, ADM_DSC_DTH, num_meas:
    tf.equal(tf.shape(HADM_ID)[0], batch_size)
)

ds_train = ds_train.map(map_func2, num_parallel_calls=10)

train_next_element = ds_train.make_one_shot_iterator().get_next()

ds_test = tf.data.Dataset.from_generator(
    lambda : csvgen(chart_path_te, item_list, adm_path),
    (tf.int32, tf.int32, tf.string, tf.float32, tf.int32, tf.string),
    (tf.TensorShape([]), tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([None,n_input]),
     tf.TensorShape([None, n_classes]), tf.TensorShape([3]))
)

ds_test = ds_test.map(map_func1, num_parallel_calls=10)

ds_test = ds_test.padded_batch(
    batch_size,
    padded_shapes=([],[None],[None],[None,n_input],[None, n_classes], [3], [])
)

ds_test = ds_test.filter(
    lambda HADM_ID, ITEM_IDs, Charttimes, Values, M_array, ADM_DSC_DTH, num_meas:
    tf.equal(tf.shape(HADM_ID)[0], batch_size)
)

ds_test = ds_test.map(map_func2, num_parallel_calls=10)

test_iterator = ds_test.make_initializable_iterator()
test_next_element = test_iterator.get_next()

train_dataset = ds_train.prefetch(2)
test_dataset = ds_test.prefetch(2)


# In[12]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./model4-02bs60.ckpt")
    AUC = []
    i = 0
    while True:
        try:
            train_value = sess.run(train_next_element)
        except tf.errors.OutOfRangeError:
            print('End of Epoch', flush=True)
#             save_path = saver.save(sess, "./model4.ckpt")
            break
            
        print('training batch: ' + str(i), flush=True)
        i += 1
        #initializes hidden/cell states to zero at the beginning of each patient batch.
        sess.run([tf.variables_initializer([c_v, h_v])])

        for j in np.arange(np.shape(train_value[3])[0]):
            #take the jth group's input and labels to feed into the LSTM.
            batch_x, batch_y = train_value[3][j], train_value[4][j]

            feed = {X: batch_x, y: batch_y}

            #run the session.  (optimizer will update parameters).  State op will update the hidden state and cell state to the values at the end of the previous group.
            result = sess.run([X, y, yout, ysig, ylab, cost, state_op, optimizer, X_num], feed_dict=feed)

        if (i+1)%500 == 0:
            print('--------------testing start---------------')
            save_path = saver.save(sess, "./model4-02bs60.ckpt")

            #create arrays for the prediction outputs and labels.
            preds = []
            labs = []
            #initialize the test iterator to start at the beginning
            sess.run(test_iterator.initializer)
            l = 0
            while True:
                print('test batch: ' + str(l), flush=True)
                try:
                    test_value = sess.run(test_next_element)

                    sess.run(tf.variables_initializer([c_v, h_v]))

                    for k in np.arange(np.shape(test_value[3])[0]):
                        #take the k'th group's input and labels to feed into the LSTM.
                        batch_x, batch_y = test_value[3][k], test_value[4][k]

                        feed = {X: batch_x, y: batch_y}

                        result = sess.run([X, y, yout, ysig, ylab, cost, state_op], feed_dict=feed)
                        #append the predictions and labels to the arrays
                        preds.extend(result[3])
                        labs.extend(result[4])
                    l += 1
                except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError) as e:
                    break
            labs = list(zip(*labs))
            preds = list(zip(*preds))
            AUC.append([roc_auc_score(labs[i], preds[i]) for i in range(n_classes)])
            print('Test AUC:' + str(AUC), flush=True)

