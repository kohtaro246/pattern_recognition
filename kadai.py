import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from copy import copy
import time



# load datasets
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def select(dataset):
    count = 0
    select = []
    for labels in dataset["labels"]:
        if labels > 4:
            select.append(count)
        count += 1

    dataset["data"] = np.delete(dataset["data"], select, axis=0)
    for i in reversed(select):
        dataset["filenames"].pop(i)
        dataset["labels"].pop(i)
    return dataset


def reshape_pic_data(dataset):
    data_len = len(dataset["labels"])
    new_data = np.empty((data_len, 32, 32, 3))
    for i in range(data_len):
        image = dataset["data"][i]
        red = image[0:1024].reshape(32, 32)
        blue = image[1024:2048].reshape(32, 32)
        green = image[2048:].reshape(32, 32)
        new_data[i] = np.dstack((red, green, blue))
    return new_data





def load_dataset(path):
    dataset = unpickle(path)
    dataset = select(dataset)
    dataset["data"] = reshape_pic_data(dataset)
    return dataset


train_set1 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_1")["data"]
train_class1 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_1")["labels"]
train_set2 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_2")["data"]
train_class2 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_2")["labels"]
train_set3 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_3")["data"]
train_class3 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_3")["labels"]
train_set4 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_4")["data"]
train_class4 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_4")["labels"]
train_set5 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_5")["data"]
train_class5 = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/data_batch_5")["labels"]

test_set = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/test_batch")["data"]
test_class = load_dataset(
    "./cifar-10-python/cifar-10-batches-py/test_batch")["labels"]

train_set = np.concatenate(
    [train_set1, train_set2, train_set3, train_set4, train_set5])
train_class = train_class1 + train_class2 + \
    train_class3 + train_class4 + train_class5
evaluation_set = np.concatenate(
    [train_set1, train_set2, train_set3, train_set4, train_set5, test_set])
evaluation_class = train_class1 + train_class2 + \
    train_class3 + train_class4 + train_class5 + test_class

train_class = np.asarray(train_class)
train_class = train_class.reshape([train_class.shape[0], 1])
test_class = np.asarray(test_class)
test_class = test_class.reshape([test_class.shape[0], 1])
evaluation_class = np.asarray(evaluation_class)
evaluation_class = evaluation_class.reshape([evaluation_class.shape[0], 1])

# print(train_set.shape)
# print(train_class.shape)
import cupy as np
train_set = np.asarray(train_set)
train_class = np.asarray(train_class)
test_set = np.asarray(test_set)
test_class = np.asarray(test_class)
evaluation_set = np.asarray(evaluation_set)
evaluation_class = np.asarray(evaluation_class)
test_set_orig = np.array(test_set, copy = True)

# flatten and normalize
train_set = train_set.reshape(train_set.shape[0], -1).T / 255.0
test_set = test_set.reshape(test_set.shape[0], -1).T / 255.0
evaluation_set = evaluation_set.reshape(evaluation_set.shape[0], -1).T / 255.0
#print(train_set.shape)

# convert to onehot
train_class = np.eye(5)[train_class.reshape(-1)].T
test_class = np.eye(5)[test_class.reshape(-1)].T
evaluation_class = np.eye(5)[evaluation_class.reshape(-1)].T
# print(test_class.shape)


def create_random_minibatches(batch_num, train_set, train_class, seed):
    assert train_set.shape[1] % batch_num == 0
    train_sets = []
    train_classes = []
    batch_train_set_len = train_set.shape[1] // batch_num
    
    perm = np.random.permutation(train_set.shape[1]).tolist()
    
    train_set_shuffled = train_set[:, perm]
    train_class_shuffled = train_class[:, perm]

    for i in range(0, batch_num):
        train_sets.append(
            train_set_shuffled[:, (i*batch_train_set_len):((i+1)*batch_train_set_len)])
        train_classes.append(
            train_class_shuffled[:, (i*batch_train_set_len):((i+1)*batch_train_set_len)])

    return train_sets, train_classes

def create_cross_validation_set(split_num, evaluation_set, evaluation_class):
    assert evaluation_set.shape[1] % split_num == 0
    datasets = []
    classes = []
    dataset_len = evaluation_set.shape[1] // split_num

    for i in range(0, split_num):
        datasets.append(evaluation_set[:, (i*dataset_len):((i+1)*dataset_len)])
        classes.append(evaluation_class[:, (i*dataset_len):((i+1)*dataset_len)])

    return datasets, classes

def init_params(dims):

    params = {}

    for i in range(1, len(dims)):
        params['w'+str(i)] = np.random.randn(dims[i], dims[i-1]) * 0.01
        params['b'+str(i)] = np.zeros((dims[i], 1))

    return params


def relu(z):
    a = np.maximum(0, z)
    return a


def softmax(z):
    # print(z)
    a = np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True)
    return a


def for_prop_step(x, w, b, activation_type):
    z = np.dot(w, x)+b

    if activation_type == "relu":
        a = relu(z)
        # print(a.shape)
    elif activation_type == "softmax":
        a = softmax(z)

    return z, a


def for_prop(inp, params):
    layer_num = len(params)//2
    tmps = []
    x = inp
    # print(x.shape)
    for i in range(1, layer_num):
        z, a = for_prop_step(x, params['w'+str(i)], params['b'+str(i)], "relu")
        tmps.append([z, a, params['w'+str(i)], params['b'+str(i)], x])
        x = a
        # print(z.shape)

    z, a = for_prop_step(
        x, params['w'+str(layer_num)], params['b'+str(layer_num)], "softmax")
    tmps.append([z, a, params['w'+str(layer_num)],
                 params['b'+str(layer_num)], x])

    return a, tmps


# a, tmps = for_prop(train_set, init_params([train_set.shape[0], 25, 12, 5]))
# print(a)
# tmps return the following [z,a,wi,bi,xi]


def ce_loss_l2(a, label, params, lamb):
    # print(label.shape)
    layer_num = len(params) // 2
    cost_tmp = -np.sum(label*np.log(a), axis=0, keepdims=True)
    # print(cost_tmp.shape[1])
    cost = np.sum(cost_tmp)/cost_tmp.shape[1]

    l2_reg_cost = 0.0
    for i in range(0, layer_num):
        l2_norm = np.linalg.norm(params["w" + str(i+1)], "fro")
        l2_reg_cost += l2_norm ** 2

    l2_reg_cost = l2_reg_cost * lamb / 2 / cost_tmp.shape[1]
    cost = cost + l2_reg_cost
    return cost


#print(ce_loss_l2(a, train_class))


def back_prop_step(da, tmp, activation_type, lamb, minibatch_class):
    train_class = minibatch_class
    z = np.array(tmp[0], copy=True)
    a = np.array(tmp[1], copy=True)
    w = np.array(tmp[2], copy=True)
    b = np.array(tmp[3], copy=True)

    x = np.array(tmp[4], copy=True)

    if activation_type == "relu":
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0
    elif activation_type == "softmax":
        dz = a-train_class

    dw = np.dot(dz, np.transpose(x)) / \
        train_class.shape[0] + w * lamb / train_class.shape[0]
    db = np.sum(dz, axis=1, keepdims=True) / train_class.shape[0]
    dx = np.dot(np.transpose(w), dz)
    # print("w")
    # print(w.shape)
    # print("dw")
    # print(dw.shape)
    # print(x.shape)
    return dx, dw, db


# print(tmps[-1][1])


def back_prop(tmps, minibatch_class, weight_decay):
    derivs = {}
    layer_num = len(tmps)

    derivs["da" + str(layer_num-1)], derivs["dw" + str(layer_num)], derivs["db" +
                                                                           str(layer_num)] = back_prop_step(None, tmps[-1], "softmax", weight_decay, minibatch_class)
    for i in reversed(range(layer_num-1)):
        da_prev, dw, db = back_prop_step(
            derivs["da" + str(i+1)], tmps[i], "relu", weight_decay, minibatch_class)
        derivs["da" + str(i)] = da_prev
        derivs["dw" + str(i+1)] = dw
        derivs["db" + str(i+1)] = db

    return derivs


# print(back_prop(tmps))

def update(params, derivs, learning_rate):
    layer_num = len(params) // 2

    for i in range(layer_num):
        params["w" + str(i+1)] = params["w" + str(i+1)] - \
            learning_rate * derivs["dw" + str(i+1)]
        params["b" + str(i+1)] = params["b" + str(i+1)] - \
            learning_rate * derivs["db" + str(i+1)]

    return params


def learn(train_set, train_class, test_set, test_class, learning_rate, batch_num, weight_decay, epochs,layers, eval):
    cost = 0
    cost_prev = 0
    cost_prev_prev = 0
    train_accu = 0.0
    test_accu = 0.0
    costs = []
    train_accuracies = []
    test_accuracies = []

    # initialize params
    params = init_params(layers)
    
    for i in tqdm(range(0, epochs)):
        seed = 40
        train_sets, train_classes = create_random_minibatches(
            batch_num, train_set, train_class, seed)
        
        j = 0
        if i == 2000 or i == 4000 or i == 6000 or i == 8000:
            learning_rate = learning_rate * 0.5 
        for minibatch in train_sets:
            #print("layer" + str(j))
            a, tmps = for_prop(minibatch, params)
            cost_prev_prev = cost_prev
            cost_prev = cost
            cost = ce_loss_l2(a, train_classes[j], params, weight_decay)
            
            if cost < 0.2 or test_accu > 0.8:
                if eval == False:
                    print("Cost after epoch {} : {}" .format(i, np.squeeze(cost)))
                    print("learning rate: {}" .format(learning_rate))
                break
            if cost_prev_prev < cost_prev and cost_prev < cost:
                learning_rate = learning_rate * 1
            derivs = back_prop(tmps, train_classes[j], weight_decay)
            
            params = update(params, derivs, learning_rate)

            
            
            
            #print("here")

            j += 1
       
        costs.append(cost.item())
        
        # if i % 100 == 0:
        if eval == False:
            print("Cost after epoch {} : {}" .format(i, np.squeeze(cost)))
            print("learning rate: {}" .format(learning_rate))
        #print("Train accuracy: {}" .format(train_accu))
        #print("Test accuracy: {}" .format(test_accu))
        seed += 1
        # print(i)
        if i % 1000 == 0 and eval == False:
            prob1, train_accu = accuracy(
                train_set, train_class, params)
            prob2, test_accu = accuracy(
                test_set, test_class, params)
            train_accuracies.append(train_accu)
            test_accuracies.append(test_accu)
            print("Train accuracy: {}" .format(train_accu))
            print("Test accuracy: {}" .format(test_accu))

    return params, costs, train_accuracies, test_accuracies


def accuracy(data, label, params):
    # print(data.shape[1])
    m = data.shape[1]
    n = len(params) // 2
    prob, tmps = for_prop(data, params)
    #print(prob[:, 0])
    types = prob[:, 0].shape[0]
    # print(types)
    for i in range(0, prob.shape[1]):
        m_index = np.argmax(prob[:, i])
        # print(m_index)
        for j in range(0, types):
            if j == m_index:
                prob[j][i] = 1
                # print("prob")
                # print(prob[j][i])
            else:
                prob[j][i] = 0

    sum = 0.0
    for i in range(0, m):

        #print("prob: {}".format(prob[:, i]))
        #print("label: {}".format(label[:, i]))
        if (prob[:, i] == label[:, i]).all():
            sum += 1.0

    #print("Accuracy:" + str(sum/float(m)))
    return prob, sum/float(m)

def eval_params(data, label, params):
    # print(data.shape[1])
    m = data.shape[1]
    n = len(params) // 2
    prob, tmps = for_prop(data, params)
    #print(prob[:, 0])
    types = prob[:, 0].shape[0]
    # print(types)
    for i in range(0, prob.shape[1]):
        m_index = np.argmax(prob[:, i])
        # print(m_index)
        for j in range(0, types):
            if j == m_index:
                prob[j][i] = 1
                # print("prob")
                # print(prob[j][i])
            else:
                prob[j][i] = 0

    sum = 0.0
    precision = np.zeros(5)
    
    #precision
    for j in range(0, 5):
        denom = 0.0
        nom = 0.0
        for i in range(0, m):
            if np.argmax(prob[:, i]) == j:
                denom += 1.0

                if (prob[:, i] == label[:, i]).all():
                    nom += 1.0
        precision[j] = nom/denom

    recall = np.zeros(5)
    #recall
    for j in range(0,5):
        denom = 0.0
        nom = 0.0
        for i in range(0, m):
            if np.argmax(label[:, i]) == j:
                denom += 1.0
                if (prob[:, i] == label[:, i]).all():
                    nom += 1.0
        recall[j] = nom/denom

    f = np.zeros(5)
    #f
    f = 2.0 * precision * recall / (precision + recall)
            
   
    return precision, recall, f


def train(train_set, train_class, test_set, test_class, learning_rate, batch_num, weight_decay, epochs, layers, eval):
    #print("learning rate = "+ str (learning_rate))
    prob1 = 0.0
    prob2 = 0.0
    params, costs, train_accuracies, test_accuracies = learn(train_set, train_class, test_set, test_class, learning_rate, batch_num, weight_decay, epochs, layers, eval)
    prob1, train_accu = accuracy(train_set, train_class, params)
    prob2, test_accu = accuracy(test_set, test_class, params)

    if eval == False:
        print("Train accuracy: {}" .format(train_accu))
        print("Test accuracy: {}" .format(test_accu))
        train_accuracies.append(train_accu)
        test_accuracies.append(test_accu)

        path = './costs/kadai_' + str(layers) + '.txt'
        f = open(path,'w')
        f.write(str(costs[-1]))
        f.close

        with open('./params_' + str(layers) + '.pickle', mode='wb') as f:
            pickle.dump(params, f)

        fig = plt.figure()
        plt.plot(costs, label="cost")
        plt.title('cost')
        plt.xlabel('epoch')
        plt.ylabel('cost')
        plt.legend(loc="lower right")
        fig.savefig("./result/cost_" + str(layers) + ".png")
        #   plt.show()

        fig = plt.figure()
        plt.plot(train_accuracies, label="train_accuracy")
        plt.plot(test_accuracies, label="test_accuracy")
        plt.title('Accuracy')
        plt.xlabel('epoch (per 1000)')
        plt.ylabel('accuracy')
        plt.legend(loc="lower right")
        fig.savefig("./result/accuracy_" + str(layers) + ".png")
        # plt.show()

    return train_accu, test_accu, costs[-1]

def cross_validation(evaluation_set, evaluation_class, learning_rate, split_num, batch_num, weight_decay, epochs, layers):
    datasets = []
    classes = []
    datasets, classes = create_cross_validation_set(split_num, evaluation_set, evaluation_class)
    sum_accu = 0.0
    average_accu = 0.0
    sum_prec = 0.0
    average_prec = 0.0
    sum_recall = 0.0
    average_recall = 0.0
    sum_f = 0.0
    average_f = 0.0
    

    for k in tqdm(range(0, split_num)):
        prob2 = 0.0
        train_accu = 0.0
        test_accu = 0.0
        datasets_tmp = copy(datasets)
        classes_tmp = copy(classes)
        
        test_set = datasets_tmp.pop(k)
        test_class = classes_tmp.pop(k)
        train_set = np.concatenate(datasets_tmp, axis=1)
        train_class = np.concatenate(classes_tmp, axis=1)
        #print(test_set.shape)
        #print(train_set.shape)
        params, costs, train_accuracies, test_accuracies = learn(train_set, train_class, test_set, test_class, learning_rate, batch_num, weight_decay, epochs, layers, True)
        prob2, test_accu = accuracy(test_set, test_class, params)
        prec, recall, f = eval_params(test_set, test_class, params)
        sum_accu += test_accu
        sum_prec += prec
        sum_recall += recall
        sum_f += f

    average_accu = sum_accu/float(split_num)
    average_prec = sum_prec/float(split_num)
    average_recall = sum_recall/float(split_num)
    average_f = sum_f/float(split_num)
    
    return average_accu, average_prec, average_recall, average_f

    

        


def akaike_information_criterion(cost, layers):
    layer_num = len(layers)
    sum = 0.0
    for i in range(0, layer_num-1):
        sum += layers[i] * layers[i+1]
    for i in range(1, layer_num):
        sum += layers[i]
    aic = 2 * cost + 2 * (sum)

    return aic

def aic_evaluation(layers):
    aics = []
    layer_num = [1,2,3,4]
    for i in layers:
        train_accu, test_accu, cost = train(train_set, train_class, test_set, test_class, 0.0002, 25, 0.0, 10000, i, True)
        print(cost)
        aics.append(akaike_information_criterion(cost, i))
    fig = plt.figure()
    plt.plot(layer_num, aics, "o-", label="AIC")
    plt.title('AIC Based on layers')
    plt.xlabel('Number of layers')
    plt.ylabel('AIC')
    plt.legend(loc="lower right")
    fig.savefig("./result/AIC_tuning.png")



def evaluate():
    accu_list = []
    #prec_list = []
    #recall_list = []
    #f_list = []

    #learning rate
    learning_rates = [0.0001, 0.0002, 0.0003, 0.0004]
    for i in learning_rates:
        average_accu = cross_validation(evaluation_set, evaluation_class, i, 25, 25, 0.5, 2000, [train_set.shape[0], 25, 12, 5])
        accu_list.append(average_accu)
        #prec_list.append(average_prec.item())
        #recall_list.append(average_recall.item())
        #f_list.append(average_f.item())
    
    fig = plt.figure()
    plt.plot(learning_rates, accu_list, "o-", label="Accuracy")
    #plt.plot(learning_rates, prec_list, "o-", label="Precision")
    #plt.plot(learning_rates, recall_list, "o-", label="Recall")
    plt.title('Accuracy Based on Learning Rates')
    plt.xlabel('Learning Rates')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    fig.savefig("./result/learning_rate_tuning.png")

    accu_list = []
    #prec_list = []
    #recall_list = []
    #f_list = []
    #hidden layer number
    layers = [[train_set.shape[0],5], [train_set.shape[0], 12, 5], [train_set.shape[0], 25, 12, 5], [train_set.shape[0], 32, 25, 12, 5]]
    layer_num = [1,2,3,4]
    for i in layers:
        average_accu = cross_validation(evaluation_set, evaluation_class, 0.0002, 25, 25, 0.5, 2000, i)
        accu_list.append(average_accu)
        #prec_list.append(average_prec.item())
        #recall_list.append(average_recall.item())
        #f_list.append(average_f.item())
    fig = plt.figure()
    plt.plot(layer_num, accu_list, "o-", label="Accuracy")
    #plt.plot(layer_num, prec_list, "o-", label="Precision")
    #plt.plot(layer_num, recall_list, "o-", label="Recall")
    plt.title('Accuracy Based on hidden_layers')
    plt.xlabel('Number of layers')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    fig.savefig("./result/hidden_layer_tuning.png")

    accu_list = []
    #prec_list = []
    #recall_list = []
    #f_list = []
    #weight decay
    weight_decay = [0.0, 0.5, 1.0, 1.5]
    for i in weight_decay:
        average_accu = cross_validation(evaluation_set, evaluation_class, 0.0002, 25, 25, i, 2000, [train_set.shape[0], 25, 12, 5])
        accu_list.append(average_accu)
        #prec_list.append(average_prec.item())
        #recall_list.append(average_recall.item())
        #f_list.append(average_f.item())
    fig = plt.figure()
    plt.plot(weight_decay, accu_list, "o-", label="Accuracy")
    #plt.plot(weight_decay, prec_list, "o-", label="Precision")
    #plt.plot(weight_decay, recall_list, "o-", label="Recall")
    plt.title('Accuracy Based on Learning Rates')
    plt.xlabel('Weight Decay')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    fig.savefig("./result/weight_decay_tuning.png")


def predict(test_set_orig, test_set, test_class, element, parameters):
    params = unpickle(parameters)
    data = test_set[:,element].reshape((test_set[:,element].shape[0],1))
    #print(data.shape)
    start = time.time()
    prob, tmps = for_prop(data, params)
    end = time.time()
    #print(prob[:, 0])
    types = prob[:, 0].shape[0]
    # print(types)
    for i in range(0, prob.shape[1]):
        m_index = np.argmax(prob[:, i])
        # print(m_index)
        for j in range(0, types):
            if j == m_index:
                prob[j][i] = 1
                # print("prob")
                # print(prob[j][i])
            else:
                prob[j][i] = 0

    print("process time: " + str(end - start))
    if (prob == test_class[:, element].reshape((test_class[:, element].shape[0],1))).all():
        print("Correctly identified as")
    else:
        print("Wrongly identified as")

    if prob[0] == 1:
        print("Airplane")
    elif prob[1] == 1:
        print("Automobile")
    elif prob[2] == 1:
        print("Bird")
    elif prob[3] == 1:
        print("Cat")
    elif prob[4] == 1:
        print("Deer")
    plt.imshow(np.asnumpy(test_set_orig[element]/255)) 
    plt.show()
    

    



#train
'''
train(train_set, train_class, test_set, test_class, 0.0001, 25, 1.0, 2000, [train_set.shape[0],32, 25, 12, 5], False)
average_accu, average_prec, average_recall, average_f = cross_validation(evaluation_set, evaluation_class, 0.0002, 25, 25, 0.5, 2000, [train_set.shape[0], 32, 25, 12, 5])
print(average_accu)
print(average_prec)
print(average_recall)
print(average_f)
'''

#evaluate parameters using cross validation
'''
evaluate()
'''

#evaluate parameters using Akaike information criterion
#aic_evaluation([[train_set.shape[0],5], [train_set.shape[0], 12, 5], [train_set.shape[0], 25, 12, 5], [train_set.shape[0], 32, 25, 12, 5]])

#predict
predict(test_set_orig, test_set, test_class, 1234, "params_0.5.pickle")