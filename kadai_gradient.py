import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import pickle


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


def convert_image(train_set_orig):
    converted_image = train_set_orig["data"].astype(np.uint8)
    gray = np.empty((train_set_orig["data"].shape[0],
                     train_set_orig["data"].shape[1], train_set_orig["data"].shape[2]), dtype=np.uint8)
    for i in range(train_set_orig["data"].shape[0]):
        gray[i] = cv2.cvtColor(converted_image[i], cv2.COLOR_RGB2GRAY)
    return gray


def load_dataset(path):
    dataset = unpickle(path)
    dataset = select(dataset)
    dataset["data"] = reshape_pic_data(dataset)
    #dataset["data"] = convert_image(dataset)
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

train_class = np.asarray(train_class)
train_class = train_class.reshape([train_class.shape[0], 1])
test_class = np.asarray(test_class)
test_class = test_class.reshape([test_class.shape[0], 1])

# print(train_set.shape)
# print(train_class.shape)
import cupy as np
train_set = np.asarray(train_set)
train_class = np.asarray(train_class)
test_set = np.asarray(test_set)
test_class = np.asarray(test_class)

# flatten and normalize
train_set = train_set.reshape(train_set.shape[0], -1).T / 255.0
test_set = test_set.reshape(test_set.shape[0], -1).T / 255.0
print(train_set.shape)

# convert to onehot
train_class = np.eye(5)[train_class.reshape(-1)].T
test_class = np.eye(5)[test_class.reshape(-1)].T
# print(test_class.shape)





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


def back_prop_step(da, tmp, activation_type, lamb, batch_class):
    train_class = batch_class
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


def back_prop(tmps, batch_class):
    derivs = {}
    layer_num = len(tmps)

    derivs["da" + str(layer_num-1)], derivs["dw" + str(layer_num)], derivs["db" +
                                                                           str(layer_num)] = back_prop_step(None, tmps[-1], "softmax", 0.0, batch_class)
    for i in reversed(range(layer_num-1)):
        da_prev, dw, db = back_prop_step(
            derivs["da" + str(i+1)], tmps[i], "relu", 0.0, batch_class)
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


def learn(train_set, train_class, learning_rate, iteration):
    params = init_params([train_set.shape[0], 25, 12, 5])
    cost = 0
    cost_prev = 0
    cost_prev_prev = 0
    train_accu = 0.0
    test_accu = 0.0
    global test_set
    global test_class
    costs = []
    train_accuracies = []
    test_accuracies = []
    for i in range(0, iteration):
        a, tmps = for_prop(train_set, params)
        cost_prev_prev = cost_prev
        cost_prev = cost
        cost = ce_loss_l2(a, train_class,params, 0.0)
        if cost < 0.2 or test_accu > 0.8:
            print("Cost after iteration {} : {}" .format(i, np.squeeze(cost)))
            print("learning rate: {}" .format(learning_rate))
            break
        if cost_prev_prev < cost_prev and cost_prev < cost:
            learning_rate = learning_rate * 0.9
        derivs = back_prop(tmps, train_class)
        params = update(params, derivs, learning_rate)
        print("Cost after iteration {} : {}" .format(i, np.squeeze(cost)))
        print("learning rate: {}" .format(learning_rate))
        
        if i % 1000 == 0:
            prob1, train_accu = accuracy(train_set, train_class, params)
            prob2, test_accu = accuracy(test_set, test_class, params)
            
            print("Train accuracy: {}" .format(train_accu))
            print("Test accuracy: {}" .format(test_accu))
            train_accuracies.append(train_accu)
            test_accuracies.append(test_accu)

        costs.append(cost.item())
    return params, costs, train_accuracies, test_accuracies

'''
def learn(train_set, train_class, learning_rate, batch_num, epochs):
    cost = 0
    cost_prev = 0
    cost_prev_prev = 0
    train_accu = 0.0
    test_accu = 0.0
    global test_set
    global test_class
    costs = []
    train_accuracies = []
    test_accuracies = []

    # initialize params
    params = init_params([train_set.shape[0], 25, 12, 5])
    
    for i in range(0, epochs):
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
            cost = ce_loss_l2(a, train_classes[j], params, 0.0)
            
            if cost < 0.2 or test_accu > 0.8:
                print("Cost after epoch {} : {}" .format(i, np.squeeze(cost)))
                print("learning rate: {}" .format(learning_rate))
                break
            if cost_prev_prev < cost_prev and cost_prev < cost:
                learning_rate = learning_rate * 1
            derivs = back_prop(tmps, train_classes[j])
            
            params = update(params, derivs, learning_rate)

            
            
            
            #print("here")

            j += 1
       
        costs.append(cost.item())
        
        # if i % 100 == 0:
        print("Cost after epoch {} : {}" .format(i, np.squeeze(cost)))
        print("learning rate: {}" .format(learning_rate))
        #print("Train accuracy: {}" .format(train_accu))
        #print("Test accuracy: {}" .format(test_accu))
        seed += 1
        # print(i)
        if i % 1000 == 0:
            prob1, train_accu = accuracy(
                train_set, train_class, params)
            prob2, test_accu = accuracy(
                test_set, test_class, params)
            train_accuracies.append(train_accu)
            test_accuracies.append(test_accu)
            print("Train accuracy: {}" .format(train_accu))
            print("Test accuracy: {}" .format(test_accu))

    return params, costs, train_accuracies, test_accuracies

'''
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


#print("learning rate = 0.0005")
#params = learn(train_set, train_class, 0.0005, 3500)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
#print("learning rate = 0.0004")
#learn(train_set, train_class, 0.0004, 3500)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
#print("learning rate = 0.0003")
#learn(train_set, train_class, 0.0003, 4000)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
#print("learning rate = 0.0002")
#learn(train_set, train_class, 0.0002, 4000)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
print("learning rate = 0.0002")
params, costs, train_accuracies, test_accuracies = learn(
    train_set, train_class, 0.0002, 10000)
prob1, train_accu = accuracy(train_set, train_class, params)
prob2, test_accu = accuracy(test_set, test_class, params)
print("Train accuracy: {}" .format(train_accu))
print("Test accuracy: {}" .format(test_accu))
train_accuracies.append(train_accu)
test_accuracies.append(test_accu)

path = './costs/kadai_gradient.txt'
f = open(path,'w')
f.write(str(costs[-1]))
f.close

with open('./params_gradient.pickle', mode='wb') as f:
    pickle.dump(params, f)


fig = plt.figure()
plt.plot(costs, label="cost")
plt.title('cost')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(loc="lower right")
fig.savefig("./result/cost_gradient.png")
# plt.show()

fig = plt.figure()
plt.plot(train_accuracies, label="train_accuracy")
plt.plot(test_accuracies, label="test_accuracy")
plt.title('Accuracy')
plt.xlabel('epoch (per 1000)')
plt.ylabel('accuracy')
plt.legend(loc="lower right")
fig.savefig("./result/accuracy_gradient.png")
# plt.show()


#print("learning rate = 0.00005")
#learn(train_set, train_class, 0.00005, 6000)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
#print("learning rate = 0.00001")
#learn(train_set, train_class, 0.00001, 10000)
#accuracy(train_set, train_class, params)
#accuracy(test_set, test_class, params)
