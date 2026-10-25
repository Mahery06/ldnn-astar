#deter the appropriate weights of the two elements in modules:value

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy.typing import NDArray
from neural_trainset import x_train,y_train
import random

x_train=np.array(x_train)
y_train=np.array(y_train)

def variance(tensor:NDArray):
    total=0
    for i in range(tensor.__len__()):
        total+=tensor[i]
    average=total/tensor.__len__()
    total=0
    for i in range(tensor.__len__()):
        total+=(tensor[i]-average)**2
    return total/tensor.__len__()


def create_model():
    kernel_info_h1=[[0.037, 0.678, 0.074, 0.204, 0.193, 0.033, 0.846, 0.213, 0.819, 0.205, 0.93, 0.358, 0.261, 0.392, 0.834, 0.119, 0.473, 0.4, 0.144, 0.83], [0.648, 0.626, 0.09, 0.636, 0.377, 0.953, 0.493, 0.351, 0.064, 0.442, 0.528, 0.103, 0.595, 0.712, 0.225, 0.584, 0.023, 0.283, 0.035, 0.873]]


    kernel_info_h2=[[0.132, 0.025, 0.562, 0.771, 0.954, 0.627, 0.393, 0.136, 0.591, 0.275], [0.856, 0.689, 0.489, 0.237, 0.41, 0.207, 0.655, 0.034, 0.118, 0.848], [0.659, 0.809, 0.885, 0.081, 0.561, 0.694, 0.43, 0.922, 0.538, 0.221], [0.189, 0.325, 0.53, 0.616, 0.156, 0.161, 0.721, 0.839, 0.088, 0.913], [0.632, 0.739, 0.163, 0.729, 0.305, 0.849, 0.725, 0.813, 0.876, 0.12], [0.574, 0.777, 0.627, 0.249, 0.683, 0.466, 0.797, 0.97, 0.144, 0.767], [0.065, 0.21, 0.762, 0.015, 0.367, 0.135, 0.296, 0.862, 0.139, 0.528], [0.26, 0.408, 0.14, 0.09, 0.25, 0.022, 0.25, 0.174, 0.182, 0.997], [0.851, 0.915, 0.543, 0.446, 0.44, 0.442, 0.917, 0.893, 0.15, 0.891], [0.946, 0.766, 0.097, 0.779, 0.239, 0.171, 0.633, 0.145, 0.918, 0.465], [0.615, 0.672, 0.135, 0.627, 0.651, 0.039, 0.502, 0.78, 0.213, 0.546], [0.411, 0.327, 0.428, 0.339, 0.634, 0.752, 0.597, 0.588, 0.195, 0.366], [0.835, 0.376, 0.201, 0.176, 0.9, 0.575, 0.892, 0.152, 0.966, 0.082], [0.065, 0.801, 0.654, 0.086, 0.476, 0.754, 0.227, 0.138, 0.28, 0.967], [0.146, 0.269, 0.326, 0.207, 0.247, 0.22, 0.299, 0.63, 0.826, 0.509], [0.052, 0.845, 0.079, 0.664, 0.464, 0.987, 0.581, 0.937, 0.449, 0.117], [0.615, 0.921, 0.718, 0.552, 0.387, 0.552, 0.492, 0.073, 0.647, 0.056], [0.069, 0.117, 0.616, 0.624, 0.877, 0.612, 0.529, 0.493, 0.711, 0.177], [0.503, 0.259, 0.403, 0.33, 0.483, 0.711, 0.376, 0.617, 0.819, 0.067], [0.721, 0.094, 0.874, 0.656, 0.954, 0.555, 0.263, 0.846, 0.967, 0.497]]

    kernel_info_out=[[0.982], [0.439], [0.471], [0.485], [0.555], [0.045], [0.775], [0.371], [0.007], [0.44]]

    bias_h1= [0.964, 0.177, 0.373, 0.597, 0.287, 0.95, 0.918, 0.242, 0.531, 0.407, 0.481, 0.579, 0.944, 0.704, 0.697, 0.427,
     0.326, 0.541, 0.632, 0.133]

    bias_h2=[0.814, 0.163, 0.665, 0.521, 0.608, 0.566, 0.011, 0.972, 0.09, 0.521]

    bias_out=[0.888]

    kernel_info_h1 = np.array(kernel_info_h1, dtype='float32')
    bias_h1 = np.array(bias_h1, dtype='float32')
    kernel_info_h2 = np.array(kernel_info_h2, dtype='float32')
    bias_h2 = np.array(bias_h2, dtype='float32')
    kernel_info_out = np.array(kernel_info_out, dtype='float32')
    bias_out = np.array(bias_out, dtype='float32')

    model = Sequential([
        Dense(20, activation='relu', name='h1'),
        Dense(10, activation='relu', name='h2'),
        Dense(1, activation='sigmoid')  # score between (0,1) —— 越低=越好（expanded占比小）
    ])

    model.build(input_shape=(None, 2))

    model.layers[0].set_weights([kernel_info_h1, bias_h1])
    model.layers[1].set_weights([kernel_info_h2, bias_h2])
    model.layers[2].set_weights([kernel_info_out, bias_out])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(x_train, y_train, epochs=100, batch_size=15, verbose=0)
    return model

# use sensitivity and spearman correlation
def data_analysis(model):

    table1=np.array([1]*20)
    table2=np.array([1]*10)

    idx_list1=[]
    idx_list2=[]

    extract_layer1=Model(
        inputs=model.get_layer('h1').input,
        outputs=model.get_layer('h1').output
    )
    extract_layer2=Model(
        inputs=model.get_layer('h1').input,
        outputs=model.get_layer('h2').output
    )

    for i in range(100):
        away=random.random()
        proximity=random.random()
        test=np.array([[away,proximity]])

        ans1=extract_layer1.predict(test,verbose=0)
        ans2=extract_layer2.predict(test,verbose=0)

        table1=np.vstack((table1,ans1))
        table2=np.vstack((table2,ans2))

    for i in range(20):
        tensor=table1[1:,i]
        var=variance(tensor)
        if var<0.001:
            idx_list1.append(i)

    for i in range(10):
        tensor=table2[1:,i]
        var=variance(tensor)
        if var<0.0001:
            idx_list2.append(i)

    return idx_list1,idx_list2

def modify_model(model):
    idx_list1,idx_list2=data_analysis(model)

    layer1_weight=model.layers[0].get_weights()[0]
    layer1_bias=model.layers[0].get_weights()[1]
    layer2_weight=model.layers[1].get_weights()[0]
    layer2_bias=model.layers[1].get_weights()[1]
    out_weight=model.layers[2].get_weights()[0]
    out_bias=model.layers[2].get_weights()[1]

    layer1_new_weight=np.delete(layer1_weight,idx_list1,axis=1)
    layer1_new_bias=np.delete(layer1_bias,idx_list1)
    layer2_new_weight=np.delete(layer2_weight,idx_list1,axis=0)
    layer2_new_weight=np.delete(layer2_new_weight,idx_list2,axis=1)
    layer2_new_bias=np.delete(layer2_bias,idx_list2)
    out_new_weight=np.delete(out_weight,idx_list2,axis=0)

    new_model=Sequential([
        Dense(20-len(idx_list1), activation='relu', name='hidden1'),
        Dense(10-len(idx_list2), activation='relu', name='hidden2'),
        Dense(1,activation='sigmoid')]
    )
    new_model.build(input_shape=(None,2))
    new_model.layers[0].set_weights([layer1_new_weight, layer1_new_bias])
    new_model.layers[1].set_weights([layer2_new_weight, layer2_new_bias])
    new_model.layers[2].set_weights([out_new_weight, out_bias])

    new_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    new_model.fit(x_train, y_train, epochs=10,verbose=0)
    return new_model


attribute_model=modify_model(create_model())


