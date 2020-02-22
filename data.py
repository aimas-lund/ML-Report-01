from auxiliary import one_out_of_k
import numpy as np

def load(path):
    file = open(path, 'r')
    path = file.readlines()
    file.close()

    return path

def format(data):

    remove_index = []

    for i in range(len(data)):
        if "NA" in data[i]:
            remove_index.append(i)

    for index in sorted(remove_index, reverse=True):
        del data[index]

    for j in range(len(data)):
        data[j] = data[j].replace('\n','').split(' ')

        for k in range(len(data[j])):
            data[j][k] = int(data[j][k])

    data = np.array(data)
    print(data.shape)
    return data

path = "./res/marketing.data"
data = load(path)
format(data)
