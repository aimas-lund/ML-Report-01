from auxiliary import one_out_of_k
import numpy as np

def load(path):
    file = open(path, 'r')
    path = file.readlines()
    file.close()

    return path

def format_data(data):
    remove_index = []
    init_lines = len(data)

    # finds all rows with an NA and adds the index to a list of
    for i in range(len(data)):
        if "NA" in data[i]:
            remove_index.append(i)

    # remove indexes in data specified in remove_list
    for index in sorted(remove_index, reverse=True):
        del data[index]

    # remove newlines and transform from string to integer
    for j in range(len(data)):
        data[j] = data[j].replace('\n','').split(' ')

        for k in range(len(data[j])):
            data[j][k] = int(data[j][k])

    # transform list to numpy array
    data = np.array(data)

    rows_removed = init_lines-data.shape[0]
    print("-------------------------------------------------")
    print("Successfully loaded data and removed rows with NA")
    print("Initial row count: {}".format(init_lines))
    print("Final row count: {}".format(data.shape[0]))
    print("Number of attributes: {}".format(data.shape[1]))
    print("Rows removed: {} ({}%)".format(rows_removed, round(rows_removed/data.shape[0]*100,1)))

    return data

#path = "./res/marketing.data"
#data = load(path)
