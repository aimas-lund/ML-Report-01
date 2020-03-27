# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:44:37 2020

@author: Team Forest
"""
import numpy as np


def add_elements_to_list(list,
                         input_list,
                         index,
                         added_string=''):
    if type(list) == list:
        if added_string == '':
            for i in range(len(input_list)):
                list.insert(-index, added_string + str(input_list[-i]))
        else:
            for i in range(len(input_list)):
                list.insert(-index, input_list[-i])
    else:  # assuming if not list, then the input is numpy type
        input_list = np.array(input_list, dtype=object)
        if added_string != '':
            for i in range(len(input_list)):
                input_list[i] = added_string + str(int(input_list[i]))
        list = np.insert(list, index, input_list)
    return list


def one_out_of_k(input, column_index=0, return_uniques=False):
    """
    Function replacing a specified column of matrix with the One-out-of-K equivalent.
    :param input:
    :param column_index:
    :param return_uniques: 1xN Numpy array, boolean specifying if the unique values of the input should be returned
    :return: The One-out-of-K numpy-matrix and optionally the unique values from the input matrix.
    """
    chosen_column = input[:, column_index]
    uniques = np.unique(chosen_column)

    # creates a matrix of 0's with number of unique entries as columns 
    # and number of input rows as row numbers
    output = np.array(np.zeros(len(uniques) * len(chosen_column))).reshape(len(chosen_column), len(uniques))

    # create a dictionary to lookup the corresponding index of the values in input
    lookup = dict(zip(uniques, np.arange(len(uniques))))

    # insert 1 in correct columns, iterating through the rows
    for i in range(len(chosen_column)):
        j = lookup[chosen_column[i]]
        output[i, j] = 1

    output = np.hstack((np.hstack((input[:, :column_index], output)), input[:, column_index + 1:]))

    # output uniques and output binary matrix in tuple
    if return_uniques:
        return uniques, output
    else:
        return output


def every_nth(input, n, iteration=1):
    output = input

    for i in range(iteration):
        output = output[np.mod(np.arange(output.size), n) != 0]

    return output


def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def get_percentiles(x, lower=10., upper=90.):
    return np.percentile(np.array(x), lower), np.percentile(np.array(x), upper)


def get_limits(x):
    return min(x), max(x)
