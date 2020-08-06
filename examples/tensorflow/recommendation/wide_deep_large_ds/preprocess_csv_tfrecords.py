#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import pandas
import argparse
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
parser = argparse.ArgumentParser()
parser.add_argument('--inputcsv-datafile', type=str,
                    help='full path of data file e.g. eval.csv',
                    dest='evaldatafile_path',
                    required=True)
parser.add_argument('--calibrationcsv-datafile', type=str,
                    help='full path of data file of calibration/train dataset to get normalization ranges',
                    dest='traindatafile_path',
                    default='NULL',
                    required=False)

parser.add_argument('--outputfile-name', type=str,
                    help='output tfrecord file name e.g. processed_eval.[tfrecords]',
                    dest='outputfile_path',
                    default="processed_data.tfrecords",
                    required=False)

args = parser.parse_args()

eval_csv_file = args.evaldatafile_path
train_csv_file = args.traindatafile_path
output_file = args.outputfile_path

if not os.path.isfile(eval_csv_file):
    print("Please input a valid csv file")
    sys.exit(1)

filename, file_ext = os.path.splitext(output_file)
in_filename, _ = os.path.splitext(os.path.basename(eval_csv_file))

if file_ext != ".tfrecords":
    output_file = output_file + ".tfrecords"

output_file = "{}_{}".format(in_filename,output_file)
csv = pandas.read_csv(eval_csv_file, header=None)
if len(csv.columns)==39:
    dataset_type = 'test'
else:
    dataset_type = 'eval'
fill_na_dict  = {}
if dataset_type=='test':
    for i in range(0,13):
        fill_na_dict[i]=0.0
    for i in range(13,39):
        fill_na_dict[i]=""
else:
    for i in range(1,14):
        fill_na_dict[i]=0.0
    for i in range(14,40):
        fill_na_dict[i]=""
csv=csv.fillna(value=fill_na_dict).values
numeric_feature_names = ["numeric_1"]
string_feature_names = ["string_1"]
LABEL_COLUMN =["clicked"]
CATEGORICAL_COLUMNS1 = ["C"+str(i)+"_embedding" for i in range(1, 27)]
NUMERIC_COLUMNS1 = ["I"+str(i) for i in range(1, 14)]
if dataset_type=='eval':
    DATA_COLUMNS = LABEL_COLUMN + NUMERIC_COLUMNS1 + CATEGORICAL_COLUMNS1
else:
    DATA_COLUMNS = NUMERIC_COLUMNS1 + CATEGORICAL_COLUMNS1
CATEGORICAL_COLUMNS2 = ["C"+str(i)+"_embedding" for i in range(1, 27)]
NUMERIC_COLUMNS2 = ["I"+str(i) for i in range(1, 14)]

CATEGORICAL_COLUMNS1.sort()
NUMERIC_COLUMNS1.sort()
no_of_rows = 0
with open(eval_csv_file, 'r') as f:
    if not os.path.isfile(train_csv_file):
        nums=[line.strip('\n\r').split(',') for line in f.readlines()]
    else: 
        f1 = open(train_csv_file, 'r')
        nums=[line.strip('\n\r').split(',') for line in f.readlines(
        )]+[line.strip('\n\t').split(',') for line in f1.readlines()]
    numpy_arr = np.array(nums)
    numpy_arr[numpy_arr=='']='0'
    min_list,max_list,range_list = [],[],[]
    for i in range(len(DATA_COLUMNS)):
        if DATA_COLUMNS[i] in NUMERIC_COLUMNS1:
            col_min = numpy_arr[:,i].astype(np.float32).min()
            col_max = numpy_arr[:,i].astype(np.float32).max()
            min_list.append(col_min)
            max_list.append(col_max)
            range_list.append(col_max-col_min)
    if os.path.isfile(train_csv_file):
        f1.close()
    print('min list',min_list)
    print('max list',max_list)
    print('range list',range_list)


with tf.python_io.TFRecordWriter(output_file) as writer:
    print('*****Processing data******')
    for row in csv:
        no_of_rows = no_of_rows+1
        if dataset_type == 'eval':
            unnormalized_vals = np.array(row[1:14])
        else:
            unnormalized_vals = np.array(row[0:13])
        normalized_vals = (unnormalized_vals-min_list)/range_list
        if dataset_type == 'eval':
            new_categorical_dict = dict(zip(CATEGORICAL_COLUMNS2, row[14:40]))
        else:
            new_categorical_dict = dict(zip(CATEGORICAL_COLUMNS2, row[13:39]))
        new_categorical_list = []
        for i in CATEGORICAL_COLUMNS1:
            if pandas.isnull(new_categorical_dict[i]):
                new_categorical_list.append("")
            else:
                new_categorical_list.append(new_categorical_dict[i])
        hash_values =  tf.string_to_hash_bucket_fast(
            new_categorical_list, 1000).numpy()
        new_numerical_dict = dict(zip(NUMERIC_COLUMNS2, normalized_vals))
        example = tf.train.Example()
        for i in NUMERIC_COLUMNS1:
            example.features.feature[numeric_feature_names[0]].float_list.value.extend([new_numerical_dict[i]])
        for i in range(0, 26):
            example.features.feature[string_feature_names[0]].int64_list.value.extend([i])
            example.features.feature[string_feature_names[0]].int64_list.value.extend([hash_values[i]])
        if dataset_type == 'eval':
            example.features.feature["label"].int64_list.value.append(row[0])
        writer.write(example.SerializeToString())

print('Total number of rows ', no_of_rows)
print('Generated output file name :'+output_file)
