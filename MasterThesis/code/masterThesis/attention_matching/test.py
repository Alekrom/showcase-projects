import os

import logging
import sys

# If the python binding doesnt work mxnet can be added manually
#sys.path.insert(1, '/work/mxnet/python/')

from mxnet import nd
from attention_matching import get_student_output_dict, get_teacher_output_dict, convOutputsDictTeacher, convOutputsDictStudent, layer_names, calculate_attention_loss

layer_name_1 = "test"
layer_name_2 = "test2"
layer_names.add(layer_name_1)
layer_names.add(layer_name_2)
teacher_out = nd.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape(2,2,2,2)
student_out = nd.array([17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]).reshape(2,2,2,2)
convOutputsDictTeacher[layer_name_1] = teacher_out
convOutputsDictStudent[layer_name_1] = student_out
convOutputsDictTeacher[layer_name_2] = teacher_out
convOutputsDictStudent[layer_name_2] = student_out
loss = calculate_attention_loss()

print(loss)
