import os

import sys

# If the python binding doesnt work mxnet can be added manually
#sys.path.insert(1, '/work/mxnet/python/')

from mxnet import nd
from mxnet.gluon.loss import KLDivLoss
import mxnet as mx
from shared_weight_network import SharedWeightNetwork
from util.arg_parser import get_parser
from datasets.util import get_num_classes
import gluoncv
from training.util import test, get_data_iters
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric

class Top1Agreement(mx.metric.EvalMetric):
    def __init__(self, num=None):
        super(Top1Agreement, self).__init__('top1_agreement', num)

    def update(self, teacher_output, student_output):
        self.sum_metric += self.calculate_top1_agreement(teacher_output, student_output)
        self.num_inst += 1

    def calculate_top1_agreement(self, teacher_output, student_output):
        teacher_softmaxed = nd.softmax(teacher_output)
        teacher_argmax = nd.argmax(teacher_softmaxed, axis=1)

        student_softmaxed = nd.softmax(student_output)
        student_argmax = nd.argmax(student_softmaxed, axis=1)

        num_equal = nd.sum(teacher_argmax == student_argmax)

        num_equal = num_equal.asscalar()
        agreement = num_equal / teacher_output.shape[0]
        return agreement


class KLAgreement(mx.metric.EvalMetric):
    def __init__(self, num=None):
        super(KLAgreement, self).__init__('kl_agreement', num)

    def update(self, teacher_output, student_output):
        self.sum_metric += self.calculate_avg_predictive_kl(teacher_output, student_output)
        self.num_inst += 1

    def calculate_avg_predictive_kl(self, teacher_output, student_output):
        kl_div = KLDivLoss(from_logits=False)
        student_softmaxed = nd.softmax(student_output)
        kl_div_values = kl_div(teacher_output, student_softmaxed)
        kl_div_sum = nd.sum(kl_div_values).asscalar()
        agreement = kl_div_sum / teacher_output.shape[0]
        return agreement

def test_for_bit(bit, student_model, metrics_acc_student, metrics_agreement, val_data, batch_fn, teacher_out):
    metric_acc = metrics_acc_student[bit]
    metric_agreement = metrics_agreement[bit]
    metric_acc, metric_agreement = test(student_model.get_net_with_bits(bit), metric_acc, context, val_data, batch_fn,
                                        False,
                                        teacher_out, False, metric_agreement)
    name_acc, acc = metric_acc.get()
    name_agr, agr = metric_agreement.get()
    print(f'{bit},{acc[0]},{acc[1]}, {agr[0]}, {agr[1]}')
    metrics_acc_student[bit] = metric_acc


def calculate_agreements(teacher_output, student_output, top1_agr_metric, kl_agr_metric):
    top_agreement = top1_agr_metric.update(teacher_output, student_output)
    kl_agreement = kl_agr_metric.update(teacher_output, student_output)
    return top_agreement, kl_agreement

def compare_agreements_teachers(teacher1, teacher2, opt, context):
    mx.random.seed(opt.seed)

    batch_size, dataset, classes = opt.batch_size, opt.dataset, get_num_classes(opt.dataset)

    num_gpus = len(context)
    batch_size *= max(1, num_gpus)
    opt.batch_size = batch_size

    metric_teacher1 = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"teacher1")
    metric_teacher2 = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"teacher2")
    metric_agreement = CompositeEvalMetric([Top1Agreement(), KLAgreement()],
                                                       name=f"Agreement")

    train_data, val_data, batch_fn = get_data_iters(opt)

    metric_teacher1, teacher1_out = test(teacher1, metric_teacher1, context, val_data, batch_fn, False, None, True)
    name, acc = metric_teacher1.get()
    print(f'teacher1 acc: {name[0]}={acc[0]} {name[1]}={acc[1]}')

    metric_acc, metric_agreement = test(teacher2, metric_teacher2, context, val_data, batch_fn,
                                        False,
                                        teacher1_out, False, metric_agreement)
    name, acc = metric_teacher2.get()
    print(f'teacher2 acc: {name[0]}={acc[0]} {name[1]}={acc[1]}')

    name_acc, acc = metric_acc.get()
    name_agr, agr = metric_agreement.get()
    print(f'Agreement,{acc[0]},{acc[1]}, {agr[0]}, {agr[1]}')


def compare_agreements(student_model, teacher, opt, context, only_bit=None):
    mx.random.seed(opt.seed)

    batch_size, dataset, classes = opt.batch_size, opt.dataset, get_num_classes(opt.dataset)

    num_gpus = len(context)
    batch_size *= max(1, num_gpus)
    opt.batch_size = batch_size

    metrics_acc_student = {}
    for bit_width in opt.bit_widths:
        metrics_acc_student[bit_width] = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Eval Bit width {bit_width}")

    metric_acc_teacher = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Eval teacher")

    metrics_agreement = {}
    for bit_width in opt.bit_widths:
        metrics_agreement[bit_width] = CompositeEvalMetric([Top1Agreement(), KLAgreement()],
                                                           name=f"Agreement Bit width {bit_width}")

    train_data, val_data, batch_fn = get_data_iters(opt)

    metric_acc_teacher, teacher_out = test(teacher, metric_acc_teacher, context, val_data, batch_fn, False, None, True)
    name, acc = metric_acc_teacher.get()
    print(f'teacher acc: {name[0]}={acc[0]} {name[1]}={acc[1]}')

    print("bits, acc_1, acc_5, agr_top, agr_kl")

    if only_bit is None:
        for bit in metrics_acc_student.keys():
            test_for_bit(bit, student_model, metrics_acc_student, metrics_agreement, val_data, batch_fn, teacher_out)
    else:
        test_for_bit(only_bit, student_model, metrics_acc_student, metrics_agreement, val_data, batch_fn, teacher_out)






if __name__ == '__main__':

    parser = get_parser()
    opt = parser.parse_args()
    result_path = opt.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]
    if opt.dry_run:
        context = [mx.cpu()]
    """
    teacher = gluoncv.model_zoo.get_model('cifar_resnet110_v2', classes=100)
    teacher.hybridize()
    teacher.load_parameters(f'../teacher_params/cifar_resnet110_v2.params', ctx=context)

    #student_model = SharedWeightNetwork(opt, context, opt.bit_widths, None)
    #student_model.load_parameters(f'../teacher_params/resnet110_v2_unext_simple.params', ctx=context)

    student_model = gluoncv.model_zoo.get_model('cifar_resnet110_v2', classes=100)
    student_model.load_parameters(f'../teacher_params/resnet110_v2_unext_simple.params', ctx=context)

    #compare_agreements(student_model, teacher, opt, context, False)
    compare_agreements_teachers(student_model, teacher, opt, context)
    """