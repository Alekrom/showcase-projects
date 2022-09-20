import sys
import logging

# If the python binding doesnt work mxnet can be added manually
#sys.path.insert(1, '../work/mxnet/python/')

import os
import mxnet as mx
from agreement_calculation import Top1Agreement, KLAgreement
#from ..agreement_calculation import Top1Agreement, KLAgreement
import csv
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from mxnet.gluon.loss import KLDivLoss
from mxnet import nd
from datasets.test_images import get_test_image_loader
from shared_weight_network import SharedWeightNetwork
import gluoncv
from util.arg_parser import get_parser
import matplotlib.pyplot as plt
from operator import add
from collections import OrderedDict


def plot_class_stats(class_stats, plot_path, name, overall_stats):
    num_img_per_class = 100
    print(overall_stats)
    plt.clf()
    plt.figure(figsize=(12, 3))
    both_true_list = [float(overall_stats["both_true"]) * num_img_per_class]
    agree_false_list = [float(overall_stats["agree_false"]) * num_img_per_class]
    teacher_only_list = [float(overall_stats["teacher_only"]) * num_img_per_class]
    student_only_list = [float(overall_stats["student_only"]) * num_img_per_class]
    disagree_false = [float(overall_stats["disagree_false"]) * num_img_per_class]

    labels = ['avg']

    # sort class stats
    class_stats = OrderedDict(sorted(class_stats.items(), key=lambda x: x[1]["both_true"]))


    for label, value in class_stats.items():
        both_true_list.append(value["both_true"])
        teacher_only_list.append(value["teacher_only"])
        student_only_list.append(value["student_only"])
        agree_false_list.append(value["agree_false"])
        disagree_false.append(value["disagree_false"])
        labels.append(str(label))

    width = 0.7
    plt.rc('xtick', labelsize=5)


    plt.bar(labels, both_true_list, width, label="both_true")
    bottom = both_true_list
    plt.bar(labels, teacher_only_list, width, bottom=bottom, label="teacher_only")
    bottom = list(map(add, bottom, teacher_only_list))
    plt.bar(labels, student_only_list, width, bottom=bottom, label="student_only")
    bottom = list(map(add, bottom, student_only_list))
    plt.bar(labels, agree_false_list, width, bottom=bottom, label="agree_false")
    bottom = list(map(add, bottom, agree_false_list))
    plt.bar(labels, disagree_false, width, bottom=bottom, label="disagree_false")

    plt.ylabel('% of total images')
    plt.title('Distribution over classes')
    plt.legend(prop={'size': 6})

    plt.savefig(f'{plot_path}{name}.pdf')
    print(f'{plot_path}{name}.pdf')
    print("plotted")

def test_images_and_log(test_images, csv_path, student, teacher, bits, ctx):
    metric_acc_student = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Eval Bit width {bits}")
    metric_acc_teacher = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)], name=f"Eval Teacher")
    metric_agreement = CompositeEvalMetric([Top1Agreement(), KLAgreement()], name=f"Eval Bit width {bits}")

    num_at = 0
    num_af = 0
    num_ttsf = 0
    num_tfst = 0
    num_tfsf = 0

    class_stats = {}
    with open(f'{csv_path}_{bits}.csv', mode='a') as results_csv:
        csv_writer = csv.writer(results_csv, delimiter=',')
        csv_writer.writerow(
            ['bits', 'prediction_student', 'prediction_teacher', 'label', 'agreement', 'AT', 'AF', 'TTSF', 'TFST', 'TFSF'])
        i = 0
        for batch_idx, (x, y) in enumerate(test_images):
            x = x.as_in_context(ctx[0])
            y = y.as_in_context(ctx[0])
            label = y.asscalar()
            if label not in class_stats.keys():
                class_stats[label] = {"both_true": 0, "teacher_only": 0, "student_only": 0, "agree_false": 0, "disagree_false": 0}
            output_student = student(x)
            pred_label_student = nd.argmax(output_student, axis=1).asscalar()
            metric_acc_student.update(y, output_student)
            output_teacher = teacher(x)
            pred_label_teacher = nd.argmax(output_teacher, axis=1).asscalar()
            metric_acc_teacher.update(y, output_teacher)
            metric_agreement.update(output_teacher, output_student)
            agreement = pred_label_student == pred_label_teacher

            at = agreement and (pred_label_teacher == label)
            if at:
                num_at = num_at + 1
                class_stats[label]["both_true"] += 1

            af = agreement and (pred_label_teacher != label)
            if af:
                num_af = num_af + 1
                class_stats[label]["agree_false"] += 1

            ttsf = (pred_label_teacher == label) and (pred_label_student != label)
            if ttsf:
                num_ttsf = num_ttsf + 1
                class_stats[label]["teacher_only"] += 1

            tfst = (pred_label_teacher != label) and (pred_label_student == label)
            if tfst:
                num_tfst = num_tfst + 1
                class_stats[label]["student_only"] += 1

            tfsf = (pred_label_teacher != label) and (pred_label_student != label) and (pred_label_teacher != pred_label_student)
            if tfsf:
                num_tfsf = num_tfsf + 1
                class_stats[label]["disagree_false"] += 1

            #csv_writer.writerow([bits, pred_label_student, pred_label_teacher, label, agreement, at, af, ttsf, tfst])
            i = i+1

        at_overall = num_at/i
        af_overall = num_af/i
        ttsf_overall = num_ttsf / i
        tfst_overall = num_tfst / i
        tfsf_overall = num_tfsf / i

        csv_writer.writerow([])
        name, acc = metric_acc_student.get()
        csv_writer.writerow([f"Accuracy student {acc[0]}"])
        name, acc = metric_acc_teacher.get()
        csv_writer.writerow([f"Accuracy teacher {acc[0]}"])
        name, acc = metric_agreement.get()
        csv_writer.writerow([f"Agreement {acc[0]}"])
        csv_writer.writerow([f"AT: {at_overall}"])
        csv_writer.writerow([f"AF: {af_overall}"])
        csv_writer.writerow([f"TTSF: {ttsf_overall}"])
        csv_writer.writerow([f"TFST: {tfst_overall}"])
        csv_writer.writerow([f"TFSF: {tfsf_overall}"])
        csv_writer.writerow([])
        csv_writer.writerow([f"Class stats:"])
        csv_writer.writerow(["class", "both_true", "teacher_only", "student_only", "agree_false", "disagree_false"])
        for key, item in class_stats.items():
            csv_writer.writerow([key, float(item["both_true"]), float(item["teacher_only"]), float(item["student_only"]), float(item["agree_false"]), float(item["disagree_false"])])

        overall_stats = {"both_true": at_overall, "teacher_only": ttsf_overall, "student_only": tfst_overall, "agree_false": af_overall, "disagree_false": tfsf_overall}
        plot_class_stats(class_stats, f'{csv_path}_{bits}_', 'class_stats', overall_stats)


def test_verification_images(shared_weight_net, img_path, csv_path, num_workers, teacher, ctx):

    test_images = get_test_image_loader(img_path, num_workers)

    for bit_width in shared_weight_net.bit_widths:
        print(f"BITS: {bit_width}")
        net_b = shared_weight_net.get_net_with_bits(bit_width)
        test_images_and_log(test_images, csv_path, net_b, teacher, bit_width, ctx)



def test_img_within_ensemble(ensemble, num_workers, ctx):
    bit_combinations = [(2,4), (2,8), (2,32), (4,8), (4,32), (8,32)]
    test_images = get_test_image_loader(img_path, num_workers)

    for combi in bit_combinations:
        bit_1, bit_2 = combi
        model_1 = ensemble.get_net_with_bits(bit_1)
        model_2 = ensemble.get_net_with_bits(bit_2)

        print(f"BITS: {combi}")

        test_images_and_log(test_images, csv_path, model_1, model_2, combi, ctx)


def test_across_ensemble(ensemble, num_workers, ctx):
    test_images = get_test_image_loader(img_path, num_workers)
    # didnt need dict in actual impl, but leave because i may refactor
    model_dict = {2: ensemble.get_net_with_bits(2), 4: ensemble.get_net_with_bits(4), 8: ensemble.get_net_with_bits(8),
                 32: ensemble.get_net_with_bits(32)}

    agree_all = 0
    agree_3 = 0
    agree_2 = 0
    disagree = 0

    for batch_idx, (x, y) in enumerate(test_images):
        x = x.as_in_context(ctx[0])
        y = y.as_in_context(ctx[0])
        label = y.asscalar()
        outputs = []
        for bit, model in model_dict.items():
            output = model(x)
            outputs.append(nd.argmax(output, axis=1).asscalar())

        most_common = max(outputs, key = outputs.count)
        count = outputs.count(most_common)
        if count == 4:
            agree_all += 1
        elif count == 3:
            agree_3 += 1
        elif count == 2:
            agree_2 += 1
        elif count == 1:
            disagree += 1

    with open(f'{csv_path}_ensemble_compare.txt', 'a') as file:
        file.write(f'agree_all: {agree_all}')
        file.write(f'agree_3: {agree_3}')
        file.write(f'agree_2: {agree_2}')
        file.write(f'disagree: {disagree}')

def test_across_ensemble_kldivloss(ensemble, num_workers, ctx, with_random=False):
    kd_loss_fn = KLDivLoss(from_logits=False)
    filename = f'{csv_path}_ensemble_compare_kldiv.txt'
    if with_random:
        filename = f'{csv_path}_ensemble_compare_kldiv_with_onehot.txt'

    bit_combinations = [(2, 4), (2, 8), (2, 32), (4, 8), (4, 32), (8, 32)]
    test_images = get_test_image_loader(img_path, num_workers)
    model_dict = {2: ensemble.get_net_with_bits(2), 4: ensemble.get_net_with_bits(4), 8: ensemble.get_net_with_bits(8),
                 32: ensemble.get_net_with_bits(32)}

    losses_all = []

    for batch_idx, (x, y) in enumerate(test_images):
        losses = []
        x = x.as_in_context(ctx[0])
        y = y.as_in_context(ctx[0])
        label = y.asscalar()
        outputs_dict = {}

        for bit, model in model_dict.items():
            output = model(x)
            outputs_dict[bit] = output

        if not with_random:
            for bit_1, bit_2 in bit_combinations:
                #logger.info("output")
                #logger.info(outputs_dict[bit_1])
                losses.append(kd_loss_fn(outputs_dict[bit_1], mx.nd.softmax(outputs_dict[bit_2])).asscalar())
                #logger.info("loss")
                #logger.info(kd_loss_fn(outputs_dict[bit_1], mx.nd.softmax(outputs_dict[bit_2])))
        else:
            for bit in ensemble.bit_widths:
                # random
                #rand_nums = nd.random.uniform(0, 100, shape=(1,100), ctx=ctx[0])
                #losses.append(kd_loss_fn(outputs_dict[bit], mx.nd.softmax(rand_nums)).asscalar())

                # one hot
                mx.npx.set_np()

                one_hot = mx.npx.one_hot(y.as_np_ndarray(), 100)
                #logger.info(label)
                #logger.info(one_hot)
                one_hot = one_hot.as_nd_ndarray()
                mx.npx.reset_np()
                losses.append(kd_loss_fn(outputs_dict[bit], mx.nd.softmax(one_hot)).asscalar())

        losses_avg = sum(losses) / len(losses)
        losses_all.append(losses_avg)

        with open(filename, 'a') as file:
            file.write(f'img num {batch_idx} \n')
            file.write(str(losses))
            file.write(str(losses_avg))


    losses_all_avg = (sum(losses_all) / len(losses_all))
    with open(filename, 'a') as file:
        file.write(f'final\n')
        file.write(f'{losses_all}\n')
        file.write(f'final avg {str(losses_all_avg)}\n')


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log_prefix = opt.log
    logger = logging.getLogger()
    logging.debug('\n%s', '-' * 100)

    context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]

    #teacher = gluoncv.model_zoo.get_model('cifar_resnet20_v2', classes=100)
    #teacher.hybridize()
    #teacher.load_parameters(f'../../results_cifar100_final/ensemble_328421_kd_unext_simple_resnet202021-08-09_13:44:13.894148_best_params_teacher.params', ctx=context)
    #
    #student = SharedWeightNetwork(opt, context, opt.bit_widths, None)
    #student.load_parameters(f'../../results_cifar100_final/ensemble_328421_kd_unext_simple_resnet202021-08-09_13:44:13.894148_best_params.params', ctx=context)

    student_model = SharedWeightNetwork(opt, context, opt.bit_widths, None, True)
    student_model.load_parameters(f'../../teacher_params_final/ensemble_high4_unext_resnet110_progressive_fp_weights2021-11-09_14:14:22.149366_best_params.params', ctx=context)

    #for part in ["test", "train"]:
    for part in ["train"]:
        img_path = f"../../../cifar100testimg/{part}"
        csv_path = f'../../verification_images/cifar100_final/{part}/ensemblecompare_{part}'
        #test_img_within_ensemble(student_model, opt.num_workers, context)
        #test_across_ensemble(student_model, opt.num_workers, context)
        #test_across_ensemble_kldivloss(student_model, opt.num_workers, context)
        test_across_ensemble_kldivloss(student_model, opt.num_workers, context, with_random=True)

