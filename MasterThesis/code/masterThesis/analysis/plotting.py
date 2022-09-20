import matplotlib.pyplot as plt
import argparse
import os
import re
import csv



def plot_csv(csv):
    path = create_dir_for_csv_plots(csv)
    plot_csv_at_path(csv, path)


def plot_csv_at_path(csv_path, plot_path):
    with open(csv_path) as csv_file:
        top_acc_results_train = {}
        top_5_acc_results_train = {}
        top_acc_results_test = {}
        top_5_acc_results_test = {}
        agree_top_train = {}
        agree_kl_train = {}
        agree_top_test = {}
        agree_kl_test = {}

        epochs = []
        result_dicts = [top_acc_results_train, top_5_acc_results_train, top_acc_results_test, top_5_acc_results_test]
        agree_dicts = [agree_top_train, agree_kl_train, agree_top_test, agree_kl_test]
        reader = csv.DictReader(csv_file, delimiter=',')

        i = 0
        for row in reader:
            #if i < 450:
            #    i += 1
            #    continue
            bits = row['bits']
            for dict in result_dicts:
                # include teacher switch
                #if not bits in dict and 'teacher' not in bits:
                if not bits in dict and bits != 'teacher_t':
                    dict[bits] = []
            for dict in agree_dicts:
                # include teacher switch
                if not bits in dict and bits != 'teacher_t':
                #if not bits in dict and 'teacher' not in bits:
                    dict[bits] = []

            epoch = int(row['epoch'])
            if epoch not in epochs:
                epochs.append(epoch)

            # include teacher switch
            #if 'teacher' not in bits:
            if 'teacher_t' not in bits:
                top_acc_results_train[bits].append(float(row['top_acc_train']))
                top_5_acc_results_train[bits].append(float(row['top5_acc_train']))
                top_acc_results_test[bits].append(float(row['top_acc_test']))
                top_5_acc_results_test[bits].append(float(row['top5_acc_test']))
            #if bits != 'teacher' and row['top_agreement_train'] is not None:
            #    agree_top_train[bits].append(float(row['top_agreement_train']))
            #    agree_kl_train[bits].append(float(row['kl_agreement_train']))
            #    agree_top_test[bits].append(float(row['top_agreement_test']))
            #    agree_kl_test[bits].append(float(row['kl_agreement_test']))

    plot_single_result_dict(top_acc_results_train, "Top train accuracy", epochs, plot_path)
    plot_single_result_dict(top_acc_results_test, "Top test accuracy", epochs, plot_path)
    plot_single_result_dict(top_5_acc_results_train, "Top 5 train accuracy", epochs, plot_path)
    plot_single_result_dict(top_5_acc_results_test, "Top 5 test accuracy", epochs, plot_path)

    #plot_single_result_dict(top_acc_results_train, "Top train accuracy", epochs, plot_path)
    #plot_single_result_dict_for_wd(top_acc_results_test, "0.0001", epochs, plot_path)
    #plot_single_result_dict(top_5_acc_results_train, "Top 5 train accuracy", epochs, plot_path)
    #plot_single_result_dict(top_5_acc_results_test, "Top 5 test accuracy", epochs, plot_path)

    #if agree_top_train != {}:
        #plot_agreement(agree_top_train, agree_top_test, "Top 1 agreement", epochs, plot_path)
        #plot_agreement(agree_kl_train, agree_kl_test, "KL agreement", epochs, plot_path)


def plot_single_result_dict(dict_res, name, epochs, plot_path):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(name)
    best_text = ''
    for bits in dict_res.keys():
        values = dict_res[bits]
        ax.scatter(epochs, values, marker=' ')
        ax.plot(epochs, values, label=bits)

        top = max(values)
        epoch = values.index(top)
        best_text = best_text + f'{bits} Bits: {top} at epoch {epoch} \n'

    print(best_text)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.legend(loc="lower right")
    plt.savefig(f'{plot_path}{name}.pdf')

    with open(f'{plot_path}{name}_top_values.txt', 'w') as out_file:
        out_file.write(best_text)


def plot_single_result_dict_for_wd(dict_res, name, epochs, plot_path):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(f'Test Accuracy using wd={name}')
    best_text = ''
    for bits in dict_res.keys():
        values = dict_res[bits]
        ax.scatter(epochs, values, marker=' ')
        ax.plot(epochs, values, label=bits)

        top = max(values)
        epoch = values.index(top)
        best_text = best_text + f'{bits} Bits: {top} at epoch {epoch} \n'

    print(best_text)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.legend(loc="lower right")
    plt.savefig(f'{plot_path}{name}.pdf')
    fig.tight_layout()

    with open(f'{plot_path}{name}_top_values.txt', 'w') as out_file:
        out_file.write(best_text)



def plot_agreement(train_dict, test_dict, name, epochs, plot_path):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(name)
    best_text = ''
    for bits in train_dict.keys():
        # train
        values_train = train_dict[bits]
        ax.scatter(epochs, values_train, marker=' ')
        ax.plot(epochs, values_train, label=f'{bits}_train')
        last = values_train[-1]
        best_text = best_text + f'Train {bits} Bits: {last}  \n'

        #test
        values_test = test_dict[bits]
        ax.scatter(epochs, values_test, marker=' ')
        ax.plot(epochs, values_test, label=f'{bits}_test')
        last = values_test[-1]
        best_text = best_text + f'Test {bits} Bits: {last}  \n'

    print(best_text)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.legend(loc="lower right")
    plt.savefig(f'{plot_path}{name}')

    with open(f'{plot_path}{name}_end_values.txt', 'w') as out_file:
        out_file.write(best_text)


def create_dir_for_csv_plots(csv):
    split = csv.rsplit('/', 1)
    csv_name = split[1]
    csv_name = csv_name.replace('.csv', '')
    path = split[0]
    path += f"/plots_{csv_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to csv file")
    parser.add_argument("-r", "--result", help="path to result dir", default = "")
    args = parser.parse_args()

    csv_path = args.path
    result_path = args.result
    if result_path == "":
        result_path = create_dir_for_csv_plots(csv_path)

    plot_csv_at_path(csv_path, result_path)
