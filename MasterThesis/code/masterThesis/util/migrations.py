import copy

import re

from binary_models.meliusnet import meliusnet_spec


def remove_arg(original_arg):
    # Ignores the second part of the argument. Only works under the bold assumption that each removable is only once in
    # the list of arguments
    def execute_remove_arg(arg_list, previous_msgs=[]):
        full_arg = next((arg_tuple for arg_tuple in arg_list if arg_tuple[0] == original_arg), None)
        arg_list.remove(full_arg)

        # Copy is necessary otherwise the fixture will keep the state of previous message lists and duplicate messages
        msgs = copy.deepcopy(previous_msgs)
        msgs.append("Remove argument \"" + " ".join(full_arg) + "\"")
        return arg_list, msgs

    return execute_remove_arg


def replace_arg(original_arg, replacement):
    def execute_replace_arg(arg_list, previous_msgs=[]):
        idx = arg_list.index(original_arg)
        arg_list[idx] = replacement

        # Copy is necessary otherwise the fixture will keep the state of previous message lists and duplicate messages
        msgs = copy.deepcopy(previous_msgs)
        msgs.append("Replace argument \"" + " ".join(original_arg) + "\" with \"" + " ".join(replacement) + "\"")
        return arg_list, msgs

    return execute_replace_arg


def chain(*args):
    def execute_chain(arg_list):
        msg = []
        for func in args:
            arg_list, msg = func(arg_list, msg)
        return arg_list, msg

    return execute_chain


# Works for melius- and naivenet might need some changes for other networks
def change_model_to(new_model_name, orig_arg):
    def execute_change_model(arg_list):
        old_model_name = next(arg_tuple[1] for arg_tuple in arg_list if arg_tuple[0] == "--model")
        block_config = next(arg_tuple[1] for arg_tuple in arg_list if arg_tuple[0] == "--block-config")
        model_num = str(sum([int(block) for block in block_config.split(",")]) + 5)

        # Modify model name if downsampling structure is given
        ds_arg = next((arg_tuple for arg_tuple in arg_list if arg_tuple[0] == "--downsample-structure"), None)
        if ds_arg:
            block_config_list = [int(block) for block in block_config.split(",")]
            fp_conv_match = re.match(".*(fp_conv:[0-9]+).*", ds_arg[1])
            assert fp_conv_match, "Extracting modified model name not possible because fp_conv not in arguments"

            # TODO: handle StopIteration
            idx = next(i for i, value in enumerate(list(meliusnet_spec.values()))
                       if value[0] == block_config_list and fp_conv_match[1] in value[2])

            net_suffix = list(meliusnet_spec.keys())[idx]

            # This is necessary because chars are concatenated with"_" (meliusnet_a) but numbers aren't (meliusnet23)
            if net_suffix.isalpha():
                extended_model_name = new_model_name + "_" + net_suffix
            else:
                extended_model_name = new_model_name + net_suffix
        else:
            extended_model_name = new_model_name + model_num

        replacement_rules = []
        # Edge case: some old block configs don't have a designated model
        if model_num == "31" and new_model_name == "meliusnet":
            extended_model_name = new_model_name + "_flex"
            replacement_rules.append(
                replace_arg(("--model", old_model_name),
                            ("--model", extended_model_name)),
            )
        else:
            replacement_rules.extend([
                replace_arg(("--model", old_model_name),
                            ("--model", extended_model_name)),
                remove_arg("--block-config"),
                remove_arg("--growth-rate"),
                remove_arg("--init-features"),
                remove_arg("--reduction")
            ])
        if orig_arg != "":
            replacement_rules.append(remove_arg(orig_arg))

        replacement_chain = chain(*replacement_rules)
        return replacement_chain(arg_list)

    return execute_change_model
