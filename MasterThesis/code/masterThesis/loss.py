from enum import Enum
from mxnet.gluon.loss import KLDivLoss
from mxnet import nd

"""
    Calculate loss for different teachers and knowledge distillation modes
"""

class LossCalculationModes(Enum):
    # Different methods of calculating overall loss
    # Weight all losses equally
    equal = 'equal'


class ModelLoss:
    def __init__(self, loss, bits):
        self.loss = loss
        self.bits = bits

class KnowledgeDistiTeacher(Enum):
    none = 'none'
    pretrained_external = 'pretrained_external'
    untrained_external = 'untrained_external'
    internal = 'internal'


class KnowledgeDistiMode(Enum):
    none = 'none'
    # knwoledge from 32 bit net in ensemble
    simple = 'simple'
    progressiv = 'progressive'


class LossCalculator:

    def __init__(self, loss_fn, num_nets, loss_mode_string, knowledge_distillation_mode='none', knowledge_distillation_teacher='none',  alpha=0.0, tt_knowledge_distillation_mode='none'):
        self.loss = loss_fn
        self.num_nets = num_nets
        self.loss_mode = LossCalculationModes(loss_mode_string)
        self.knowledge_distillation_mode = KnowledgeDistiMode(knowledge_distillation_mode)
        self.knowledge_distillation_teacher_mode = KnowledgeDistiTeacher(knowledge_distillation_teacher)
        self.kd_loss_fn = KLDivLoss(from_logits=False)
        self.alpha = alpha
        self.tt_knowledge_distillation_mode = KnowledgeDistiMode(tt_knowledge_distillation_mode)

    def calulate_ensemble_loss(self, outputs, y, teacher_output=None, teacher_ensemble=False, use_tt_mode=False):
        if use_tt_mode:
            kd_mode = self.tt_knowledge_distillation_mode
        else:
            kd_mode = self.knowledge_distillation_mode

        if teacher_output is not None:
            if type(teacher_output) is dict:
                for bits, output in teacher_output.items():
                    teacher_output[bits] = output.detach()
            else:
                teacher_output = teacher_output.detach()
        #(teacher_output, y, t_teacher_output, False)
        losses = []
        # default loss
        # TODO refactor default loss to be optional
        for bits, output in outputs.items():
            losses.append(ModelLoss(loss=self.loss(output, y), bits=bits))
        if kd_mode == KnowledgeDistiMode.simple:
            for model_loss in losses:
                output = next((output for (bits, output) in outputs.items() if bits == model_loss.bits), None)
                t_out = self.get_teacher_output(model_loss, teacher_output, teacher_ensemble)
                distillation_loss = self.calculate_distillation_loss(t_out, output)
                model_loss.loss = (self.alpha * model_loss.loss) + ((1-self.alpha) * distillation_loss)

        elif kd_mode == KnowledgeDistiMode.progressiv:
            outputs_sorted_by_bits = sorted(outputs.items(), reverse=True)

            # Calculate disti loss of highest bit width
            bits_highest, output_highest = outputs_sorted_by_bits[0]
            model_loss = next((model_loss for model_loss in losses if model_loss.bits == bits_highest), None)
            t_out = self.get_teacher_output(model_loss, teacher_output, teacher_ensemble)
            # enable progressive without teacher
            if t_out is not None:
                distillation_loss = self.calculate_distillation_loss(t_out, output_highest)
                model_loss.loss = (self.alpha * model_loss.loss) + ((1 - self.alpha) * distillation_loss)

            # Calculate for other bit widths
            for (bits_high, output_high), (bits_low, output_low) in zip(outputs_sorted_by_bits, outputs_sorted_by_bits[1:]):
                model_loss = next((model_loss for model_loss in losses if model_loss.bits == bits_low), None)
                distillation_loss = self.calculate_distillation_loss(output_high, output_low)
                model_loss.loss = (self.alpha * model_loss.loss) + ((1 - self.alpha) * distillation_loss)

        losses_arr = None
        for loss in losses:
            l = nd.expand_dims(loss.loss, axis=0)
            if losses_arr is None:
                losses_arr = l
            else:
                losses_arr = nd.concat(losses_arr, l, dim=0)
        final_loss = losses_arr

        return final_loss

    def calculate_equal_loss(self, losses):
        # TODO Copied from older version, Refactor this
        Ls = None
        for model_loss in losses:
            if Ls is None:
                Ls = model_loss.loss
            else:
                Ls = Ls + model_loss.loss
        return Ls / self.num_nets

    def calculate_distillation_loss(self, output_high_prec, output_low_prec):
        output_high = nd.softmax(output_high_prec)
        return self.kd_loss_fn(output_low_prec, output_high)

    def calculate_teacher_loss(self, output, y, teacher_ensemble):
        if teacher_ensemble:
            # TODO refactor this and for normal net
            losses_arr = None
            for bits, output in output.items():
                l = nd.expand_dims(self.loss(output, y), axis=0)
                if losses_arr is None:
                    losses_arr = l
                else:
                    losses_arr = nd.concat(losses_arr, l, dim=0)
            return losses_arr
        else:
            return self.loss(output, y)

    def get_teacher_output(self, model_loss, teacher_output, teacher_ensemble):
        if teacher_ensemble:
            # TODO hotfix when using a single model as ensemble teacher, refactor
            if len(teacher_output.keys()) == 1:
                t_out = list(teacher_output.values())[0]
            else:
                t_out = teacher_output[model_loss.bits]
            return t_out
        else:
            return teacher_output