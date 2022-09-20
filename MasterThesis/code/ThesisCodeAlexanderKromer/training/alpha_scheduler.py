class alpha_scheduler:
    def __init__(self, alpha_max, alpha_min, num_epochs):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.num_epochs = num_epochs

    #def current_alpha(self, current_epoch):
