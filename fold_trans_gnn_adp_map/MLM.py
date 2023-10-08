import torch
import numpy as np


class MLMBase(object):
    """ Base class for Masking Language Model

    Attributes:
        seq_len: src sequence length
        sensors: number of sensors
        masking_ratio: the percent to mask, float out of 1
        save_tensor: if not None, save to a path
    """
    def __init__(self, seq_len, sensors, masking_ratio, save_tensor):
        self.seq_len = seq_len
        self.sensors = sensors
        self.masking_ratio = masking_ratio
        self.save_tensor = save_tensor

    def generate_mask(self):
        mask = torch.zeros(self.seq_len, self.sensors).bool()

        if (self.save_tensor is not None):
            self.save_tensor(mask, 'mask.pt')

        return mask

    def generate_gaussian_noise(self, all_random):
        torch.manual_seed(17)
        if all_random:
            noise = torch.randn(self.seq_len, self.sensors)
        else:
            noise = torch.randn(self.seq_len, 1)
            noise = noise.repeat(1, self.sensors)

        if (self.save_tensor is not None):
            self.save_tensor(noise, 'noise.pt')

        return noise
    
    def apply_mask(self, input_tensor, all_random=True):
        mask = self.generate_mask()
        noise = self.generate_gaussian_noise(all_random)
        input_tensor[mask] = noise[mask]
        return input_tensor

# dummy = torch.ones(5, 10).fill_(-3.)
# tmp = MLMBase(5, 10, 0.8, None)
# out = tmp.apply_mask(dummy)


class NaiveMLM(MLMBase):
    """ masking by selecting certain sensors, replace the entire seq_len

    Attributes:
        gamble: if gamble, 20% of masking_ratio would have p=0.8 bernoulli to gamble to be masked
    """
    def __init__(self, seq_len, sensors, masking_ratio, save_tensor, gamble):
        super().__init__(seq_len, sensors, masking_ratio, save_tensor)
        self.gamble = gamble
        
    def generate_mask(self):
        torch.manual_seed(17)
        rand = torch.rand(self.sensors)
        if self.gamble:
            # define sure mask
            mask_arr = rand < self.masking_ratio * 0.8 
            selection = torch.flatten(mask_arr.nonzero()).tolist()
    
            # define gamble mask
            gamble_arr = (rand < self.masking_ratio) & (rand >= self.masking_ratio * 0.8)
            selection_gamble = torch.flatten(gamble_arr.nonzero()) 

            # gamble with bern(0.8) # cannot ensure reproducibility
            gen = torch.ones(selection_gamble.shape).fill_(0.8)
            gen = torch.bernoulli(gen).bool()
            selection_gamble = selection_gamble[gen].tolist()

            # aggregate masked indices
            selection = selection + selection_gamble

            # generate mask
            mask_arr = torch.zeros(mask_arr.shape)
            mask_arr[selection] = 1.
            mask_arr = mask_arr.bool() 

        else:
            mask_arr = rand < self.masking_ratio
        
        mask = mask_arr.unsqueeze(0).repeat(self.seq_len, 1)

        if (self.save_tensor is not None):
            self.save_tensor(mask, 'mask.pt')

        return mask

# X = torch.rand(12, 207)
# mlm = NaiveMLM(12, 207, 0.15, None, True)
# mask = mlm.generate_mask()
# noise = mlm.generate_gaussian_noise(all_random=False)
# out = mlm.apply_mask(X, all_random=False)



class GeomMLM(MLMBase):
    """ masking by geometric/uniform distribution, allow both concurrent and separate mask across sensors

    Attributes:
        lm: average length of masking subsequences (streaks of 1s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_sensors: iterable of indices corresponding to sensor to be excluded from masking (i.e. to remain all 1s)
    """
    def __init__(self, seq_len, sensors, masking_ratio, save_tensor, lm, mode, distribution, exclude_sensors):
        super().__init__(seq_len, sensors, masking_ratio, save_tensor)
        self.lm = lm
        self.mode = mode
        self.distribution = distribution
        self.exclude_sensors = exclude_sensors

    def geom_noise_mask_single(self):
        """
        Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 1s a `masking_ratio`
        proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
        
        Returns:
            (seq_len,) boolean numpy array intended to mask ('drop') with 1s a sequence of length seq_len
        """
        keep_mask = np.zeros(self.seq_len, dtype=bool)
        p_m = 1 / self.lm  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = p_m * self.masking_ratio / (1 - self.masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_u, p_m]

        # Start in state 0 with masking_ratio probability
        state = int(np.random.rand() < self.masking_ratio)  # state 1 means masking, 0 means not masking
        for i in range(self.seq_len):
            keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
            if np.random.rand() < p[state]:
                state = 1 - state

        return keep_mask

    def generate_mask(self):
        np.random.seed(17)
        """
        Creates a random boolean mask of the same shape as X, with 1s at places where a feature should be masked.
        Returns:
            boolean numpy array with the shape [seq_len, sensors], with 1s at places where a feature should be masked
        """
        if (self.exclude_sensors is not None):
            self.exclude_sensors = set(self.exclude_sensors)

        if self.distribution == 'geometric':  # stateful (Markov chain)
            if self.mode == 'separate':  # each variable (feature) is independent
                mask = np.ones((self.seq_len, self.sensors), dtype=bool)
                for m in range(self.sensors):  # feature dimension
                    if self.exclude_sensors is None or m not in self.exclude_sensors:
                        mask[:, m] = self.geom_noise_mask_single()  # time dimension
            else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
                mask = np.tile(np.expand_dims(self.geom_noise_mask_single(), 1), self.sensors)
        else:  # each position is independent Bernoulli with p = 1 - masking_ratio
            if self.mode == 'separate':
                mask = np.random.choice(np.array([True, False]), size=(self.seq_len, self.sensors), replace=True,
                                        p=(self.masking_ratio, 1 - self.masking_ratio))
            else:
                mask = np.tile(np.random.choice(np.array([True, False]), size=(self.seq_len, 1), replace=True,
                                                p=(self.masking_ratio, 1 - self.masking_ratio)), self.sensors)
        
        if (self.save_tensor is not None):
            self.save_tensor(torch.from_numpy(mask), 'mask.pt')

        return torch.from_numpy(mask)


# mlm_rand = GeomMLM(seq_len=12, sensors=207, masking_ratio=0.3, save_tensor=True, 
#                     lm=3, mode='concur', distribution='rand', exclude_sensors=None) 
# mask = mlm_rand.generate_mask()
# noise = mlm_rand.generate_gaussian_noise(False)
# out = mlm_rand.apply_mask(X, False)