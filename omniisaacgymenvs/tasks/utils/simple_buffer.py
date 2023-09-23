import torch

class SimpleVecBuffer():
    def __init__(self, max_size, thread_num, data_dim, device, pad='zero'):
        # data_dim: tuple, (thread_num, data_dim), e.g. = (num_envs, num_obs)
        self.max_size = max_size
        assert self.max_size >= 1
        self.thread_num = thread_num
        self.dim = data_dim
        self.pad = pad
        self.size = torch.zeros(self.thread_num, device=device, dtype=torch.long)
        self.device = device
        # buffer.shape: (buffer_n, thread_num, data_col_num), zero padded
        self.buf = self.initialize(device)


    def initialize(self, device):
        return torch.zeros((self.thread_num, self.max_size, self.dim), device=device) # buffer


    def get(self):
        # squash dimension to (thread_num, max_size * data_dim)
        return self.buf.reshape((self.thread_num, self.max_size * self.dim))


    def reset(self, reset_idx):
        num_reset = len(reset_idx)
        self.size[reset_idx] = 0
        self.buf[reset_idx] = torch.zeros((num_reset, self.max_size, self.dim), device = self.device)


    def push(self, x):
        assert len(x.shape) == 2, "len(x.shape) == 2"
        assert x.shape[0] == self.thread_num, "x.shape[0] == self.thread_num"
        assert x.shape[1] == self.dim, "x.shape[1] == self.dim"
        # fill empty buf with first entry
        if self.pad == 'same':
            fill_thread = torch.where(self.size == 0, 1, 0)
            fill_indeces = fill_thread.nonzero(as_tuple=False)
            self.buf[fill_indeces, :, :] = x[fill_indeces].unsqueeze(1)
        self.buf = self._push(x)
        self.size = torch.clamp(self.size + 1, max=self.max_size)

 
    def _push(self, x):
        return torch.cat((self.buf[:, 1:, :], x.reshape(self.thread_num, 1, self.dim)), dim=1)


class SimpleExtremeValueChecker():
    def __init__(self, num_envs, data_dim, device):
        # NOTE: this does not calculate the actual mean and var of each reading
        self.num_envs = num_envs
        self.data_dim = data_dim
        self.device = device
        self.count = torch.zeros(num_envs, device=device, dtype=torch.int)
        self._means = torch.zeros(num_envs, data_dim, device=device)
        self._vars = torch.zeros(num_envs, data_dim, device=device)
        self.last_data_buf = None        # dict{str: tensor[num_env, dim]}
        self.eps = 1e-4

        self._initialized = False

    def check(self, readings, threshold=5e2, min_count=3):
        flatten = [v.reshape(self.num_envs, -1) for v in readings.values()]
        data_tensor = torch.cat(flatten, dim=-1) # num_env, dim
        std = torch.sqrt(self._vars)
        assert not torch.any(torch.isnan(std))

        # print(data_tensor.mean())

        std_delta = torch.abs(data_tensor - self._means) / (std + self.eps)
        # std_delta = torch.abs(data_tensor - self._means)  #self._means # deviate from mean

        max_std_delta = torch.max(std_delta, dim=-1)[0]    # num_env
        # print('delta', max_std_delta)
        # print(torch.min(std, dim=-1)[0])
        # only check if count > 1 (valid std and mean value)
        mask = self.count > min_count

        # print('mask num: ', (mask & (max_std_delta > threshold)).sum())
        # print('max_std: ', max_std_delta.max())
        return (mask & (max_std_delta > threshold)).nonzero(as_tuple=False).squeeze(-1)
    

    def update(self, readings):
        # assume readings is valid
        flatten = [v.reshape(self.num_envs, -1) for v in readings.values()]
        data_tensor = torch.cat(flatten, dim=-1) # num_env, dim
        # update mean and vars
        self._means[:], self._vars[:] = self._get_update_means_and_vars(data_tensor)
        self.count += 1
    
    def save_last(self, data):
        if not self._initialized:
            self.last_data_buf = data
            self._initialized = True
        else:
            for k, v in data.items():
                self.last_data_buf[k][:] = v

    def reset(self, reset_id):
        # print(len(reset_id))
        self.count[reset_id] = 0
        # for some reason broadcasting doesnt work with pytorch_deterministic
        self._means[reset_id] = torch.zeros(len(reset_id), self.data_dim, device=self.device)
        self._vars[reset_id] = torch.zeros(len(reset_id), self.data_dim, device=self.device)
        # print('in_reset: ', reset_id)
        # print(self._means.shape)
        # print(self._means[reset_id])
        # print(self._vars[reset_id])
        # self.count.index_fill_(0, reset_id, 0)
        # self._means.index_fill_(0, reset_id, 0)
        # self._vars.index_fill_(0, reset_id, 0)
    
    def load_last(self):
        return self.last_data_buf
    
    def _get_update_means_and_vars(self, data):
        delta = data - self._means
        new_count = self.count + 1
        mu = self._means + delta / new_count[:,None]

        E = self.count[:,None] * (self._vars + delta**2)
        var = E/new_count[:,None]
        return mu, var



    

# class ExtremeReadingsChecker():
#     def __init__(self, num_readings, num_envs, device):
#         # NOTE: this does not calculate the actual mean and var of each reading
#         self.num_readings = num_readings
#         self.count = torch.zeros(num_envs)
#         self._means = torch.zeros(num_readings, num_envs, device=device)
#         self._vars = torch.zeros(num_readings, num_envs, device=device)
#         self.last_data_buf = None        # dict{str: tensor[num_env, dim]}
#         self.eps = 1e-5
    
#     def check(self, readings):
#         # suffice to check max and min values of each dim
#         if self.count == 0:
#             self.update(readings)
#             return []
        
#         for idx, k in enumerate(readings):
#             (readings[k] - self.means[idx]) / (self.)
        
    
#     def update(self, readings):
#         # update means and vars of *each reading*, not actual may not make sense to use
#         for idx, k in enumerate(readings):
#             new_mean = torch.mean(readings[k], dim=-1)  # num_env
#             new_var = torch.variance(readings[k], dim=-1) # num_env
#             self.means[idx] = self._get_update_means(oldnew_mean, count)
#             self.vars[idx] = self._get_update_vars(var, count)

#         # update last_values
#         self.last_data_buf = readings
#         self.count += 1

#     def _get_update_means(self, new_mean, count):
#         # get var for each env
#         return

#     def _get_update_vars(self, new_var, count):
#         return
