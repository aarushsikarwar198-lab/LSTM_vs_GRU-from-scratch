import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.nn.LSTM(8, 16), torch.nn.GRU(8, 16))
print(torch.cuda.get_device_name(0))