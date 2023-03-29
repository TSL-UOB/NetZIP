import torch
import torch.nn as nn

class QuantizedNN(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedNN, self).__init__()
        # QuantStub converts tensors from floating point to quantized. This is only used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point. This is only used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x
