import BitDiffusion
import torch


def test_back_to_back():
    bits=8
    x=torch.rand([20,40,100])
    x=torch.rand([10, 4, 12, 14])
    y=x.clone()

    y=BitDiffusion.decimal_to_qubits(y,bits=bits)
    y=BitDiffusion.qubit_to_decimal(y, bits=bits)

    assert (torch.abs(y-x)<=1/(bits-1)).all()