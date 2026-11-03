import numpy as np
import torch

from dataloader import PINNDataset, load_supervised_dataset, load_unsupervised_dataset, make_loader
from utils import autodiff, read_result_data, read_points_data

def test_autodiff():
    x = torch.tensor([[1.0, 3.5, 1.5], [2.0, 4.0, 2.0]], requires_grad=True)
    # y = x**2 + 3*x + 2
    W = torch.empty((3, 5)).uniform_(0.5, 1.5)
    b = torch.empty((5, )).uniform_(0.5, 1.5)
    y = x @ W + b
    y1 = y[:, 0]
    y2 = y[:, 1]
    y3 = y[:, 2]
    dy_dx = autodiff(y1 + y2 + y3, x)
    print(dy_dx)
    jac = torch.autograd.functional.jacobian(lambda inp: (inp @ W + b), x)
    print(jac)
    assert torch.allclose(dy_dx, jac), f"Expected {jac}, but got {dy_dx}"

def test_read_result_data():
    data = read_result_data("../算例/program_25_09_27_2", "1e-05", "p")
    print(data[:10])
    data2 = read_result_data("../算例/program_25_09_27_2", "1e-05", "U")
    print(data2[:10])
    data3 = read_result_data("../算例/program_25_09_27_2", "1e-05", "T")
    print(data3[:10])

def test_read_points_data():
    points = read_points_data("../算例/program_25_09_30/0", "C")
    print(points["internal"].shape)
    print(points["internal"][:10])
    print(points["inlet"].shape)
    print(points["inlet"][:10])
    print(points["outlet"].shape)
    print(points["outlet"][:10])
    print(points["bottom"].shape)
    print(points["bottom"][:10])    
    print(points["top"].shape)
    print(points["top"][:10])
    print(points["obstacle"].shape)
    print(points["obstacle"][:10])


def test_pinn_dataset_mixed():
    train = load_unsupervised_dataset(
        points_path="../算例/program_25_09_30/0",
        points_filename="C"
    )
    print(train.inputs[:10])
    print(train.inputs.shape)
    print(train.condition_types.count("internal"))
    print(train.condition_types.count("inlet"))
    print(train.condition_types.count("outlet"))
    print(train.condition_types.count("bottom"))
    print(train.condition_types.count("top"))
    print(train.condition_types.count("obstacle"))
    val = load_supervised_dataset(
        result_path="../算例/program_25_09_30",
        result_time="1e-05",
        result_filename="U",
        points_path="../算例/program_25_09_30/0",
        points_filename="C"
    )
    print(val.inputs[:10])
    print(val.targets[:10])

    train_loader = make_loader(train, shuffle_flag=True, batch_size=100, drop_last=True)
    for i, data in enumerate(train_loader):
        inputs, conditions = data[0], data[1]
        print(inputs)
        print(conditions)
        print(conditions.count('internal'))
        print(conditions.count('inlet'))
        print(conditions.count('outlet'))
        print(conditions.count('bottom'))
        print(conditions.count('top'))
        print(conditions.count('obstacle'))
        idx = np.argwhere(np.array(conditions) == 'inlet')[0]
        print(inputs[idx])
        break


if __name__ == "__main__":
    test_autodiff()