import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train import Trainer
from dataloader import load_supervised_dataset, load_unsupervised_dataset, load_predict_dataset, make_loader
from FNN import FNN

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

def pinn_for_u():
    model = FNN(layers=[3, 32, 96, 64, 5], activation=nn.Tanh())
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    trainer = Trainer(model, optimizer, scheduler, device='cuda' if torch.cuda.is_available() else 'cpu')
    # trainer = Trainer(model, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = load_unsupervised_dataset(
        points_path="../算例/program_25_09_30/0",
        points_filename="C"
    )
    train_loader = make_loader(train_dataset, shuffle_flag=True, batch_size=100, drop_last=True)

    val_dataset = load_supervised_dataset(
        result_path="../算例/program_25_09_30",
        result_time="1e-05",
        points_path="../算例/program_25_09_30/0",
        points_filename="C"
    )
    val_loader = make_loader(val_dataset, shuffle_flag=False, batch_size=100, drop_last=False)

    trainer.train(epochs=200, train_loader=train_loader, val_loader=val_loader, log_dir='./logs', step_size=100)

    test_dataset = load_supervised_dataset(
        result_path="../算例/program_25_09_30", 
        result_time="1e-05",
        points_path="../算例/program_25_09_30/0",
        points_filename="C"
    )
    test_loader = make_loader(test_dataset, shuffle_flag=False, batch_size=100, drop_last=False)

    trainer.evaluate(test_loader=test_loader)    

    predict_dataset = load_predict_dataset(
        points_path="../算例/program_25_09_30/0",
        points_filename="C"
    )

    trainer.predict(datasets=predict_dataset, time=1e-05)

if __name__ == "__main__":
    pinn_for_u()