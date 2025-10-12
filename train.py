import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import all_loss, initial_loss, boundary_loss, MSE_loss
import time
import os

class Trainer:
    def __init__(self, model, optimizer, scheduler=None, device='cpu', alpha=1, beta=1, gamma=1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(self.device)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.best_val_loss = float('inf')
        self.log_dir = None

    def train(self, epochs, train_loader, val_loader, log_dir, step_size = 100):
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        log_dir += f'/{current_time}'
        self.log_dir = log_dir
        writer = SummaryWriter(log_dir=log_dir)
        print('Start Training...')

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            running_loss = 0.0
            progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
            for i, data in progress:
                inputs, conditions = data[0], data[1]

                t = torch.empty(inputs.shape[0], 1, dtype=torch.float64).uniform_(0.0, 1e-5)
                t0 = torch.zeros(inputs.shape[0], 1, dtype=torch.float64)
                inputs0 = torch.hstack((inputs, t0))
                inputs0 = inputs0.to(self.device).requires_grad_(True)
                inputs = torch.hstack((inputs, t))
                inputs = inputs.to(self.device).requires_grad_(True)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                outputs0 = self.model(inputs0)

                # Compute losses
                loss_all = all_loss(inputs, outputs, conditions)
                loss_initial = initial_loss(inputs0, outputs0, conditions)
                loss_boundary = boundary_loss(inputs, outputs, conditions)
                # print(f'Losses: All={loss_all.item():.4e}, Initial={loss_initial.item():.4e}, Boundary={loss_boundary.item():.4e}')
                
                # Total loss
                total_loss = self.alpha * loss_all + self.beta * loss_initial + self.gamma * loss_boundary
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                running_loss += total_loss.item()

                if i % step_size == step_size - 1:
                    progress.set_postfix({'Batch Loss': running_loss / step_size})
                    running_loss = 0.0

            progress.close()

            if self.scheduler:
                self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for data in val_loader:
                    inputs, results = data[0].to(self.device), data[1].to(self.device)

                    outputs = self.model(inputs)

                    loss_all = MSE_loss(outputs, results)

                    val_loss += loss_all.item()

            self.model.train()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss / len(val_loader):.4f}')

            writer.add_scalar('Training Loss', epoch_loss / len(train_loader), epoch + 1)
            writer.add_scalar('Validation Loss', val_loss / len(val_loader), epoch + 1)

            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                    }, f'{log_dir}/model_epoch_{epoch + 1}.pth')
            if self.best_val_loss > val_loss / len(val_loader):
                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                    }, f'{log_dir}/best_model.pth')
                
                self.best_val_loss = min(self.best_val_loss, val_loss / len(val_loader))

        writer.flush()
        writer.close()


    def evaluate(self, test_loader, log_dir = None):
        print('Start Testing...')

        if log_dir:
            checkpoint = torch.load(f'{log_dir}/best_model.pth', weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded model from {log_dir}/best_model.pth')
        else:
            checkpoint = torch.load(f'{self.log_dir}/best_model.pth', weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print('Evaluating current best model...')

        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for data in test_loader:
                inputs, results = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(inputs)

                loss_all = MSE_loss(outputs, results)

                total_loss += loss_all.item()
        
        print(f'Test Loss: {total_loss / len(test_loader):.4f}')

    
    def predict(self, datasets, time, log_dir = None):
        print('Start Testing...')

        if log_dir:
            checkpoint = torch.load(f'{log_dir}/best_model.pth', weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded model from {log_dir}/best_model.pth')
        else:
            checkpoint = torch.load(f'{self.log_dir}/best_model.pth', weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print('Evaluating current best model...')
    
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for dataset in datasets:
                inputs = dataset.unsqueeze(0)
                t = torch.full((inputs.shape[0], 1), fill_value=float(time))
                inputs = torch.hstack((inputs, t))
                inputs = inputs.to(self.device).requires_grad_(True)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu())

        target_dir = log_dir or self.log_dir
        target_dir += f'/predictions_{time}'
        os.makedirs(target_dir, exist_ok=True)

        U, rho, p, T = [], [], [], []
        for pred in predictions:
            data = pred.numpy()
            if data.shape[1] < 5:
                raise ValueError('Prediction tensor列数不足5。')

            U.append(data[0][:2].tolist()+[0.0])
            rho.append(data[0][2:3].tolist())
            p.append(data[0][3:4].tolist())
            T.append(data[0][4:5].tolist())

        with open(f'{target_dir}/U.txt', 'w', encoding='utf-8') as f:
            for row in U:
                f.write('(' + ' '.join(f'{value:.6e}' for value in row) + ')\n')
        with open(f'{target_dir}/rho.txt', 'w', encoding='utf-8') as f:
            for row in rho:
                f.write(f'{row[0]:.6e}' + '\n')
        with open(f'{target_dir}/p.txt', 'w', encoding='utf-8') as f:
            for row in p:
                f.write(f'{row[0]:.6e}' + '\n')
        with open(f'{target_dir}/T.txt', 'w', encoding='utf-8') as f:
            for row in T:
                f.write(f'{row[0]:.6e}' + '\n')