import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import learn2learn as l2l
from dataloader.dataloader import DataLoader
from Model.CompareModels.BaselineModel import PINN
from utils.util import compute_loss, evaluate_model

class MAML_PINN:
    def __init__(self, model, lr, adaptation_steps):
        self.maml = l2l.algorithms.MAML(model, lr=lr)
        self.adaptation_steps = adaptation_steps

    def adapt(self, task_data):
        learner = self.maml.clone()
        for step in range(self.adaptation_steps):
            data, target = task_data
            data, target = data.to(device), target.to(device)
            loss = compute_loss(learner, data, target)
            learner.adapt(loss)
        return learner

    def evaluate(self, test_tasks):
        meta_test_loss = 0.0
        for task_data in test_tasks:
            learner = self.adapt(task_data)
            data, target = task_data
            data, target = data.to(device), target.to(device)
            loss = compute_loss(learner, data, target)
            meta_test_loss += loss
        print(f'Meta Test Loss: {meta_test_loss.item()}')

class Trainer:
    def __init__(self, model, train_tasks, meta_lr, adaptation_lr, adaptation_steps, meta_epochs):
        self.model = model
        self.train_tasks = train_tasks
        self.meta_lr = meta_lr
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.meta_epochs = meta_epochs

    def train(self):
        maml_pinn = MAML_PINN(self.model, lr=self.adaptation_lr, adaptation_steps=self.adaptation_steps)
        meta_optimizer = optim.Adam(maml_pinn.maml.parameters(), lr=self.meta_lr)

        for epoch in range(self.meta_epochs):
            meta_optimizer.zero_grad()
            meta_train_loss = 0.0
            for task_data in self.train_tasks:
                learner = maml_pinn.adapt(task_data)
                data, target = task_data
                data, target = data.to(device), target.to(device)
                loss = compute_loss(learner, data, target)
                meta_train_loss += loss
                loss.backward()
            meta_optimizer.step()
            print(f'Epoch {epoch+1}, Meta Train Loss: {meta_train_loss.item()}')
        return maml_pinn

def main(args):
    # Set device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_loader = DataLoader(args.data_path)
    train_tasks = data_loader.get_train_tasks()
    test_tasks = data_loader.get_test_tasks()

    # Initialize model
    model = PINN().to(device)

    # Initialize and train
    trainer = Trainer(
        model=model,
        train_tasks=train_tasks,
        meta_lr=args.meta_lr,
        adaptation_lr=args.adaptation_lr,
        adaptation_steps=args.adaptation_steps,
        meta_epochs=args.meta_epochs
    )
    maml_pinn = trainer.train()

    # Evaluate
    maml_pinn.evaluate(test_tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAML for PINN')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate')
    parser.add_argument('--adaptation_lr', type=float, default=0.01, help='Adaptation learning rate')
    parser.add_argument('--adaptation_steps', type=int, default=5, help='Adaptation steps')
    parser.add_argument('--meta_epochs', type=int, default=100, help='Meta epochs')
    
    args = parser.parse_args()
    main(args)
