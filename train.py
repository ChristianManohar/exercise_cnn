import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from cnn import CNN
from im_dataset import get_loaders
import tqdm

labels = ["deadlift", "hammer_curl", "tricep_pushdown", "squat", "tricep_dips", "lat_pulldown", "push_up", "barbell_biceps_curl",
          "pec_fly", "incline_bench_press", "leg_extension", "shoulder_press", "t_bar_row", "decline_bench_press", "bench_press", "pull_up",
          "plank", "lateral_raise", "leg_raise", "hip_thrust", "RDLs", "russian_twists"]

model = CNN()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 32

tr_loader, va_loader, te_loader = get_loaders(batch_size)

n_epochs = 10
val_steps = 1000
train_losses, val_losses = [], []
val_acc = []
val_loss, val_acc = np.nan, np.nan
min_val_loss = float("inf")
step = 0

for epoch in range(n_epochs):
    pbar = tqdm.tqdm(tr_loader, desc=f"Epoch {epoch}")
    
    for i, (X, y) in enumerate(pbar):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_losses.append(float(loss))
        loss.backward()
        optimizer.step()
        pbar.set_postfix(
            loss=float(loss), val_loss=val_loss, val_acc=val_acc, refresh=False
        )
        step += 1

        if step % val_steps == 0:
            val_loss = 0
            count_correct = 0

            for a, (X, y) in enumerate(va_loader):
                model.eval()
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                count_correct += (y_pred.argmax(dim=1) == y).sum().item()
