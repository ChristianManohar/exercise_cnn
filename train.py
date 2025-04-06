import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from cnn import CNN
from im_dataset import get_loaders_and_datasets
import tqdm
import pathlib

labels = ["deadlift", "hammer_curl", "tricep_pushdown", "squat", "tricep_dips", "lat_pulldown", "push_up", "barbell_biceps_curl",
          "pec_fly", "incline_bench_press", "leg_extension", "shoulder_press", "t_bar_row", "decline_bench_press", "bench_press", "pull_up",
          "plank", "lateral_raise", "leg_raise", "hip_thrust", "RDLs", "russian_twists"]

model = CNN()

out_dir = pathlib.Path("exercise_ml/exp").resolve()
out_dir.mkdir(parents=True, exist_ok=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 32

tr_loader, va_loader, te_loader, tr_dataset, va_dataset, te_dataset = get_loaders_and_datasets(batch_size)

n_epochs = 10
val_steps = 1000
train_losses, val_losses = [], []
val_accs = []
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

            for a, (ims, labels) in enumerate(va_loader):

                model.eval()
                prediction = model(ims)
                loss = loss_fn(prediction, labels)
                val_loss += len(ims) * float(loss)

                count_correct += torch.count_nonzero(prediction.argmax(1) == labels)

            np.save(out_dir / "train_losses.npy", train_losses)
            
            val_loss = val_loss / len(va_dataset)
            val_losses.append(val_loss)

            np.save(out_dir / "val_losses.npy", val_losses)

            val_acc = float(count_correct / len(va_dataset))
            val_accs.append(val_acc)

            pbar.set_postfix(
                loss=float(loss), val_loss=val_loss, val_acc=val_acc
            )

            if val_loss < min_val_loss:
                torch.save(model.state_dict(), out_dir / f"best_model.pt")
                min_val_loss = val_loss
    
    w = 100
    train_losses_ma = np.convolve(train_losses, np.ones(w), "valid") / w

    plt.plot(np.arange(len(train_losses)), train_losses, color="C0", alpha=0.2)
    plt.plot(
        np.arange(len(train_losses_ma)) + w / 2, train_losses_ma,
        label="Training", color="C1"
    )
    plt.plot(
        np.arange(len(val_losses)) * val_steps, val_losses,
        label="Validation", color="C1"
    )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("losses.png")
    plt.show()

    """for a, (X, y) in enumerate(va_loader):
                model.eval()
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                count_correct += (y_pred.argmax(dim=1) == y).sum().item()
"""