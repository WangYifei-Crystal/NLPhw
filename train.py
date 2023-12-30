import numpy as np
import model
import wandb
from kaggle_secrets import UserSecretsClient
from model import CFG, train_loop

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB")
wandb.login(key=secret_value_0)
wandb.init(project='feedback-prize-effectiveness', name='mlm-aurora-large')

device = CFG.device
model.to(device)
history = []
best_loss = np.inf
prev_loss = np.inf
model.gradient_checkpointing_enable()
print(f"Gradient Checkpointing: {model.is_gradient_checkpointing}")

for epoch in range(CFG.epochs):
    loss = train_loop(model, device)
    history.append(loss)
    print(f"Loss: {loss}")
    if loss < best_loss:
        print("New Best Loss {:.4f} -> {:.4f}, Saving Model".format(prev_loss, loss))
        # torch.save(model.state_dict(), "./deberta_mlm.pt")
        model.save_pretrained('./')
        best_loss = loss
    prev_loss = loss