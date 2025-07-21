from tqdm import tqdm
import time

EPOCHS = 50
STEPS_PER_EPOCH = 100

for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    inner = tqdm(range(STEPS_PER_EPOCH), desc="Step", leave=False)
    for step in inner:
        loss = 0.01 * step  # pretend loss
        acc = 0.1 * epoch   # pretend acc
        # tqdm.write(f"Epoch {epoch+1}, Step {step+1}, Loss {loss:.4f}, Acc {acc:.2f}")
        inner.set_postfix(epoch=epoch+1, step=step+1, loss=loss, acc=acc)
        time.sleep(0.1)  # simulate work
    time.sleep(0.1)  # simulate work