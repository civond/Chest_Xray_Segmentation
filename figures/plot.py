import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("training_metrics.csv")
epochs = range(len(df['training_loss']))


fig, axes = plt.subplots(2, 1, figsize=(4, 6))  # Big square figure

# Flatten axes array for easy iteration
axes = axes.flatten()

# --- Training Loss ---
axes[0].plot(epochs, 
             df['training_loss'], 
             color='b', 
             label='train_loss')
axes[0].plot(epochs, 
             df['validation_loss'], 
             color='r', 
             label='valid_loss')
axes[0].set_title("Training Loss")
axes[0].set_xlim(0, np.max(epochs))
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Avg. Loss")
axes[0].grid(True)
axes[0].legend()
#axes[0].set_aspect('equal')  # Makes subplot square

axes[1].plot(epochs, 
             df['training_dice'], 
             color='b', 
             linestyle="--", 
             label='train_dice')
axes[1].plot(epochs, 
             df['validation_dice'], 
             color='r',
             linestyle="--", 
             label='valid_dice')
axes[1].set_title("Dice Score")
axes[1].set_xlim(0, np.max(epochs))
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Avg. Dice Score")
axes[1].grid(True)
axes[1].legend()
#axes[1].set_aspect('equal')  # Makes subplot square
plt.tight_layout()
plt.show()

"""plt.figure(1)
plt.subplot(2,1,1)
plt.plot(epochs, 
         df['training_loss'], 
         color='b', 
         label='train_loss')
plt.plot(epochs, 
         df['validation_loss'], 
         color='r', 
         label='valid_loss')
plt.show()"""


"""plt.figure(1, figsize=(6,6))
plt.subplot(2,2,1)
plt.title("Train Loss")
plt.ylabel('Avg. Loss')
plt.xlabel('Epoch')
plt.plot(df['training_loss'], color='b')
plt.xlim(0,9)
plt.ylim(0,np.max(df['training_loss'])+.02)

plt.subplot(2,2,2)
plt.title("Train Dice Score")
plt.ylabel('Avg. Dice Score')
plt.xlabel('Epoch')
plt.plot(df['training_dice'], color='b', linestyle="--")
plt.xlim(0,9)

plt.subplot(2,2,3)
plt.title("Validation Loss")
plt.ylabel('Avg. Loss')
plt.xlabel('Epoch')
plt.plot(df['validation_loss'], color='r')
plt.xlim(0,9)
plt.ylim(0,np.max(df['validation_loss'])+.02)

plt.subplot(2,2,4)
plt.title("Validation Dice Score")
plt.ylabel('Avg. Dice Score')
plt.xlabel('Epoch')
plt.plot(df['validation_dice'], color='r', linestyle="--")
plt.xlim(0,9)

plt.tight_layout()
plt.show()"""