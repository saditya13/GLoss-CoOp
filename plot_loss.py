import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def parse_log_file(log_path):
    epochs = []
    losses = []
    accs = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match lines like "epoch [1/50] batch [1/100] loss: 0.123 acc: 45.6"
            match = re.search(r'epoch \[(\d+)/\d+\] batch \[\d+/\d+\] loss: ([\d.]+) acc: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))
                epochs.append(epoch)
                losses.append(loss)
                accs.append(acc)
    
    return epochs, losses, accs

def parse_tensorboard(tb_path):
    ea = EventAccumulator(tb_path)
    ea.Reload()
    
    epochs = []
    losses = []
    accs = []
    
    # Get scalar tags
    scalar_tags = ea.Tags()['scalars']
    print("Available tags:", scalar_tags)
    
    if 'train/loss' in scalar_tags:
        loss_events = ea.Scalars('train/loss')
        for event in loss_events:
            epochs.append(event.step)
            losses.append(event.value)
    
    if 'train/acc' in scalar_tags:
        acc_events = ea.Scalars('train/acc')
        accs = [event.value for event in acc_events]
    
    return epochs, losses, accs

def plot_loss_curve(log_path=None, tb_path=None, output_path=None):
    if tb_path:
        epochs, losses, accs = parse_tensorboard(tb_path)
    elif log_path:
        epochs, losses, accs = parse_log_file(log_path)
    else:
        print("No log or tensorboard path provided.")
        return
    
    if not epochs:
        print("No training data found.")
        return
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    if accs:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accs, label='Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy')
        plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage for gloss
    tb_path = "/home/aditya/CoOp/output/fgvc_aircraft/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1/gloss/tensorboard/events.out.tfevents.1776498250.a10080gb.2367093.0"
    plot_loss_curve(tb_path=tb_path, output_path="loss_curve_gloss.png")