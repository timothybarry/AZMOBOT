"""
Train Imitation Learning Model
Learn from expert StarCraft 2 replays
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path

import sys
sys.path.append('..')
from config import *
from imitation_model import create_imitation_model


class ReplayDataset(Dataset):
    """
    Dataset for loading replay data
    In a real implementation, this would load actual parsed replays
    """
    
    def __init__(self, data_dir, num_replays=1000):
        self.data_dir = Path(data_dir)
        self.num_replays = num_replays
        
        # In production, load actual replay files here
        # For now, we'll simulate with random data
        print(f"Loading {num_replays} replays from {data_dir}...")
        print("(In production, this would parse actual .SC2Replay files)")
        
        # Simulated: each replay has varying length
        self.replay_lengths = torch.randint(100, 500, (num_replays,))
        self.total_steps = self.replay_lengths.sum().item()
        
    def __len__(self):
        return self.num_replays
    
    def __getitem__(self, idx):
        """
        Returns a sequence from a replay
        
        In production, this would return:
        - screen: Visual observations
        - minimap: Minimap observations
        - nonspatial: Game info (resources, supply, etc.)
        - actions: Expert actions taken
        - spatial_coords: Where expert clicked
        """
        seq_len = self.replay_lengths[idx].item()
        
        # Simulated data (replace with actual replay parsing)
        screen = torch.randn(seq_len, 27, 84, 84)
        minimap = torch.randn(seq_len, 7, 64, 64)
        nonspatial = torch.randn(seq_len, 64)
        
        # Expert actions
        actions = torch.randint(0, 573, (seq_len,))
        spatial_x = torch.randint(0, 84, (seq_len,))
        spatial_y = torch.randint(0, 84, (seq_len,))
        
        return {
            'screen': screen,
            'minimap': minimap,
            'nonspatial': nonspatial,
            'actions': actions,
            'spatial_x': spatial_x,
            'spatial_y': spatial_y,
        }


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences
    Pads sequences to same length
    """
    # Find max length in batch
    max_len = max(item['screen'].shape[0] for item in batch)
    
    # Pad all sequences
    padded_batch = {
        'screen': [],
        'minimap': [],
        'nonspatial': [],
        'actions': [],
        'spatial_x': [],
        'spatial_y': [],
        'lengths': []
    }
    
    for item in batch:
        seq_len = item['screen'].shape[0]
        pad_len = max_len - seq_len
        
        # Pad each tensor
        padded_batch['screen'].append(
            torch.nn.functional.pad(item['screen'], (0, 0, 0, 0, 0, 0, 0, pad_len))
        )
        padded_batch['minimap'].append(
            torch.nn.functional.pad(item['minimap'], (0, 0, 0, 0, 0, 0, 0, pad_len))
        )
        padded_batch['nonspatial'].append(
            torch.nn.functional.pad(item['nonspatial'], (0, 0, 0, pad_len))
        )
        padded_batch['actions'].append(
            torch.nn.functional.pad(item['actions'], (0, pad_len), value=-1)
        )
        padded_batch['spatial_x'].append(
            torch.nn.functional.pad(item['spatial_x'], (0, pad_len), value=-1)
        )
        padded_batch['spatial_y'].append(
            torch.nn.functional.pad(item['spatial_y'], (0, pad_len), value=-1)
        )
        padded_batch['lengths'].append(seq_len)
    
    # Stack into batches
    return {
        'screen': torch.stack(padded_batch['screen']),
        'minimap': torch.stack(padded_batch['minimap']),
        'nonspatial': torch.stack(padded_batch['nonspatial']),
        'actions': torch.stack(padded_batch['actions']),
        'spatial_x': torch.stack(padded_batch['spatial_x']),
        'spatial_y': torch.stack(padded_batch['spatial_y']),
        'lengths': torch.tensor(padded_batch['lengths'])
    }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    action_correct = 0
    spatial_correct = 0
    total_steps = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        # Move to device
        screen = batch['screen'].to(device)
        minimap = batch['minimap'].to(device)
        # Upsample minimap to match screen size
        minimap = torch.nn.functional.interpolate(
            minimap.view(-1, 7, 64, 64),
            size=(84, 84),
            mode='bilinear'
        ).view(screen.shape[0], screen.shape[1], 7, 84, 84)
        
        nonspatial = batch['nonspatial'].to(device)
        actions = batch['actions'].to(device)
        spatial_x = batch['spatial_x'].to(device)
        spatial_y = batch['spatial_y'].to(device)
        lengths = batch['lengths']
        
        # Forward pass
        outputs = model(screen, minimap, nonspatial)
        
        # Compute loss (only on non-padded steps)
        action_loss = 0
        spatial_loss = 0
        n_valid = 0
        
        for i, length in enumerate(lengths):
            # Action type loss
            action_logits = outputs['action_type'][i, :length]
            action_targets = actions[i, :length]
            action_loss += criterion(action_logits, action_targets)
            
            # Spatial loss
            spatial_logits = outputs['spatial'][i, :length].reshape(length, -1)
            spatial_targets = spatial_y[i, :length] * 84 + spatial_x[i, :length]
            spatial_loss += criterion(spatial_logits, spatial_targets)
            
            # Accuracy
            action_pred = torch.argmax(action_logits, dim=-1)
            action_correct += (action_pred == action_targets).sum().item()
            
            spatial_pred = torch.argmax(spatial_logits, dim=-1)
            spatial_correct += (spatial_pred == spatial_targets).sum().item()
            
            n_valid += length
            total_steps += length
        
        # Average loss
        loss = (action_loss + spatial_loss) / len(lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), IMITATION['grad_clip'])
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc_action': f'{action_correct/total_steps:.3f}',
            'acc_spatial': f'{spatial_correct/total_steps:.3f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    action_acc = action_correct / total_steps
    spatial_acc = spatial_correct / total_steps
    
    return avg_loss, action_acc, spatial_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    action_correct = 0
    spatial_correct = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            screen = batch['screen'].to(device)
            minimap = batch['minimap'].to(device)
            minimap = torch.nn.functional.interpolate(
                minimap.view(-1, 7, 64, 64),
                size=(84, 84),
                mode='bilinear'
            ).view(screen.shape[0], screen.shape[1], 7, 84, 84)
            
            nonspatial = batch['nonspatial'].to(device)
            actions = batch['actions'].to(device)
            spatial_x = batch['spatial_x'].to(device)
            spatial_y = batch['spatial_y'].to(device)
            lengths = batch['lengths']
            
            outputs = model(screen, minimap, nonspatial)
            
            # Compute metrics
            for i, length in enumerate(lengths):
                action_logits = outputs['action_type'][i, :length]
                action_targets = actions[i, :length]
                
                spatial_logits = outputs['spatial'][i, :length].reshape(length, -1)
                spatial_targets = spatial_y[i, :length] * 84 + spatial_x[i, :length]
                
                action_pred = torch.argmax(action_logits, dim=-1)
                action_correct += (action_pred == action_targets).sum().item()
                
                spatial_pred = torch.argmax(spatial_logits, dim=-1)
                spatial_correct += (spatial_pred == spatial_targets).sum().item()
                
                total_steps += length
    
    action_acc = action_correct / total_steps
    spatial_acc = spatial_correct / total_steps
    
    return action_acc, spatial_acc


def main(args):
    """Main training function"""
    
    print("="*60)
    print("StarCraft 2 Imitation Learning Training")
    print("="*60)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ReplayDataset(
        REPLAYS_DIR,
        num_replays=int(args.num_replays * 0.8)
    )
    val_dataset = ReplayDataset(
        REPLAYS_DIR,
        num_replays=int(args.num_replays * 0.2)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=IMITATION['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=IMITATION['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create model
    print("\nCreating model...")
    model = create_imitation_model()
    
    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=IMITATION['learning_rate'],
        weight_decay=IMITATION['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # TensorBoard
    writer = SummaryWriter(LOGS_DIR / 'imitation')
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_action_acc, train_spatial_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, epoch
        )
        
        # Validate
        val_action_acc, val_spatial_acc = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train_action', train_action_acc, epoch)
        writer.add_scalar('Accuracy/train_spatial', train_spatial_acc, epoch)
        writer.add_scalar('Accuracy/val_action', val_action_acc, epoch)
        writer.add_scalar('Accuracy/val_spatial', val_spatial_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print stats
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Action Acc: {train_action_acc:.3f} | Spatial Acc: {train_spatial_acc:.3f}")
        print(f"Val   Action Acc: {val_action_acc:.3f} | Spatial Acc: {val_spatial_acc:.3f}")
        
        # Save checkpoint
        if val_action_acc > best_val_acc:
            best_val_acc = val_action_acc
            checkpoint_path = MODELS_DIR / f'imitation_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_action_acc': val_action_acc,
                'val_spatial_acc': val_spatial_acc,
            }, checkpoint_path)
            print(f"✓ Saved best model (val_acc: {val_action_acc:.3f})")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = MODELS_DIR / f'imitation_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print("="*60)
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Imitation Learning Model')
    parser.add_argument('--num-replays', type=int, default=IMITATION['num_replays'],
                       help='Number of replays to use')
    parser.add_argument('--epochs', type=int, default=IMITATION['epochs'],
                       help='Number of training epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    main(args)
