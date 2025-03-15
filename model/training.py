import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import LRS2Dataset
from data.tokenizer import CharacterTokenizer
from model.architecture import LightweightLipReader
from config import CONFIG

def train_val_split(dataset, val_ratio=0.1):
    """Split dataset into train and validation sets"""
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])

def train_model(model, train_loader, val_loader, device, config):
    """Train the model"""
    print(f"Starting training for {config['epochs']} epochs...")
    
    # Use CTC loss for sequence prediction
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    
    # Use AdamW with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Enable mixed precision for memory efficiency
    scaler = torch.cuda.amp.GradScaler()
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        train_losses = []
        
        for batch_idx, (videos, targets, target_lengths, input_lengths) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")):
            
            # Move to device
            videos = videos.to(device)
            targets = targets.to(device)
            
            # Skip empty batches
            if videos.size(0) == 0:
                continue
                
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                logits = model(videos)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Compute CTC loss
                input_lengths = torch.full((videos.size(0),), logits.size(1), dtype=torch.long)
                loss = criterion(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            
            # Free up memory
            del videos, targets, logits, log_probs
            torch.cuda.empty_cache()
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            config['checkpoints_dir'], 
            f'lip_reading_checkpoint_epoch_{epoch+1}.pt'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    return history

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for videos, targets, target_lengths, input_lengths in tqdm(val_loader, desc="Validation"):
            videos = videos.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(videos)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute CTC loss
            input_lengths = torch.full((videos.size(0),), logits.size(1), dtype=torch.long)
            loss = criterion(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
            
            val_losses.append(loss.item())
            
            # Free memory
            del videos, targets, logits, log_probs
            torch.cuda.empty_cache()
    
    avg_loss = sum(val_losses) / len(val_losses)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss
