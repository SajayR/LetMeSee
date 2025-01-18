import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import datasets
from torch.optim import AdamW
from tqdm import tqdm
from model import SparseViT, visualize_masks
import os

DO_WANDB = True

if DO_WANDB:
    import wandb

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['jpg']
        label = sample['cls']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_one_epoch(model, loader, optimizer, epoch, 
                    sparsity_weight=1.70, binary_weight=10,
                    viz_every_n_steps=100, step=0,
                    device='cuda'):
    model.train()
    total_loss = 0
    task_loss_total = 0
    sparsity_loss_total = 0
    binary_loss_total = 0
    accuracy_total = 0
    selection_rate_total = 0
    criterion = torch.nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, masks = model(images)
        
        task_loss = criterion(logits, targets)
        sparsity_loss = model.get_sparsity_loss(masks) * sparsity_weight
        binary_loss = model.get_binary_regularization(masks)# * binary_weight
        
        loss = task_loss + sparsity_loss + binary_loss
        loss.backward()
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        task_loss_total += task_loss.item()
        sparsity_loss_total += sparsity_loss.item()
        binary_loss_total += binary_loss.item()

        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).float().mean().item()
        selection_rate = (masks > 0.5).float().mean().item()
        
        accuracy_total += accuracy
        selection_rate_total += selection_rate
        
        if batch_idx % viz_every_n_steps == 0:
            # Visualize a few sample masks
            for i in range(min(5, len(images))):
                viz_path = f'viz/step_{epoch}_{batch_idx}_sample_{i}.png'
                visualize_masks(
                    images[i].cpu(), 
                    masks[i].cpu(),
                    save_path=viz_path
                )
                if DO_WANDB:
                    wandb.log({f"Sample_{i}": wandb.Image(viz_path)}, step=step)
        
        pbar.set_postfix({
            'loss': total_loss/(batch_idx+1),
            'acc': accuracy_total/(batch_idx+1),
            'selected': f"{selection_rate_total/(batch_idx+1):.1%}",
            'step': batch_idx
        })
        
        if DO_WANDB:
            wandb.log({
                'train/loss': loss.item(),
                'train/task_loss': task_loss.item(),
                'train/sparsity_loss': sparsity_loss.item(),
                'train/binary_loss': binary_loss.item(),
                'train/accuracy': accuracy,
                'train/selection_rate': selection_rate,
                'train/step': step
            })
        step += 1
    
    epoch_metrics = {
        'loss': total_loss / len(loader),
        'task_loss': task_loss_total / len(loader),
        'sparsity_loss': sparsity_loss_total / len(loader),
        'binary_loss': binary_loss_total / len(loader)
    }
    if DO_WANDB:
        wandb.log({f"train/epoch_{epoch}_metrics": epoch_metrics})
    return epoch_metrics, step

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_selection_rate = 0
    top5_correct = 0
    
    for images, targets in tqdm(val_loader, desc='Validation'):
        images, targets = images.to(device), targets.to(device)
        B = targets.size(0)
        
        # Inference => actual truncation
        logits = model(images)
        
        # top-1 accuracy
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        
        # top-5 accuracy
        _, top5_preds = logits.topk(5, dim=1)
        top5_correct += torch.any(top5_preds == targets.view(-1, 1), dim=1).sum().item()
        
        # For selection rate, temporarily call model in training mode for soft masks
        model.train()
        _, masks = model(images)
        model.eval()
        
        total_selection_rate += (masks > 0.5).float().mean().item() * B
        total_samples += B
    
    metrics = {
        'val_accuracy': total_correct / total_samples,
        'val_top5_accuracy': top5_correct / total_samples,
        'val_selection_rate': total_selection_rate / total_samples
    }
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform, val_transform = get_transforms()
    
    if DO_WANDB:
        wandb.init(project="sparsefudge", config={
            "epochs": 10,
            "batch_size": 64,
            "learning_rate": 3e-4,
        })
    
    # Example dataset usage
    dataset = datasets.load_dataset("ImageNet")
    train_dataset = ImageNetDataset(dataset['train'], transform=train_transform)
    val_dataset = ImageNetDataset(dataset['validation'], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=150, shuffle=True, 
                              num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            num_workers=12, pin_memory=True)
    
    model = SparseViT(num_classes=1000, temperature=1.0).to(device)
    model.load_state_dict(torch.load('checkpoint_epoch_0.pt')['model_state_dict'])
    optimizer = AdamW(model.parameters(), lr=3e-4)
    
    os.makedirs('viz', exist_ok=True)
    num_epochs = 25
    step = 0
    for epoch in range(num_epochs):
        metrics, step = train_one_epoch(
            model, train_loader, optimizer, epoch,
            viz_every_n_steps=1000,
            step=step,
            device=device
        )
        print(f"Epoch {epoch} metrics:", metrics)

        val_metrics = validate(model, val_loader, device)
        print(f"Epoch {epoch} validation metrics:", val_metrics)
        
        if DO_WANDB:
            wandb.log({
                'val/accuracy': val_metrics['val_accuracy'],
                'val/top5_accuracy': val_metrics['val_top5_accuracy'],
                'val/selection_rate': val_metrics['val_selection_rate'],
                'epoch': epoch,
            })

        # Save checkpoint
        if epoch % 1 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
            # wandb.save(f'checkpoint_epoch_{epoch}.pt')
            
                    
    if DO_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()
