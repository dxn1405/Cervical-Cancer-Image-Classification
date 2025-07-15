import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler

from models.DeiT import DeiT
from models.crossViT import CrossViT
from models.swin import SwinTransformer

# ==== Hyperparameters & Device Setup ====
NUM_EPOCHS   = 50
PATIENCE     = 10
BATCH_SIZE   = 32 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 40e9 else 16
NUM_WORKERS  = min(8, os.cpu_count())
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE   = 'cuda' if device.type == 'cuda' else 'cpu'
MODEL_DIR    = "./model_checkpoints"
DATA_ROOT    = "/content/Sipakmed_Processed"  # contains Train/, Val/, Test/
os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True


# ==== Dataset & On-the-fly Augmentation ====
class MedicalImageDataset(Dataset):
   def __init__(self, root_dir, transform):
       self.classes = sorted(os.listdir(root_dir))
       self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
       self.images = []
       for c in self.classes:
           for f in os.listdir(os.path.join(root_dir,c)):
               self.images.append((os.path.join(root_dir,c,f), self.class_to_idx[c]))
       self.transform = transform
   def __len__(self): return len(self.images)
   def __getitem__(self, idx):
       path, label = self.images[idx]
       img = np.array(Image.open(path).convert("RGB"))
       return self.transform(image=img)["image"], label


#====Transform====
def get_transform(image_size, is_training=False):
   if is_training:
       return A.Compose([


           A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
           A.HorizontalFlip(p=0.5),
           A.VerticalFlip(p=0.5),
           A.RandomBrightnessContrast(p=0.7),
           A.GaussNoise(p=0.3),
           A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
           ToTensorV2(),
       ])
   else:
       return A.Compose([
           # use named args or tuple here too
           A.Resize(height=image_size, width=image_size),
           A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
           ToTensorV2(),
       ])




# ==== Sampler ====
def create_balanced_sampler(ds):
   targets = [lbl for _,lbl in ds.images]
   counts = torch.bincount(torch.tensor(targets))
   weights = 1.0/counts.float()
   sample_weights = [weights[t].item() for t in targets]
   return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ==== Metrics & Plotting ====
def evaluate(model, loader, criterion):
   model.eval()
   total_loss, correct, total = 0.0, 0, 0
   y_true, y_pred = [], []
   with torch.no_grad():
       for x,y in loader:
           x,y = x.to(device), y.to(device)
           with autocast(device_type=AMP_DEVICE):
               out = model(x)
               if out.ndim==4: out=out.mean((-2,-1))
               loss = criterion(out, y)
           total_loss += loss.item()
           preds = out.argmax(1)
           correct += (preds==y).sum().item()
           total   += y.size(0)
           y_true.extend(y.cpu().tolist()); y_pred.extend(preds.cpu().tolist())
   return total_loss/len(loader), correct/total, y_true, y_pred


def plot_training_history(history, name):
   epochs = len(history['train_acc'])
   plt.figure(figsize=(12,5))
   plt.subplot(1,2,1)
   plt.plot(range(1,epochs+1), history['train_acc'], label='Train Acc')
   plt.plot(range(1,epochs+1), history['val_acc'],   label='Val Acc')
   plt.title(f'{name} Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
   plt.subplot(1,2,2)
   plt.plot(range(1,epochs+1), history['train_loss'], label='Train Loss')
   plt.plot(range(1,epochs+1), history['val_loss'],   label='Val Loss')
   plt.title(f'{name} Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
   plt.tight_layout()
   plt.savefig(f'{name}_history.png')
   plt.show()


def plot_confusion_matrix(y_true, y_pred, name):
   cm = confusion_matrix(y_true, y_pred)
   disp = ConfusionMatrixDisplay(cm)
   disp.plot(cmap='Blues', xticks_rotation='vertical')
   plt.title(f'{name} Confusion Matrix')
   plt.tight_layout()
   plt.savefig(f'{name}_confusion_matrix.png')
   plt.show()


def generate_classification_report(y_true, y_pred, name):
   print(f"\nClassification Report for {name}:")
   print(classification_report(y_true, y_pred, digits=4))


# ==== Training Loop ====
def train_model(model, name, train_loader, val_loader, criterion, optimizer, scheduler):
   history = {'train_acc':[], 'val_acc':[], 'train_loss':[], 'val_loss':[]}
   best_val_loss = float('inf')
   early_cnt = 0
   scaler = GradScaler()
   model.to(device)
   for epoch in range(NUM_EPOCHS):
       model.train()
       run_loss, corr, tot = 0.0, 0, 0
       for x,y in tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{NUM_EPOCHS}"):
           x,y = x.to(device), y.to(device)
           with autocast(device_type=AMP_DEVICE):
               out = model(x)
               if out.ndim==4: out=out.mean((-2,-1))
               loss = criterion(out, y)
           scaler.scale(loss).backward()
           scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
           run_loss += loss.item()
           preds = out.argmax(1)
           corr += (preds==y).sum().item(); tot += y.size(0)
       train_acc = corr/tot
       val_loss, val_acc, yt, yp = evaluate(model, val_loader, criterion)
       history['train_acc'].append(train_acc)
       history['train_loss'].append(run_loss/len(train_loader))
       history['val_acc'].append(val_acc)
       history['val_loss'].append(val_loss)
       print(f"{name} Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
             f"Train Loss={run_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")
       if val_loss < best_val_loss:
           torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{name}_best.pth"))
           best_val_loss = val_loss; early_cnt = 0
       else:
           early_cnt += 1
           if early_cnt >= PATIENCE:
               print("🛑 Early stopping")
               break
   return model, history


# ==== Main ====
def main():
   train_transform = get_transform(224, True)
   val_transform   = get_transform(224, False)
   train_ds = MedicalImageDataset(os.path.join(DATA_ROOT, 'Sipakmed_Train'), train_transform)
   val_ds   = MedicalImageDataset(os.path.join(DATA_ROOT, 'Sipakmed_Val'),   val_transform)
   test_ds  = MedicalImageDataset(os.path.join(DATA_ROOT, 'Sipakmed_Test'),  val_transform)


   train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=create_balanced_sampler(train_ds),
                             num_workers=NUM_WORKERS, pin_memory=True)
   val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
   test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


   models = {"Swin":SwinTransformer(), "DeiT":DeiT(), "CrossViT":CrossViT()}
   criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
   for name, mdl in models.items():
       optimizer = optim.AdamW(mdl.parameters(), lr=(5e-4 if name=="Swin" else 1e-3), weight_decay=0.05)
       scheduler = OneCycleLR(optimizer, max_lr=(5e-4 if name=="Swin" else 1e-3), epochs=NUM_EPOCHS,
                              steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='linear',
                              div_factor=25, final_div_factor=1e4)
       print(f"\n🚀 Training {name}")
       trained, history = train_model(mdl, name, train_loader, val_loader, criterion, optimizer, scheduler)
       plot_training_history(history, name)
       loss, acc, yt, yp = evaluate(trained, test_loader, criterion)
       print(f"\nTest set - {name}: Loss={loss:.4f}, Accuracy={acc:.4f}")
       plot_confusion_matrix(yt, yp, name)
       generate_classification_report(yt, yp, name)


if __name__ == "__main__":
   main()
