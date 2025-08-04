# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import wandb
# from src.dataset import PolygonColorDataset
# from src.model import ConditionalUNet
# from src.utils import save_checkpoint
# import argparse
# import json

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_json', required=True)
#     parser.add_argument('--val_json', required=True)
#     parser.add_argument('--train_in', required=True)
#     parser.add_argument('--train_out', required=True)
#     parser.add_argument('--val_in', required=True)
#     parser.add_argument('--val_out', required=True)
#     parser.add_argument('--epochs', type=int, default=20)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--image_size', type=int, default=128)
#     parser.add_argument('--checkpoint_dir', default='checkpoints')
#     return parser.parse_args()

# def build_color_vocab(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     colors = sorted(list({entry['color'] for entry in data}))
#     return {c: i for i, c in enumerate(colors)}, colors

# def main():
#     args = parse_args()
#     wandb.init(project='ayna-unet-color-fill', config=vars(args))
#     config = wandb.config

#     color_to_idx, color_names = build_color_vocab(args.train_json)
#     num_colors = len(color_to_idx)

#     train_dataset = PolygonColorDataset(
#         json_path=args.train_json,
#         input_dir=args.train_in,
#         output_dir=args.train_out,
#         color_to_idx=color_to_idx,
#         image_size=config.image_size,
#         augment=True
#     )
#     val_dataset = PolygonColorDataset(
#         json_path=args.val_json,
#         input_dir=args.val_in,
#         output_dir=args.val_out,
#         color_to_idx=color_to_idx,
#         image_size=config.image_size,
#         augment=False
#     )

#     train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = ConditionalUNet(num_colors=num_colors, img_channels=1, out_channels=3).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
#     criterion = nn.MSELoss()

#     for epoch in range(config.epochs):
#         model.train()
#         train_loss = 0.0
#         for inp, color_idx, out in train_loader:
#             inp = inp.to(device)
#             color_idx = color_idx.to(device)
#             out = out.to(device)
#             pred = model(inp, color_idx)
#             loss = criterion(pred, out)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * inp.size(0)
#         train_loss /= len(train_loader.dataset)

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inp, color_idx, out in val_loader:
#                 inp = inp.to(device)
#                 color_idx = color_idx.to(device)
#                 out = out.to(device)
#                 pred = model(inp, color_idx)
#                 loss = criterion(pred, out)
#                 val_loss += loss.item() * inp.size(0)
#         val_loss /= len(val_loader.dataset)

#         wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
#         print(f"Epoch {epoch+1}/{config.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
#         os.makedirs(config.checkpoint_dir, exist_ok=True)
#         ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}.pt")
#         save_checkpoint({
#             'model_state': model.state_dict(),
#             'optim_state': optimizer.state_dict(),
#             'epoch': epoch + 1
#         }, ckpt_path)
#         wandb.save(ckpt_path)

# if __name__ == '__main__':
#     main()


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from src.dataset import PolygonColorDataset
from src.model import ConditionalUNet
from src.utils import save_checkpoint
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', required=True)
    parser.add_argument('--val_json', required=True)
    parser.add_argument('--train_in', required=True)
    parser.add_argument('--train_out', required=True)
    parser.add_argument('--val_in', required=True)
    parser.add_argument('--val_out', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    return parser.parse_args()

def normalize_entry(entry):
    # support both {"input","color","output"} and {"input_polygon","colour","output_image"}
    in_name = entry.get('input') or entry.get('input_polygon')
    color = entry.get('color') or entry.get('colour')
    out_name = entry.get('output') or entry.get('output_image')
    if in_name is None or color is None or out_name is None:
        raise ValueError(f"Entry missing required field: {entry}")
    return {'input': in_name, 'color': color, 'output': out_name}

def load_and_normalize(json_path):
    with open(json_path, 'r') as f:
        raw = json.load(f)
    normalized = [normalize_entry(e) for e in raw]
    return normalized

def build_color_vocab(json_path):
    data = load_and_normalize(json_path)
    colors = sorted(list({entry['color'] for entry in data}))
    return {c: i for i, c in enumerate(colors)}, colors

class FlexiblePolygonColorDataset(PolygonColorDataset):
    def __init__(self, json_path, input_dir, output_dir, color_to_idx, image_size=128, augment=False):
        # load and normalize JSON to the expected format, then dump to temp list
        self.data = load_and_normalize(json_path)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.color_to_idx = color_to_idx
        self.image_size = image_size
        self.augment = augment
        import torchvision.transforms as T
        base_transforms = [T.Resize((image_size, image_size))]
        if augment:
            base_transforms += [
                T.RandomRotation(30),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0))
            ]
        self.input_transform = T.Compose(base_transforms + [T.Grayscale(num_output_channels=1), T.ToTensor()])
        self.output_transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        in_name = entry['input']
        color_name = entry['color']
        out_name = entry['output']
        in_path = os.path.join(self.input_dir, in_name)
        out_path = os.path.join(self.output_dir, out_name)
        from PIL import Image
        in_img = Image.open(in_path).convert('RGB')
        out_img = Image.open(out_path).convert('RGB')
        in_tensor = self.input_transform(in_img)
        out_tensor = self.output_transform(out_img)
        import torch
        color_idx = torch.tensor(self.color_to_idx[color_name], dtype=torch.long)
        return in_tensor, color_idx, out_tensor

def main():
    args = parse_args()
    wandb.init(project='ayna-unet-color-fill', config=vars(args))
    config = wandb.config

    color_to_idx, color_names = build_color_vocab(args.train_json)
    num_colors = len(color_to_idx)

    train_dataset = FlexiblePolygonColorDataset(
        json_path=args.train_json,
        input_dir=args.train_in,
        output_dir=args.train_out,
        color_to_idx=color_to_idx,
        image_size=config.image_size,
        augment=True
    )
    val_dataset = FlexiblePolygonColorDataset(
        json_path=args.val_json,
        input_dir=args.val_in,
        output_dir=args.val_out,
        color_to_idx=color_to_idx,
        image_size=config.image_size,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalUNet(num_colors=num_colors, img_channels=1, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for inp, color_idx, out in train_loader:
            inp = inp.to(device)
            color_idx = color_idx.to(device)
            out = out.to(device)
            pred = model(inp, color_idx)
            loss = criterion(pred, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inp.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, color_idx, out in val_loader:
                inp = inp.to(device)
                color_idx = color_idx.to(device)
                out = out.to(device)
                pred = model(inp, color_idx)
                loss = criterion(pred, out)
                val_loss += loss.item() * inp.size(0)
        val_loss /= len(val_loader.dataset)

        wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f"Epoch {epoch+1}/{config.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}.pt")
        save_checkpoint({
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'epoch': epoch + 1
        }, ckpt_path)
        wandb.save(ckpt_path)

if __name__ == '__main__':
    main()
