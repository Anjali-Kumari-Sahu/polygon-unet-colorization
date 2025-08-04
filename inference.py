# import torch
# import json
# import argparse
# import os
# from src.model import ConditionalUNet
# from src.utils import visualize_output
# from PIL import Image
# import torchvision.transforms as T

# def normalize_entry(entry):
#     in_name = entry.get('input') or entry.get('input_polygon')
#     color = entry.get('color') or entry.get('colour')
#     out_name = entry.get('output') or entry.get('output_image')
#     if in_name is None or color is None or out_name is None:
#         raise ValueError(f"Entry missing required field: {entry}")
#     return {'input': in_name, 'color': color, 'output': out_name}

# def load_and_normalize(json_path):
#     with open(json_path, 'r') as f:
#         raw = json.load(f)
#     return [normalize_entry(e) for e in raw]

# def build_color_vocab(json_path):
#     data = load_and_normalize(json_path)
#     colors = sorted(list({entry['color'] for entry in data}))
#     return {c: i for i, c in enumerate(colors)}, colors

# class SimpleDataset:
#     def __init__(self, json_path, input_dir, output_dir, color_to_idx, image_size=128):
#         self.entries = load_and_normalize(json_path)
#         self.input_dir = input_dir
#         self.output_dir = output_dir
#         self.color_to_idx = color_to_idx
#         self.image_size = image_size
#         base = [T.Resize((image_size, image_size))]
#         self.input_transform = T.Compose(base + [T.Grayscale(num_output_channels=1), T.ToTensor()])
#         self.output_transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

#     def __len__(self):
#         return len(self.entries)

#     def __getitem__(self, idx):
#         entry = self.entries[idx]
#         in_path = os.path.join(self.input_dir, entry['input'])
#         out_path = os.path.join(self.output_dir, entry['output'])
#         in_img = Image.open(in_path).convert('RGB')
#         out_img = Image.open(out_path).convert('RGB')
#         in_tensor = self.input_transform(in_img)
#         out_tensor = self.output_transform(out_img)
#         color_idx = torch.tensor(self.color_to_idx[entry['color']], dtype=torch.long)
#         return in_tensor, color_idx, out_tensor

# def run_single_example(checkpoint_path, json_path, input_dir, output_dir, image_size=128):
#     color_to_idx, color_names = build_color_vocab(json_path)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = ConditionalUNet(num_colors=len(color_to_idx), img_channels=1, out_channels=3).to(device)
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state'])
#     model.eval()

#     dataset = SimpleDataset(json_path, input_dir, output_dir, color_to_idx, image_size=image_size)
#     inp, color_idx, target = dataset[0]
#     with torch.no_grad():
#         pred = model(inp.unsqueeze(0).to(device), color_idx.unsqueeze(0).to(device))
#     visualize_output(inp, color_idx, target, pred[0], color_names)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--checkpoint', required=True)
#     parser.add_argument('--json', required=True)
#     parser.add_argument('--input_dir', required=True)
#     parser.add_argument('--output_dir', required=True)
#     parser.add_argument('--image_size', type=int, default=128)
#     args = parser.parse_args()
#     run_single_example(
#         checkpoint_path=args.checkpoint,
#         json_path=args.json,
#         input_dir=args.input_dir,
#         output_dir=args.output_dir,
#         image_size=args.image_size
#     )

import torch
import json
import argparse
import os
from src.model import ConditionalUNet
from src.utils import visualize_output
from PIL import Image
import torchvision.transforms as T

def normalize_entry(entry):
    in_name = entry.get('input') or entry.get('input_polygon')
    color = entry.get('color') or entry.get('colour')
    out_name = entry.get('output') or entry.get('output_image')
    if in_name is None or color is None or out_name is None:
        raise ValueError(f"Entry missing required field: {entry}")
    return {'input': in_name, 'color': color, 'output': out_name}

def load_and_normalize(json_path):
    with open(json_path, 'r') as f:
        raw = json.load(f)
    return [normalize_entry(e) for e in raw]

def build_color_vocab(json_path):
    data = load_and_normalize(json_path)
    colors = sorted(list({entry['color'] for entry in data}))
    return {c: i for i, c in enumerate(colors)}, colors

class SimpleDataset:
    def __init__(self, json_path, input_dir, output_dir, color_to_idx, image_size=128):
        self.entries = load_and_normalize(json_path)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.color_to_idx = color_to_idx
        self.image_size = image_size
        base = [T.Resize((image_size, image_size))]
        self.input_transform = T.Compose(base + [T.Grayscale(num_output_channels=1), T.ToTensor()])
        self.output_transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        in_path = os.path.join(self.input_dir, entry['input'])
        out_path = os.path.join(self.output_dir, entry['output'])
        in_img = Image.open(in_path).convert('RGB')
        out_img = Image.open(out_path).convert('RGB')
        in_tensor = self.input_transform(in_img)
        out_tensor = self.output_transform(out_img)
        color_idx = torch.tensor(self.color_to_idx[entry['color']], dtype=torch.long)
        return in_tensor, color_idx, out_tensor

def run_single_example(checkpoint_path, vocab_json, example_json, input_dir, output_dir, image_size=128):
    color_to_idx, color_names = build_color_vocab(vocab_json)  # vocab from training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalUNet(num_colors=len(color_to_idx), img_channels=1, out_channels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    dataset = SimpleDataset(example_json, input_dir, output_dir, color_to_idx, image_size=image_size)
    inp, color_idx, target = dataset[0]
    with torch.no_grad():
        pred = model(inp.unsqueeze(0).to(device), color_idx.unsqueeze(0).to(device))
    visualize_output(inp, color_idx, target, pred[0], color_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vocab_json', required=True, help='JSON used to build color vocabulary (e.g., training JSON)')
    parser.add_argument('--example_json', required=True, help='JSON containing examples to run (e.g., validation JSON)')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--image_size', type=int, default=128)
    args = parser.parse_args()
    run_single_example(
        checkpoint_path=args.checkpoint,
        vocab_json=args.vocab_json,
        example_json=args.example_json,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size
    )
