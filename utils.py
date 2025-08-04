# import torch
# import matplotlib.pyplot as plt

# def save_checkpoint(state, path):
#     torch.save(state, path)

# def load_checkpoint(path, model, optimizer=None):
#     checkpoint = torch.load(path, map_location='cpu')
#     model.load_state_dict(checkpoint['model_state'])
#     if optimizer and 'optim_state' in checkpoint:
#         optimizer.load_state_dict(checkpoint['optim_state'])
#     return checkpoint

# def visualize_output(input_img, color_idx, target_img, pred_img, color_names, save_path=None):
#     fig, axs = plt.subplots(1, 4, figsize=(12, 3))
#     # input_img: [C,H,W] or [1,H,W]
#     axs[0].imshow(input_img.squeeze().cpu().permute(1, 2, 0).clamp(0, 1))
#     axs[0].set_title('Input Polygon')
#     axs[1].imshow(target_img.cpu().permute(1, 2, 0).clamp(0, 1))
#     axs[1].set_title('Ground Truth')
#     axs[2].imshow(pred_img.detach().cpu().permute(1, 2, 0).clamp(0, 1))
#     axs[2].set_title('Prediction')
#     axs[3].text(0.1, 0.5, f"Color: {color_names[color_idx]}", fontsize=12)
#     axs[3].axis('off')
#     for ax in axs:
#         ax.set_xticks([]); ax.set_yticks([])
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()


import torch
import matplotlib.pyplot as plt

def _to_image(tensor):
    # Accept [C,H,W], [1,H,W], or [H,W]; return HxW or HxWx3 numpy array ready for imshow
    t = tensor.detach().cpu()
    if t.dim() == 2:  # [H,W]
        img = t
    elif t.dim() == 3:
        if t.size(0) == 1:  # grayscale [1,H,W]
            img = t.squeeze(0)
        else:  # [C,H,W]
            img = t.permute(1, 2, 0)
    else:
        raise ValueError(f"Unsupported tensor shape for image: {t.shape}")
    img = img.clamp(0, 1).numpy()
    return img

def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optim_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optim_state'])
    return checkpoint

def visualize_output(input_img, color_idx, target_img, pred_img, color_names, save_path=None):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    inp_img = _to_image(input_img)
    tgt_img = _to_image(target_img)
    prd_img = _to_image(pred_img)

    axs[0].imshow(inp_img, interpolation='nearest')
    axs[0].set_title('Input Polygon')
    axs[1].imshow(tgt_img, interpolation='nearest')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(prd_img, interpolation='nearest')
    axs[2].set_title('Prediction')
    axs[3].text(0.1, 0.5, f"Color: {color_names[color_idx]}", fontsize=12)
    axs[3].axis('off')
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
