import math
import torch

def adjust_lr(optimizer,LR_max,LR_min,e,num_epochs,warm_e,period=1):
    "adjusted learning cosine learning rate with linear warm-up"
    if e < warm_e: #linear warmup
        lr = LR_max * e / warm_e
    else: #cosine anearling 
        lr = LR_min + 0.5 * (LR_max-LR_min) * (1.0 + math.cos(period*math.pi * (e - warm_e) / (num_epochs - warm_e))) 
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group: # account for learning rate scale in optimizer
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    
def euclid_dist(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def create_motion_robust(keypoints, pairs):
    num_steps = 3
    limbs = []
    for pose in keypoints:
        pose_limbs = []
        for start, end in pairs:
            start_point = pose[start]
            end_point = pose[end]
            if start_point[0] <= 0.01 or start_point[1] <= 0.01 or end_point[0] <= 0.01 or end_point[1] <= 0.01:
                limb = torch.zeros((num_steps, 2))
            else:
                x_linspace = torch.linspace(start_point[0], end_point[0], steps=num_steps)
                y_linspace = torch.linspace(start_point[1], end_point[1], steps=num_steps)
                limb = torch.stack([x_linspace, y_linspace], dim=1)
            pose_limbs.append(limb)
        limbs.append(torch.stack(pose_limbs))
    return torch.stack(limbs)

def get_2d_sincos_pos_embed(embed_dim, seq_len, cls_token=False):
    "Used from MAE paper"
    pos = np.arange(seq_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    "Used from MAE paper"
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D) ## specific pose connection possible here.
    return emb


def main():
    return 1
        
if __name__ == '__main__':
    main()