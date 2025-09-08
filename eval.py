import numpy as np
import argparse
import os
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms
from skimage.io import imread
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
from core.inception import InceptionV3
import pathlib


# Frechet Distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def psnr(frames1, frames2):
    error = 0
    for i in range(len(frames1)):
        error += compare_psnr(frames1[i], frames2[i])
    return error / len(frames1)

# Inception activations
def get_activations(images, model, device, batch_size=64, dims=2048):
    model.eval()
    n_images = images.shape[0]
    batch_size = min(batch_size, n_images)
    n_batches = n_images // batch_size
    pred_arr = np.empty((n_batches * batch_size, dims), dtype=np.float32)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch = torch.from_numpy(images[start:end]).float().to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().numpy().reshape(batch_size, -1)
    return pred_arr


# Activation
def calculate_activation_statistics(images, model, device, batch_size=64, dims=2048):
    act = get_activations(images, model, device, batch_size, dims)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# mu,sigma
def _compute_statistics_of_path(path, model, device, batch_size, dims):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'], f['sigma']
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        imgs = np.array([imread(str(fn)).astype(np.float32) for fn in files])
        imgs = imgs.transpose((0, 3, 1, 2)) / 255.0
        m, s = calculate_activation_statistics(imgs, model, device, batch_size, dims)
    return m, s


# FID
def calculate_fid_given_paths(paths, batch_size, device, dims):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError(f'Invalid path: {p}')
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = _compute_statistics_of_path(paths[0], model, device, batch_size, dims)
    m2, s2 = _compute_statistics_of_path(paths[1], model, device, batch_size, dims)
    return calculate_frechet_distance(m1, s1, m2, s2)


def read_images_from_folder(folder_path):
    imgs = []
    for fn in sorted(os.listdir(folder_path)):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            imgs.append(imread(os.path.join(folder_path, fn)))
    return imgs


# Preprocess
torch.manual_seed(0)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda im: im.convert('RGB')),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation on the dataset')
    parser.add_argument('--path', type=str, default='./checkpoints/predict_MISATO_C24_1222',
                        help='Path to images')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use GPU if available')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # LPIPS model (GPU)
    lpips_model = lpips.LPIPS(net='alex').to(device)

    results = []
    txt = [0, 0]
    cont = 0
    for i in range(10, 41, 10):
        cont += 1
        gt_path = f'{args.path}/{i}-{i + 10}/truth'
        out_path = f'{args.path}/{i}-{i + 10}/out'
        txt_path = f'{args.path}/{i}-{i + 10}/result.txt'
        f = open(txt_path, 'r')
        lines = f.read()
        lines = lines.split(' ')
        txt[0] += float(lines[1].split(':')[1])
        txt[1] += float(lines[2].split(':')[1])
        f.close()
        # FID
        fid_val = calculate_fid_given_paths([gt_path, out_path], batch_size=64, device=device, dims=2048)

        # preprocess
        gt_imgs = read_images_from_folder(gt_path)
        out_imgs = read_images_from_folder(out_path)
        gt_raw = torch.stack([preprocess(img) for img in gt_imgs]).to(device)
        out_raw = torch.stack([preprocess(img) for img in out_imgs]).to(device)

        psnr_val=psnr(gt_imgs, out_imgs)

        # SSIM
        gt_np = gt_raw.cpu().permute(0, 2, 3, 1).numpy()
        out_np = out_raw.cpu().permute(0, 2, 3, 1).numpy()
        ssim_score = np.mean([
            compare_ssim(gt_np[j], out_np[j], multichannel=True, win_size=51)
            for j in range(gt_np.shape[0])
        ])

        # LPIPS
        gt_norm = gt_raw * 2 - 1
        out_norm = out_raw * 2 - 1
        lpips_val = lpips_model(gt_norm, out_norm).view(-1).mean().item()

        results.append((i, i + 10, psnr_val, ssim_score, fid_val, lpips_val))
        print(f'{i}-{i + 10} PSNR:{psnr_val:.4f} SSIM:{ssim_score:.4f} FID:{fid_val:.4f} LPIPS:{lpips_val:.4f}')
    print(f'Inference Time:{txt[0] / cont} FPS:{txt[1] / cont}')
    with open(os.path.join(args.path, 'result.txt'), 'a+') as f:
        for r in results:
            f.write(f'{r[0]}-{r[1]} PSNR:{r[2]:.4f} SSIM:{r[3]:.4f} FID:{r[4]:.4f} LPIPS:{r[5]:.4f}\n')
        f.write(f'Inference Time:{txt[0] / cont} FPS:{txt[1] / cont}\n')
    print('Evaluation completed.')
