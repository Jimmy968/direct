import os
import h5py
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Optional
import argparse
import ast

##########################################################
###### DEFINITIONS FROM CMRXRECON EVALUATION SCRIPT ######
##########################################################

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if maxval is None:
        maxval = gt.max()
    return structural_similarity(gt, pred, data_range=maxval)


def calmetric(pred_recon, gt_recon):
    if gt_recon.ndim == 4:
        psnr_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        ssim_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        nmse_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))

        for i in range(gt_recon.shape[-2]):
            for j in range(gt_recon.shape[-1]):
                pred, gt = pred_recon[:, :, i, j], gt_recon[:, :, i, j]
                psnr_array[i, j] = psnr(gt / gt.max(), pred / pred.max())
                ssim_array[i, j] = ssim(gt / gt.max(), pred / pred.max())
                nmse_array[i, j] = nmse(gt / gt.max(), pred / pred.max())
    else:
        psnr_array = np.zeros((1, gt_recon.shape[-1]))
        ssim_array = np.zeros((1, gt_recon.shape[-1]))
        nmse_array = np.zeros((1, gt_recon.shape[-1]))

        for j in range(gt_recon.shape[-1]):
            pred, gt = pred_recon[:, :, j], gt_recon[:, :, j]
            psnr_array[0,j] = psnr(gt / gt.max(), pred / pred.max())
            ssim_array[0,j] = ssim(gt / gt.max(), pred / pred.max())
            nmse_array[0,j] = nmse(gt / gt.max(), pred / pred.max())

    return psnr_array, ssim_array, nmse_array

##########################################################
##########################################################

# Set up logging for debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_valid_file_pairs(recon_dir, gt_base_dir):
    """Pre-scan directories to identify valid reconstruction and ground-truth file pairs."""
    file_pairs = []
    for filename in sorted(os.listdir(recon_dir)):
        if not filename.endswith('.mat'):
            continue
        parts = filename.split('_')
        if len(parts) != 3 or parts[1] != 'cine':
            continue
        patient = parts[0]  # e.g., 'P001'
        view = parts[2].replace('.mat', '').lower()  # 'sax' or 'lax'
        if view not in ['sax', 'lax']:
            continue
        recon_path = os.path.join(recon_dir, filename)
        gt_path = os.path.join(gt_base_dir, patient, f'cine_{view}.mat')
        if os.path.exists(gt_path):
            file_pairs.append((recon_path, gt_path, patient, view))
        else:
            logger.warning(f"Ground-truth file not found for {patient} {view}")
    return file_pairs


def process_file_pair(file_pair):
    """Process a single file pair (reconstruction and ground-truth) and compute metrics."""
    recon_path, gt_path, patient, view = file_pair
    try:
        # Load only the required datasets
        with h5py.File(recon_path, 'r') as f:
            img_recon = f['reconstruction'][()]

        with h5py.File(gt_path, 'r') as f:
            kspace_gt = f['kspace_full'][()]

        # Handle complex k-space data
        kspace_gt = kspace_gt['real'] + 1j * kspace_gt['imag']
        
        # iFFT + RSS
        img_gt = np.linalg.norm(
            np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace_gt), axes=(-2, -1))),
            axis=2
        )
        
        # Transpose to match shapes: x, y, slice, t
        img_recon = img_recon.transpose(2, 3, 0, 1)
        img_gt = img_gt.transpose(3, 2, 1, 0)
        
        # Compute metrics
        psnr_array, ssim_array, nmse_array = calmetric(img_recon, img_gt)
        
        return {
            'patient': patient,
            'view': view,
            'mean_psnr': psnr_array.mean(),
            'mean_ssim': ssim_array.mean(),
            'mean_nmse': nmse_array.mean()
        }
    except Exception as e:
        logger.error(f"Error processing {patient} {view}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="MRI Reconstruction Evaluation")
    parser.add_argument("--gt_base_dir", type=str, required=True, help="Path to ground truth base directory")
    parser.add_argument("--recon_dir", type=str, required=True, help="Path to reconstruction directory")

    args = parser.parse_args()

    gt_base_dir = args.gt_base_dir
    recon_dir = args.recon_dir

    results = []
    max_workers = min(os.cpu_count(), 4)  # Cap at 4 to avoid overloading

    file_pairs = get_valid_file_pairs(recon_dir, gt_base_dir)
    logger.info(f"Processing {len(file_pairs)} volumes in {recon_dir}")
        
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(process_file_pair)
        batch_results = executor.map(process_func, file_pairs)
        for result in batch_results:
            if result is not None:
                logger.info(f"Processed {result['patient']} {result['view']} with mPSNR: {result['mean_psnr']:.3f}, mSSIM: {result['mean_ssim']:.3f}, mNMSE: {result['mean_nmse']:.3f}")
                results.append(result)
    
    df = pd.DataFrame(results)

    metrics = df[['mean_psnr', 'mean_ssim', 'mean_nmse']].agg(['mean', 'std']).round(3).to_string()
    logger.info("Average metrics:\n%s", metrics)
    
    output_path = os.path.join(recon_dir, 'mri_reconstruction_metrics.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()

