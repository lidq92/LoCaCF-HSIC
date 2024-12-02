# https://github.com/Anserw/Bjontegaard_metric
import csv
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)
    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)
    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)
    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))
    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)
    # find avg diff
    avg_diff = (int2 - int1) / (max_int - min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)
    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)
    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))
    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)
    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100

    return avg_diff


def read_csv(csv_file_path, N=50, start_k=1, end_k_p1=7, n_metrics=4):
    rps = end_k_p1 - start_k
    psnrs, bpsp_values, bits = np.zeros((N, rps)), np.zeros((N, rps)), np.zeros((N, rps))
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  
        r = 0
        for row in reader:
            for k in range(rps):
                psnrs[r, k] = float(row[n_metrics * (start_k + k) + 1])
                bpsp_values[r, k] = float(row[n_metrics * (start_k + k) + 2]) 
                if n_metrics == 2:
                    continue
                # mse_values[r, k] = float(row[n_metrics * (start_k + k) + 3]) 
                bits[r, k] = float(row[n_metrics * (start_k + k) + 4])  
            r = r + 1
            if r == N: break #

        return psnrs, bpsp_values, bits


def SOTA():
    N = 50 # 
    methods = [
        'JPEG2000',
        'JPEGXL',
        'HEVC',
        'VVC',
        'LineRWKV_xs',
        'LineRWKV_l',
    ]
    showed_methods = [
        'JPEG 2000',
        'JPEG XL',
        'HEVC',
        'VVC',
        'LineRWKV$_{XS}$',
        'LineRWKV$_{L}$',
    ]
    start_ks = [
        1, 
        1,
        1, 
        1, 
        1, 
        1, 
    ]
    end_k_p1s = [
        12,
        12,
        19,
        19,
        9,
        9,
    ]
    plt.figure(figsize=(8, 6)) 
    for idx, method in enumerate(methods):
        start_k, end_k_p1 = start_ks[idx], end_k_p1s[idx]
        csv_file_path = f'SOTA_results/test_{method}-LoCaCF.csv'
        LSR_psnrs, LSR_bpsp_values, LSR_bits_values = read_csv(csv_file_path, N, start_k, end_k_p1)

        csv_file_path = f'SOTA_results/test_{method}.csv'
        base_psnrs, base_bpsp_values, base_bits_values = read_csv(csv_file_path, N, start_k, end_k_p1)

        thres = 2 # bpsp <= 2
        base_R = base_bits_values.mean(axis=0)[base_bpsp_values.mean(axis=0) <= thres]
        base_bpsp = base_bpsp_values.mean(axis=0)[base_bpsp_values.mean(axis=0) <= thres]
        base_D = base_psnrs.mean(axis=0)[base_bpsp_values.mean(axis=0) <= thres]
        LSR_R = LSR_bits_values.mean(axis=0)[LSR_bpsp_values.mean(axis=0) <= thres]
        LSR_bpsp = LSR_bpsp_values.mean(axis=0)[LSR_bpsp_values.mean(axis=0) <= thres]
        LSR_D = LSR_psnrs.mean(axis=0)[LSR_bpsp_values.mean(axis=0) <= thres]
        
        bd_rate = BD_RATE(base_R, base_D, LSR_R, LSR_D)
        bd_psnr = BD_PSNR(base_R, base_D, LSR_R, LSR_D)
        print(f'{showed_methods[idx]}: AVG BD-PSNR={round(bd_psnr,4)}, AVG BD-Rate={round(bd_rate, 4)}%')

        plt.plot(LSR_bpsp, LSR_D, label=f'Ours-{showed_methods[idx]}', linestyle='-', marker='o', alpha=0.9)
        color = plt.gca().lines[-1].get_color()
        plt.plot(base_bpsp, base_D, label=showed_methods[idx], linestyle='--', marker='x', color=color, alpha=0.9)
    plt.legend(loc='best')
    plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
    plt.savefig(f'SOTA_results/Average_RD_Curve_SOTA.png', dpi=300) 
    plt.close()


if __name__ == "__main__":
    SOTA()
