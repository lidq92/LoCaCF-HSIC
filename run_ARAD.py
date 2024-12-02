import os
import csv
import warnings
import resource
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import numpy as np
from rasterio.errors import NotGeoreferencedWarning

import Codecs
import encode
import decode


MAX_WORKERS = int( 2 * os.cpu_count() // 3) #
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft_limit = 65535 #
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def run_one_rate_point(method, file_path, i, s, q, D, precision, output_dir='outputs'):   
    
    encode.train(file_path, method, s, q, D, precision, output_dir) 

    filename = os.path.splitext(os.path.basename(file_path))[0]
    fs = '{}/{}_{}_s{}_Q{}_D{}_prec{}'
    output_dir = fs.format(output_dir, filename, method, 
                           s, q, D, precision)
    bin_path = f'{output_dir}/{filename}.bin'
    recon_path = f'{output_dir}/{filename}_recon.tif'
    decode.test(bin_path)

    mse_value, psnr, bits, bpsp = Codecs.eval_RD(bin_path, recon_path, file_path.replace('.yuv', '.tif'))
    
    subprocess.call(f'rm -f {bin_path} {recon_path}', shell=True)
    print(f"file: {file_path} @q={q}, MSE: {mse_value}, PSNR: {psnr}, bits: {bits}, bpsp: {bpsp}")
    
    return file_path.replace('.yuv', '.tif'), q, psnr, bpsp, mse_value, bits, i


def main(start_id=901, end_id_plus_1=951):
    s = 1.0 #
    D = 0 #
    precision = 32 #
    
    file_paths = [
        f"data/ARAD/ARAD_1K_{idx:04d}.mat" for idx in range(start_id, end_id_plus_1)
    ]
    output_paths = [
        f"data/ARAD/ARAD_1K_{idx:04d}.tif" for idx in range(start_id, end_id_plus_1)
    ]

    NBITS = 12
    for i, file_path in enumerate(file_paths):
        if os.path.exists(output_paths[i]):
            continue
        file = h5py.File(file_path, 'r')
        hsi = file['cube'][:].transpose(0, 2, 1)
        hsi = hsi.clip(min=0, max=1) # Handling ARAD_1K_0929
        org_img = np.round(hsi * (2 ** NBITS - 1)).astype('<u2')
        Codecs.write_tiff_with_rasterio(output_paths[i], org_img) # preprocessing to tif

    methods = [
        'JPEG2000',
        'JPEGXL',
        'HEVC',
        'VVC',
    ]
    qs = {
        'JPEG2000': [100, 36, 32, 28, 24, 20, 16, 12, 8, 6, 4, 2],
        'JPEGXL': [100, 99.9, 99.5, 99, 98, 96, 94, 90, 85, 80, 70, 50],
        'HEVC': ['lossl', *range(-6 * (NBITS - 8), -6 * (NBITS - 8) + 54, 3)],
        'VVC': ['lossl', *range(-6 * (NBITS - 8), -6 * (NBITS - 8) + 54, 3)],
        # https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/blob/master/source/Lib/CommonLib/CommonDef.h?ref_type=heads#L296
    }
    SOTA_results_dir = 'SOTA_results'
    os.makedirs(SOTA_results_dir, exist_ok=True)

    for method in methods:
        csv_file = f'{SOTA_results_dir}/test_{method}-LoCaCF.csv'
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            metrics = ['PSNR', 'bpsp', 'MSE', 'bits']
            
            csv_headers = ['Path'] + [f"q={q}_{metric}" for q in qs[method] for metric in metrics]
            writer.writerow(csv_headers)

            futures = []
            global MAX_WORKERS
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for i, file_path in enumerate(output_paths): 
                    for q_idx, q in enumerate(qs[method]):
                        if q_idx == 0: continue # no lossless
                        futures.append(executor.submit(run_one_rate_point, method, file_path, i, s, q, D, precision))
                        
            results = {}
            for future in as_completed(futures):
                try:
                    file_path, q, psnr, bpsp, mse_value, bits, i = future.result()
                    if file_path not in results:
                        results[file_path] = [None] * (len(qs[method]) * len(metrics))
                    q_idx = qs[method].index(q)
                    results[file_path][len(metrics) * q_idx] = psnr
                    results[file_path][len(metrics) * q_idx + 1] = bpsp
                    results[file_path][len(metrics) * q_idx + 2] = mse_value
                    results[file_path][len(metrics) * q_idx + 3] = bits
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

            for file_path in output_paths:
                row = [file_path] + results.get(file_path, [None] * (len(qs[method]) * len(metrics)))
                writer.writerow(row)


if __name__ == "__main__":
    main()
