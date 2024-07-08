import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u
from transitleastsquares import (transitleastsquares,cleaned_array,catalog_info,transit_mask)
import sys
sys.path.append("/Users/matthewlastovka/local/TESSphomo")
from tessphomo import TESSTargetPixelModeler
from tessphomo.mast import retrieve_tess_ffi_cutout_from_mast, get_tic_sources
from astropy.stats import sigma_clip
from wotan import flatten
import pathlib
from astropy.table import QTable
import os.path
import pandas as pd
import makinglc as mk
import time

def run_the_search(ticid, sigma_upper=4., sigma_lower=12., window_length=0.8, method='biweight'):
    light_curves = []
    sectors = []
    file_test = 0
    for i in range(85):
        finame = 'tessphomo_lightcurves/tessphomo_ffi_lightcurve_sector_'+str(i).zfill(4)+'_tic_'+str(ticid).zfill(12)+'_cutout_25x25.fits'
        if os.path.exists(finame):
            print("light curve exists")
            lightc = QTable.read(finame, format='fits', astropy_native=True)
            lightc['time'] = lightc['time'].btjd
            light_curves.append(lightc)
            sectors.append(str(i).zfill(4))
            file_test += 1
    if file_test == 0:
        print("retrieving tpfs")
        all_tpfs = retrieve_tess_ffi_cutout_from_mast(ticid=ticid, cutout_size=(25,25), sector=None)
        print(len(all_tpfs), "tpfs")
        print("retrieving background stars")
        input_catalog = get_tic_sources(ticid, tpf_shape=all_tpfs[0].shape[1:])
        print(input_catalog.shape[0], "background stars")
        print("making light curves")
        
        for tpf in all_tpfs:
            try:
                TargetData = TESSTargetPixelModeler(tpf, input_catalog=input_catalog)
                lc = TargetData.get_corrected_LightCurve(assume_catalog_mag=True)
                mk.write_lc_to_fits_file(TargetData, lc, lc_save_direc='./example_data/', overwrite=True)
                sectors.append(str(TargetData.tpf.sector).zfill(4))
                light_curves.append(lc)
            except:
                print("ERROR!!!!!")
    print("Finished")
    flux_id = mk.determine_best_flux(light_curves)
    print("Detrending and clipping light curves")
    all_flat = []
    all_trend = []
    all_mask = []
    all_times = []
    for i in range(len(light_curves)):
        clipped_flux = sigma_clip(light_curves[i][flux_id].value, sigma_upper=sigma_upper, sigma_lower=sigma_lower, masked=True)
        mask = clipped_flux.mask
        times = light_curves[i]['time'].value
        flatten_lc, trend_lc = flatten(times[~mask], clipped_flux.data[~mask], window_length=window_length, 
                                   return_trend=True, method=method, break_tolerance=0.1)
        all_flat.append(flatten_lc)
        all_trend.append(trend_lc)
        all_mask.append(mask)
        all_times.append(times[~mask])
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=ticid)
    print("running transit search")
    all_results = []
    for i in range(len(light_curves)):
        model = transitleastsquares(all_times[i], all_flat[i], verbose=True)
        results = model.power(u=ab, use_threads=14)
        all_results.append(results)
    print("Determine Flag")
    flag = mk.flagging_criteria(all_results)
    if flag is True:
        flag_file = './transit_search/flagged_tic.txt'
        file1 = open(flag_file, "a")  # append mode
        file1.write(str(ticid) + '\n')
        file1.close()
    print("saving transit search results")
    params = [flag, sigma_upper, sigma_lower, window_length, method, flux_id]
    path_name = './transit_search/TIC_'+str(ticid)
    new_dir = pathlib.Path(path_name)
    new_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(all_results)):
        file_name = path_name + '/transitsearch_TIC_' + str(ticid) + '_sector_' + str(sectors[i]) + '_'
        mk.save_results_file(file_name, results, params)

if __name__ == "__main__":
    start_time = time.time()

    #flag_file = './transit_search/flagged_tic.txt'
    #file1 = open(flag_file, "w")  # append mode
    #file1.close()

    n2_list = pd.read_csv("n2_targets_list.csv")
    T_mask = (n2_list['Tmag'] > 10.5) & (n2_list['Tmag'] < 15)
    n2_targ = n2_list[T_mask]
    n2_targ = n2_targ.sort_values('MES', ascending=False).reset_index()

    for tic_index in range(162, 182):
        ticid = int(n2_targ['ID'][tic_index])
        #ticid = 154383758
        print("TIC", ticid)
        print("TESSmag", n2_targ['Tmag'][tic_index])
        run_the_search(ticid)

    end_time = time.time()

    runtime = (end_time - start_time) / 60

    print("runtime =", runtime, "minutes for", len(n2_targ[30:60]), "targets.")
