import ffisearch_knowntoi as ff
import time
import pathlib
from astropy.table import QTable, vstack
import logging
import lightkurve as lk
import os
import glob
import sqlite3
import pandas as pd
import numpy as np
import io
import re

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)

def run_the_search(ticid, mass, radius, save_direc, logger, sigma_upper=4., sigma_lower=12., window_length=0.8, 
                    method='biweight', sde_thresh=6, sec_thresh=2, num_threads=1, clear_cache=True):
    logger.info("getting light curves")
    #Define directory where light curves are
    light_curve_direc = (save_direc + 'tessphomo_lightcurves/')
    dir_list = np.asarray(os.listdir(light_curve_direc))
    #get file names for the given TIC ID
    mask = np.asarray([(str(ticid).zfill(12) in i) for i in dir_list])
    masked_dir_list = dir_list[mask]
    #Open the light curve files and 
    light_curves2 = []
    sectors = []
    full_2d_list = []
    for i in masked_dir_list:
        #Open the light curve using astropy
        read_lc = QTable.read((light_curve_direc + i), format='fits', astropy_native=True)
        #Correct the time column to TESS times (BTJD)
        read_lc['time'] = lk.LightCurve(read_lc).time.btjd
        light_curves2.append(read_lc)
        #Determine the sector using the light curve file name
        sector = re.findall('sector_00(.*)_tic', i)[0]
        sectors.append(int(sector))
        full_2d_list.append([read_lc, int(sector)])
    #use the light curves to determine CAP or PRF
    flux_id = ff.determine_best_flux(light_curves2)
    #Sort sectors in numerical order to determnine if consecutive
    sorted_sectors = sorted(sectors)
    consecutive_sectors = []
    sec_group = [sorted_sectors[0]]
    for i in range(1, len(sorted_sectors)):
        if sorted_sectors[i] == (sorted_sectors[i-1] + 1):
            sec_group.append(sorted_sectors[i])
        else:
            consecutive_sectors.append(sec_group)
            sec_group = [sorted_sectors[i]]
    consecutive_sectors.append(sec_group)
    light_curves = []
    for i in consecutive_sectors:
        #stack the light curves using astropy vstack
        stack = vstack([full_2d_list[j][0] for j in range(len(full_2d_list)) if full_2d_list[j][1] in i])
        #Resort the new light curve so that it is in sequential order
        stack.sort('time')
        light_curves.append(stack)

    logger.info("Detrending and clipping light curves")
    all_flat = []
    all_times = []
    for i in range(len(light_curves)):
        #Run all the light curves through wotan detrending. The flattened light curve is "flat"
        time = light_curves[i]['time'].value
        flux = light_curves[i][flux_id].value
        times, flat, trend = ff.flatten_lightcurve(time, flux, sigma_upper, sigma_lower, window_length, method)
        all_flat.append(flat)
        all_times.append(times)
    logger.info("running transit search")
    all_results = []
    for i in range(len(light_curves)):
        #run the transit search, add all the results to a list
        results = ff.search_for_transit(all_times[i], all_flat[i], mass=mass, radius=radius, num_threads=num_threads)
        all_results.append(results)
    logger.info("Determine Flag")
    #Set up so that flagged in either 2 or 20% of light curves (whichever is more)
    if 0.2*len(all_results) > sec_thresh:
        sec_thresh1 = int(0.2*len(all_results))
    else:
        sec_thresh1 = sec_thresh
    flag, sec_num = ff.flagging_criteria(all_results, sde_thresh = sde_thresh, sec_thresh=sec_thresh1, save_direc=save_direc)
    if flag is True:
        logger.info("potential transit detected")
        flag_file = save_direc + 'toi_flagged_tic_multi.txt'
        file1 = open(flag_file, "a")  # append mode
        file1.write(str(ticid) + ' ' + str(sec_num) + '\n')
        file1.close()
    else:
        logger.info("no transit found")
    logger.info("saving transit search results")
    params = [flag, sec_num, sigma_upper, sigma_lower, window_length, method, flux_id]
    for i in range(len(all_results)):
        ff.save_results_file(all_results[i], params, ticid, consecutive_sectors[i], save_direc)

	#clear lightkurve tesscut cache
    if clear_cache is True:
        tesscut_cache_dir = lk.config.get_cache_dir() + "/tesscut/*"
        files = glob.glob(tesscut_cache_dir)
        for f in files:
            os.remove(f)
        logger.info("lightkurve cache cleared")
    return 0

if __name__ == "__main__":
    start_time = time.time()
    
    unflagged = pd.read_csv("unflagged_tics.csv")
    unflagged_tics = np.array(unflagged["TIC ID"])

    target_list = pd.read_csv("ultrasat_tois.csv")

    transit_search_direc = './transit_search/'

    for ticid in unflagged_tics:
        #print(ticid)
        tic_index = np.where(target_list['TIC ID'] == ticid)[0][0]
        #print(tic_index)
        logfile = transit_search_direc + "logfiles/TIC" + str(ticid) + ".log"
        logging.basicConfig(filename=logfile, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("TIC "+ str(ticid))
        #logger.info("TESSmag " + str(target_list['TESS mag'][tic_index]))
        mass = target_list['Stellar Mass (M_Sun)'][tic_index]
        radius = target_list['Stellar Radius (R_Sun)'][tic_index]                    
        run_the_search(ticid, mass, radius, transit_search_direc, logger, clear_cache=True)
        finish_file = transit_search_direc + 'toi_multisector_finished_runs.txt'
        finfile = open(finish_file, "a")  # append mode
        finfile.write(str(tic_index) + " " + str(ticid) + '\n')
        finfile.close()

    end_time = time.time()

    runtime = (end_time - start_time) / 60

    print("runtime =", runtime, "minutes")
