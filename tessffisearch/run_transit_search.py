import ffisearch as ff
import time
from astropy.table import QTable, vstack
import logging
import lightkurve as lk
import os
import glob
import pandas as pd
import numpy as np
import re
from transitleastsquares import transit_mask
import common as cmn

def run_the_search(ticid, mass, radius, save_direc, logger, sigma_upper=4., window_length=0.8, 
                    method='biweight', sde_thresh=6, sec_thresh=2, num_threads=1, clear_cache=True):
    logger.info("getting light curves")
    #Define directory where light curves are
    light_curve_direc = cmn.light_curve_direc + str(ticid)[:-5].zfill(5) + 'XXXXX/lc-' + str(ticid).zfill(10) + "/"
    dir_list = np.asarray(os.listdir(light_curve_direc))
    #get file names for the given TIC ID
    #Open the light curve files and 
    light_curves2 = []
    sectors = []
    full_2d_list = []
    for i in dir_list:
        #Open the light curve using astropy
        data = pd.read_parquet(light_curve_direc+i)
        data_as = QTable.from_pandas(data)
        data_as['time'] = data_as['time'].jd - 2457000
        light_curves2.append(data_as)
        #Determine the sector using the light curve file name
        sector = re.findall('-s(.*)-', i)[0][:2]
        sectors.append(int(sector))
        full_2d_list.append([data_as, int(sector)])
    #use the light curves to determine CAP or PRF
    flux_id = ff.determine_best_flux(light_curves2)
    logger.info("Detrending and clipping light curves")
    for i in range(len(full_2d_list)):
        light_curve = full_2d_list[i][0]
        time = light_curve['time'].value
        flux = light_curve[flux_id].value
        times, flat, mask = ff.flatten_lightcurve(time, flux, sigma_upper, window_length, method)
        full_2d_list[i][0] = full_2d_list[i][0][~mask]
        full_2d_list[i][0]['flat_flux'] = flat
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

    logger.info("running transit search")
    all_results = []
    all_consec_sectors= []
    for i in range(len(light_curves)):
        #run the transit search, add all the results to a list
        search_time = light_curves[i]['time'].value
        search_flux = light_curves[i]['flat_flux'].value
        results = ff.search_for_transit(search_time, search_flux, mass=mass, radius=radius, num_threads=num_threads)
        all_consec_sectors.append(consecutive_sectors[i])
        all_results.append(results)
        SDE = results.SDE
        test_number = 1
        while SDE > 7:
            in_transit_mask = transit_mask(search_time, results.period, results.duration, results.T0)
            search_time = search_time[~in_transit_mask]
            search_flux = search_flux[~in_transit_mask]
            results = ff.search_for_transit(search_time, search_flux, mass=mass, radius=radius, num_threads=num_threads)
            SDE = results.SDE
            if SDE > 0:
                all_consec_sectors.append(consecutive_sectors[i])
                all_results.append(results)
            if test_number >= 10:
                break
            test_number += 1  
    logger.info("Determine Flag")
    #Set up so that flagged in either 2 or 20% of light curves (whichever is more)
    if 0.2*len(light_curves) > sec_thresh:
        sec_thresh1 = int(0.2*len(all_results))
    else:
        sec_thresh1 = sec_thresh
    flag, sec_num = ff.flagging_criteria(all_results, sde_thresh = sde_thresh, sec_thresh=sec_thresh1, save_direc=save_direc)
    if flag is True:
        logger.info("potential transit detected")
        flag_file = save_direc + 'flagged_tic.txt'
        file1 = open(flag_file, "a")  # append mode
        file1.write(str(ticid) + ' ' + str(sec_num) + '\n')
        file1.close()
    else:
        logger.info("no transit found")
    logger.info("saving transit search results")
    params = [flag, sec_num, sigma_upper, window_length, method, flux_id]
    for i in range(len(all_results)):
        ff.save_results_file(all_results[i], params, ticid, all_consec_sectors[i], save_direc)

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

    target_list = pd.read_csv("targets.csv")

    transit_search_direc = cmn.transit_search_direc

    for ticid in target_list['ID']:
        #print(ticid)
        tic_index = np.where(target_list['ID'] == ticid)[0][0]
        #print(tic_index)
        logfile = transit_search_direc + "logfiles/TIC" + str(ticid) + ".log"
        logging.basicConfig(filename=logfile, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("TIC "+ str(ticid))
        #logger.info("TESSmag " + str(target_list['TESS mag'][tic_index]))
        mass = target_list['mass'][tic_index]
        radius = target_list['rad'][tic_index]                    
        run_the_search(ticid, mass, radius, transit_search_direc, logger, clear_cache=True)
        finish_file = transit_search_direc + 'finished_runs.txt'
        finfile = open(finish_file, "a")  # append mode
        finfile.write(str(tic_index) + " " + str(ticid) + '\n')
        finfile.close()

    end_time = time.time()

    runtime = (end_time - start_time) / 60

    print("runtime =", runtime, "minutes")
