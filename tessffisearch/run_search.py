import ffisearch as ff
import time
import pandas as pd
import pathlib
import sys
import common as cmn

def run_the_search(ticid, mass, radius, save_direc, sigma_upper=4., sigma_lower=12., window_length=0.8, 
                    method='biweight', sde_thresh=6, sec_thresh=2, num_threads=4):
    light_curves, sectors = ff.retrieve_or_make_lc(ticid, lc_save_direc=(save_direc + '/tessphomo_lightcurves/'))
    print("Finished")
    flux_id = ff.determine_best_flux(light_curves)
    print("Detrending and clipping light curves")
    all_flat = []
    all_trend = []
    all_times = []
    for i in range(len(light_curves)):
        time = light_curves[i]['time'].value
        flux = light_curves[i][flux_id].value
        times, flat, trend = ff.flatten_lightcurve(time, flux, sigma_upper, sigma_lower, window_length, method)
        all_flat.append(flat)
        all_trend.append(trend)
        all_times.append(times)
    print("running transit search")
    all_results = []
    for i in range(len(light_curves)):
        results = ff.search_for_transit(all_times[i], all_flat[i], mass, radius, num_threads)
        all_results.append(results)
    print("Determine Flag")
    flag, sec_num = ff.flagging_criteria(all_results, sde_thresh = sde_thresh, sec_thresh=sec_thresh, save_direc=save_direc)
    if flag is True:
        flag_file = save_direc + 'flagged_tic.txt'
        file1 = open(flag_file, "a")  # append mode
        file1.write(str(ticid) + ' ' + str(sec_num) + '\n')
        file1.close()
    print("saving transit search results")
    params = [flag, sec_num, sigma_upper, sigma_lower, window_length, method, flux_id]
    for i in range(len(all_results)):
        ff.save_results_file(all_results[i], params, ticid, sectors[i])
    return 0

if __name__ == "__main__":
    start_time = time.time()

    begin = int(sys.argv[1])
    end = int(sys.argv[2])

    target_list = pd.read_csv("targets.csv")

    transit_search_direc = cmn.transit_search_direc

    for tic_index in range(begin, end):
        ticid = int(target_list['ID'][tic_index])
        print("TIC", ticid)
        print("TESSmag", target_list['Tmag'][tic_index])
        mass = target_list['mass'][tic_index]
        radius = target_list['rad']
        run_the_search(ticid, mass, radius, transit_search_direc)
        finish_file = transit_search_direc + 'finished_runs.txt'
        finfile = open(finish_file, "a")  # append mode
        finfile.write(str(tic_index) + " " + str(ticid) + '\n')
        finfile.close()

    end_time = time.time()

    runtime = (end_time - start_time) / 60

    print("runtime =", runtime, "minutes for", len(range(begin, end)), "targets.")
