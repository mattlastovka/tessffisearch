import ffisearch_B as ff
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
import common as cmn
from transitleastsquares import transit_mask
import sys

def run_the_search(lc_file, sector, ticid, mass, radius, save_direc, logger, sigma_upper=4., window_length=0.8, 
                    method='biweight', sde_thresh=6, sec_thresh=2, num_threads=1, clear_cache=True):
	logger.info("getting light curves")
	#Open the light curve using astropy
	data = pd.read_parquet(lc_file)
	light_curves = QTable.from_pandas(data)
	light_curves['time'] = light_curves['time'].jd - 2457000
	#use the light curves to determine CAP or PRF
	#flux_id = ff.determine_best_flux(light_curves)
	flux_id = 'cal_cap_flux'

	logger.info("Detrending and clipping light curves")
	#Run all the light curves through wotan detrending. The flattened light curve is "flat"
	time = light_curves['time'].value
	flux = light_curves[flux_id].value
	cadence_no2 = light_curves['cadenceno'].value
	times, flat, mask = ff.flatten_lightcurve(time, flux, sigma_upper, window_length, method)
	cadence_no = cadence_no2[~mask]
	logger.info("running transit search")
	#run the transit search, add all the results to a list
	#print(mass)
	#print(radius)
	all_results = ff.search_for_transit(times, flat, mass=mass, radius=radius, num_threads=num_threads)
	logger.info("Determine Flag")
	#print(all_results)
	in_transit_mask = transit_mask(times, all_results.period, all_results.duration, all_results.T0)
	#print(in_transit_mask)
	in_transit_cadence = cadence_no[in_transit_mask]
	#print(in_transit_cadence)
	in_transit_times = times[in_transit_mask]
	sector_filename = "s" + str(sector) + "_transit_cadence.txt"
	sector_filename2 = "s" + str(sector) + "_transit_times.txt"
	with open(sector_filename, 'a') as f:
		for k in in_transit_cadence:
			f.write(str(k) + " ")
		f.write("\n")
	with open(sector_filename2, 'a') as f:
		for k in in_transit_times:
			f.write(str(k) + " ")
		f.write("\n")
	return 0

if __name__ == "__main__":
	start_time = time.time()
	
	#begin = int(sys.argv[1])
	#end = int(sys.argv[2])

	sector = 51
	tic_list = np.genfromtxt("s51_tics_w_lc.txt")
	target_list = pd.read_csv("targets.csv")

	transit_search_direc = cmn.transit_search_direc
	
	for i in range(len(target_tom[:2000])):
		if target_tom['Sector'][i] == sector:
			ticid = target_tom['ID'][i]
			camera = target_tom['Camera'][i]
			ccd = target_tom['CCD'][i]
			lc_filname = cmn.light_curve_direc + str(ticid)[:-5].zfill(5) + 'XXXXX/lc-' + str(ticid).zfill(10) + "/lc-" + str(ticid) + "-s" + str(sector) + "-" + str(camera) + "-" + str(ccd) + ".parquet.gzip"
			if os.exists(lc_filname):
				#print(ticid)
				tic_index = np.where(target_list['ID'] == int(ticid))[0][0]
				#print(tic_index)
				logfile = transit_search_direc + "logfiles/TIC" + str(ticid) + ".log"
				logging.basicConfig(filename=logfile, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
				logger = logging.getLogger(__name__)
				logger.info("TIC "+ str(int(ticid)))
				#logger.info("TESSmag " + str(target_list['TESS mag'][tic_index]))
				mass = target_list['mass'][tic_index]
				radius = target_list['rad'][tic_index]                    
				run_the_search(lc_filname, sector, int(ticid), mass, radius, transit_search_direc, logger, clear_cache=True)

	end_time = time.time()

	runtime = (end_time - start_time) / 60

	print("runtime =", runtime, "minutes")
