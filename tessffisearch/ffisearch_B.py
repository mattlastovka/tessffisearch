import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from transitleastsquares import transitleastsquares
from astropy.stats import sigma_clip
from tessphomo import TESSTargetPixelModeler
from tessphomo.mast import retrieve_tess_ffi_cutout_from_mast, get_tic_sources
import numbers
from wotan import flatten
import os
from astropy.table import QTable
import re
import io
import sqlite3
import warnings

#Plot settings
myplot_specs = {
 'axes.linewidth':  1.5, 
 'xtick.top' : True,         
 'ytick.right' :  True,
 'xtick.direction' : 'in',    
 'ytick.direction' : 'in', 
 'xtick.major.size' : 10,     
 'ytick.major.size' : 10,
 'xtick.minor.size' : 5,    
 'ytick.minor.size' : 5,      
 'font.size' : 15,  
 'font.family' : 'serif',
 'figure.figsize' : [5.5, 5.5], 
 'lines.linewidth' : 2.5      
}
plt.rcParams.update(myplot_specs)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

def write_lc_to_fits_file(data, lc, lc_save_direc='tessphomo_lightcurves/', overwrite=True):
    """
    Function adapted from Robby's example notebook that saves the light curves as .fits files. 
    """

    targetid = data.tic_id[4:].zfill(12)
    sector = str(data.tpf.sector).zfill(4)
    tpfsize_0, tpfsize_1 = data.tpf.shape[1:]

    fname = lc_save_direc + 'tessphomo_ffi_lightcurve_sector_'+sector+'_tic_'+targetid+'_cutout_{}'.format(tpfsize_0)+'x{}'.format(tpfsize_1)+'.fits'
  
    lc.write(fname, format='fits', overwrite=overwrite)  
    
    return fname

def plot_lightcurve_and_systematics(lc, sector, ticid, **kwargs):
    """
    Function adapted from Robby's example notebook to plot the light curves from TESSphomo.
    Can compare "CAP" and "PRF" methods, see raw to calibrated transition, and see zp scale,
    background flux, and row and column offset.
    """

    fig, axes = plt.subplots(5,1, figsize=(10,10))

    axes[0].plot(lc['time'].value, lc['raw_cap_flux'].value, label='CAP', **kwargs)
    axes[0].plot(lc['time'].value, lc['raw_prf_flux'].value, label='PRF', **kwargs)
    
    axes[1].plot(lc['time'].value, lc['cal_cap_flux'], label='CAP', **kwargs)
    axes[1].plot(lc['time'].value, lc['cal_prf_flux'], label='PRF', **kwargs)

    axes[2].plot(lc['time'].value, lc['row_offset'], label='row', **kwargs)
    axes[2].plot(lc['time'].value, lc['col_offset'], label='col', **kwargs)
    
    axes[3].plot(lc['time'].value, lc['zp_flux_scale'], label='Zp Scale', **kwargs)

    axes[4].plot(lc['time'].value, lc['bkg_sapflux'], label='Bkg Flux', **kwargs)

    axes[0].set_ylabel('Raw Flux')
    axes[1].set_ylabel('Cal. Flux')
    axes[2].set_ylabel('pointing\noffset')
    axes[3].set_ylabel('ZP Scale')
    axes[4].set_ylabel('Bkg Flux')

    axes[-1].set_xlabel('Time [BTJD]')

    for ax in axes:
        ax.legend()

    plt.tight_layout()

def determine_best_flux(light_curves):
    """
    This function determines whether to use the CAP or PRF light curves. Calculates the standard deviation
    of each light curve, then counts how many light curves have PRF better or CAP better. If the counts for
    PRF and CAP are the same, then calculate the sum of all the standard deviations and use the one with
    the smaller sum.
    """
    cap_tal = 0
    prf_tal = 0
    for i in range(len(light_curves)):
        #Determine the standard deviation of both light curves
        cap_std = np.std(light_curves[i]['cal_cap_flux'])
        prf_std = np.std(light_curves[i]['cal_prf_flux'])
        if cap_std > prf_std:
            prf_tal += 1
        elif cap_std < prf_std:
            cap_tal += 1
    cap_d = sum([np.std(light_curves[i]['cal_cap_flux']) for i in range(len(light_curves))])
    prf_d = sum([np.std(light_curves[i]['cal_prf_flux']) for i in range(len(light_curves))])
    if (cap_d/prf_d) > 1e4:
        flux_id = 'cal_prf_flux'
    else:
        if cap_tal > prf_tal:
            #print("Use cap")
            flux_id = 'cal_cap_flux'
        elif prf_tal > cap_tal:
            #print("Use prf")
            flux_id = 'cal_prf_flux'
        else:
            if cap_d < prf_d:
                #print("Use cap")
                flux_id = 'cal_cap_flux'
            else:
                #print("Use prf")
                flux_id = 'cal_prf_flux'
    return flux_id

def count(list1, l, r):
    """
    Count the number of elements in list1 that are between l and r (lower and upper bounds)
    """
    return len(list(x for x in list1 if l <= x <= r))

def flagging_criteria(all_results, sde_thresh = 6, save_direc='./transit_search/', sec_thresh=2):
    """
    This is the code for flagging whether a transit-like signal is detected.

    For each light curve, check that the FAP is low enough, the SDE is above the threshhold, and 
    there are at least 2 transits seen.

    Then flag if there are the required number of light curves that have similar periods
    """
    new_results = []
    flag = False
    for i in all_results:
        if i.FAP < 1e-2 and (i.SDE > sde_thresh):
            if i.transit_count >= 2:
                new_results.append(i)
    new_ps = [i.period for i in new_results]
    secs = []
    for i in new_results:
        upper = i.period + 2*i.period_uncertainty
        lower = i.period - 2*i.period_uncertainty
        sim_count = count(new_ps, lower, upper)
        if sim_count > (sec_thresh-1):
            flag = True
            secs.append(sim_count)
    if flag is True:
        #print("Potential transit detected")
        sec_num = max(secs)
    else:
        #print("No transit found")
        sec_num = 0
    return flag, sec_num

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

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

def save_results_file_deprecated(results, params, ticid, sector, save_direc):
    """
    This function saves the results of the transit search to a file. This function can save 3 different,
    files, the statistics, the periodogram, and the phase-folded light curve. Based on the current iteration,
    the periodogram and phase-folded light curve functions are commented out. Much of this function is 
    made to format the statistics in such a way that makes it easy for pandas to save it as a csv. To do this, 
    we need every row/column of the array to be the same length. So, we will add "None" to fill out the array
    to match the length of the largest element. 

    Inputs:
    -----------
    file_name: (str)
        The name of the file to save the results to. Should include the TIC-ID and the sector number.

    results: (tls results object)
        The results object that is the output of the (TLS model).power function

    params: (array-like)
        An array of the additional params to add to the results file. Should be in the following order:
            [flag, sigma_upper, sigma_lower, window_length, method, flux_id]
            flag: (Boolean)
                If True, potential transit detected

            sigma_upper, sigma_lower: (number)
                The upper and lower significance bounds used for sigma-clipping of the light curve

            window_length: (number)
                Window length parameter as input for wotan detrending

            method: (str)
                Method used for detrending (see wotan documentation for options)

            flux_id: (str)
                Method used to calculate flux. Can be determined by the "determine_best_flux" function.
                Should be either "cal_cap_flux" or "cal_prf_flux"
    """
    #Determine the keys for the TLS results
    #keys = [key for key in results.keys()]
    #Determine the size of each element in the results object, so that we know the correct number of
    #"None"s to add.
    el_lengths = []
    for i in range(len(dict(list(results.items())[:28]))):
        res_line = list(dict(list(results.items())[i:(i+1)]).values())[0]
        if isinstance(res_line, numbers.Number):
            el_lengths.append(1)
        else:
            el_lengths.append(len(res_line))
    #print(el_lengths)
    full_results = []
    e2 = max(el_lengths)-1
    #key_list = ['flag', 'sigma_upper', 'sigma_lower', 'window_length', 'method', 'flux_id']
    #key_list.extend(keys)
    #Go through each of the extra parameters to add at the beginning of each file. Extend each list
    #to match the length of the largest element
    fl = [params[0]]
    fl.extend([None for i in range(e2)])
    sig_up = [params[1]]
    sig_up.extend([None for i in range(e2)])
    sig_lo = [params[2]]
    sig_lo.extend([None for i in range(e2)])
    win_len = [params[3]]
    win_len.extend([None for i in range(e2)])
    meth = [params[4]]
    meth.extend([None for i in range(e2)])
    fl_id = [params[5]]
    fl_id.extend([None for i in range(e2)])
    full_results.append(fl)
    full_results.append(sig_up)
    full_results.append(sig_lo)
    full_results.append(win_len)
    full_results.append(meth)
    full_results.append(fl_id)
    #Now, loop through the first 29 elements of the results file, and repeat the process of adding
    #Nones to each element, then add to the array
    for i in range(len(dict(list(results.items())[:28]))): 
        res_line = list(dict(list(results.items())[i:(i+1)]).values())[0]
        if isinstance(res_line, numbers.Number):
            end = max(el_lengths) - 1  
        else:
            end = max(el_lengths) - len(res_line)  
        if end == 0:
            full_results.append(list(res_line))
        else:
            if isinstance(res_line, numbers.Number):
                thelis = [res_line] 
            else:
                thelis = list(res_line)
            thelis.extend([None for i in range(end)])
            full_results.append(list(thelis))
    #Reorder the array so that each element corresponds to a different column in the dataframe
    reordered_array = np.column_stack(full_results)
    #Add the results to the database file
    file_name = save_direc + "transit_search_results.db"
    con = sqlite3.connect(file_name, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("insert into results (ticid, sector, arr) values (?, ?, ?)", (ticid, sector, reordered_array, ))
    con.commit()
    con.close()
    return 0

def save_results_file(results, params, ticid, sector, save_direc):
    """
    This function saves the results of the transit search to a file.

    Inputs:
    -----------
    file_name: (str)
        The name of the file to save the results to. Should include the TIC-ID and the sector number.

    results: (tls results object)
        The results object that is the output of the (TLS model).power function

    params: (array-like)
        An array of the additional params to add to the results file. Should be in the following order:
            [flag, sigma_upper, sigma_lower, window_length, method, flux_id]
            flag: (Boolean)
                If True, potential transit detected

            sigma_upper: (number)
                The upper significance bound used for sigma-clipping of the light curve

            window_length: (number)
                Window length parameter as input for wotan detrending

            method: (str)
                Method used for detrending (see wotan documentation for options)

            flux_id: (str)
                Method used to calculate flux. Can be determined by the "determine_best_flux" function.
                Should be either "cal_cap_flux" or "cal_prf_flux"
    """
    save_results_dict = {
        "flag": params[0],
        "SDE": results.SDE,
        "SDE_raw": results.SDE_raw,
        "period": results.period,
        "period_uncertainty": results.period_uncertainty,
        "T0": results.T0,
        "duration": results.duration,
        "depth": results.depth,
        "FAP": results.FAP,
        "transit_count": results.transit_count,
        "sec_num": params[1],
        "sigma_upper": params[2],
        "window_length": params[3],
        "method": params[4],
        "flux_id": params[5]
        }
    #Reorder the array so that each element corresponds to a different column in the dataframe
    reordered_array = np.asarray(list(save_results_dict.values()))
    #Add the results to the database file
    file_name = save_direc + "transit_search_results.db"
    sector_string = str(sector[0]) + "-" + str(sector[-1])
    con = sqlite3.connect(file_name, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("insert into results (ticid, sector, arr) values (?, ?, ?)", (ticid, sector_string, reordered_array, ))
    con.commit()
    con.close()
    return 0

def search_for_transit(time, flux, mass, radius, num_threads, period_max=12.):
    """
    This function searches for a transit using transitleastsquares. Use the star masses from the 
    TIC to make the performance better.
    """
    model = transitleastsquares(time, flux)
    with warnings.catch_warnings(action="ignore"):
        results = model.power(use_threads=num_threads, M_star=mass, M_star_min=0.8*mass, 
                                M_star_max=1.2*mass, R_star=radius, R_star_min=0.8*radius, R_star_max=1.2*radius,
                                show_progress_bar = False, verbose=False, period_max = period_max)
    return results

def flatten_lightcurve(time, flux, sigma_upper, window_length, method, sigma_lower=np.inf):
    """
    This function applies sigma clipping (to remove outlier data points) and uses wotan to flatten
    the light curve
    """
    flatten_lc, trend_lc = flatten(time, flux, window_length=window_length, 
                               return_trend=True, method=method, break_tolerance=0.1)
                               
    clipped_flux = sigma_clip(flatten_lc, sigma_upper=sigma_upper, sigma_lower=sigma_lower, masked=True)
    mask = clipped_flux.mask
    return time[~mask], flatten_lc[~mask], mask

def make_light_curves(ticid, lc_save_direc, logger, save_direc, cutout_size=(25,25)):
    tries = 3
    for i in range(tries):
        try:
            all_tpfs = retrieve_tess_ffi_cutout_from_mast(ticid=ticid, cutout_size=cutout_size, sector=None)
        except BaseException as e:
            if i < tries - 1: # i is zero indexed
                continue
            else:
                except_file = save_direc + 'failed_tic.txt'
                file1 = open(except_file, "a")  # append mode
                file1.write(str(ticid) + '\n')
                file1.close()
                logger.exception(e)
        break
    all_tpfs = retrieve_tess_ffi_cutout_from_mast(ticid=ticid, cutout_size=cutout_size, sector=None)
    print(len(all_tpfs), "tpfs")
    input_catalog = get_tic_sources(ticid, tpf_shape=all_tpfs[0].shape[1:])
    light_curves = []
    sectors = []
    for tpf in all_tpfs:
        try:
            TargetData = TESSTargetPixelModeler(tpf, input_catalog=input_catalog)
            lc = TargetData.get_corrected_LightCurve(assume_catalog_mag=True)
            with warnings.catch_warnings(action="ignore"):
                write_lc_to_fits_file(TargetData, lc, lc_save_direc=lc_save_direc, overwrite=True)
            sectors.append(str(TargetData.tpf.sector).zfill(4))
            light_curves.append(lc)
        except BaseException as e:
            logger.exception(e)
    return light_curves, sectors

def retrieve_or_make_lc(ticid, lc_save_direc, logger, save_direc):
    dir_list = np.asarray(os.listdir(lc_save_direc))
    mask = np.asarray([(str(ticid).zfill(12) in i) for i in dir_list])
    masked_dir_list = dir_list[mask]
    if len(masked_dir_list) == 0:
        logger.info("making light curves")
        light_curves, sectors = make_light_curves(ticid, lc_save_direc=lc_save_direc, logger=logger, save_direc=save_direc)
    else:
        logger.info("light curves exist")
        light_curves = []
        sectors = []
        for i in masked_dir_list:
            lightc = QTable.read((lc_save_direc + i), format='fits', astropy_native=True)
            lightc['time'] = lightc['time'].btjd
            light_curves.append(lightc)
            sectors.append(re.findall('sector_00(.*)_tic', i)[0])
    return light_curves, sectors

