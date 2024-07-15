#import packages
import matplotlib.pyplot as plt
import lightkurve as lk
import numpy as np
import pandas as pd
from tessphomo import TESSTargetPixelModeler
from tessphomo.mast import retrieve_tess_ffi_cutout_from_mast, get_tic_sources
from wotan import flatten
from transitleastsquares import (transitleastsquares,cleaned_array,catalog_info,transit_mask)
import os.path
from astropy.table import vstack, QTable
from astropy.stats import sigma_clip
import pathlib
import numbers

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

    #plt.savefig("TIC_"+str(ticid)+'/tessphomo_ffi_lightcurve_sector_'+sector+'_tic_'+str(ticid)+'_plot.jpg')


def make_lc_and_plots(ticid, window_length=1.1, flat_method = 'biweight', sigma_upper = 4., sigma_lower=12., lc_save_direc='tessphomo_lightcurves/'):
    """
    Function that calculates the light curves from FFIs for a given TIC ID, then detrends them, then
    conducts a transit search. 
    """
    #Get the light curves for the target in all available sectors
    light_curves = []
    sectors = []
    #See if there are light curves already made for this target
    for i in range(85):
        finame = lc_save_direc + 'tessphomo_ffi_lightcurve_sector_'+str(i).zfill(4)+'_tic_'+str(ticid).zfill(12)+'_cutout_25x25.fits'
        if os.path.exists(finame):
            #If light curves already exist, read them in here. Since astropy saves Time objects in a weird way to fits, this
            #is how the light curve file needs to be read in
            lightc = QTable.read(finame, format='fits', astropy_native=True)
            lightc['time'] = lightc['time'].btjd
            light_curves.append(lightc)
            sectors.append(str(i).zfill(4))
    #If there are no light curves available, then get tpfs and make light curves
    if len(light_curves) == 0:
        new_dir = pathlib.Path('./', "TIC_"+str(ticid))
        new_dir.mkdir(parents=True, exist_ok=True)
        #retrieve the ffi cutouts from mast
        print("retrieving tpfs")
        all_tpfs = retrieve_tess_ffi_cutout_from_mast(ticid=ticid, cutout_size=(25,25), sector=None)
        print(len(all_tpfs), "tpfs")
        #Determine the background stars in the area
        print("retrieving background stars")
        input_catalog = get_tic_sources(ticid, tpf_shape=all_tpfs[0].shape[1:])
        print(input_catalog.shape[0], "background stars")
        print("making light curves")
        
        #Loop over each sector
        for tpf in all_tpfs:
            try:
                #Make the light curves, and save them to .fits files
                TargetData = TESSTargetPixelModeler(tpf, input_catalog=input_catalog)
                lc = TargetData.get_corrected_LightCurve(assume_catalog_mag=True)
                write_lc_to_fits_file(TargetData, lc, lc_save_direc='./example_data/', overwrite=True)
                sectors.append(str(TargetData.tpf.sector).zfill(4))
                light_curves.append(lc)
            except:
                print("ERROR!!!!!")
    #Plot the light curves
    print("Finished. Now plotting")
    for i in range(len(light_curves)):
        plot_lightcurve_and_systematics(light_curves[i], sectors[i], ticid)

    #Determine whether to use "cap" or "prf" flux for the light curves. Do this by finding the standard deviation
    #of each calibrated light curve using each method, then determine which one has a smaller std. After determining
    #this for each sector, choose the method that has the smaller std for most of the sectors/light curves
    cap_tal = 0
    prf_tal = 0
    for i in range(len(light_curves)):
        cap_std = np.std(light_curves[i]['cal_cap_flux'].value)
        prf_std = np.std(light_curves[i]['cal_prf_flux'].value)
        if cap_std > prf_std:
            prf_tal += 1
        elif cap_std < prf_std:
            cap_tal += 1
    if cap_tal > prf_tal:
        print("Use cap")
        flux_id = 'cal_cap_flux'
    elif prf_tal > cap_tal:
        print("Use prf")
        flux_id = 'cal_prf_flux'
    else:
        #If the same number of sectors have lower "cap" and "prf", then find the sum of the std over every sector, 
        #then use the one that has the smaller sum
        cap_d = sum([np.std(light_curves[i]['cal_cap_flux'].value) for i in range(len(light_curves))])
        prf_d = sum([np.std(light_curves[i]['cal_prf_flux'].value) for i in range(len(light_curves))])
        if cap_d < prf_d:
            print("Use cap")
            flux_id = 'cal_cap_flux'
        else:
            print("Use prf")
            flux_id = 'cal_prf_flux'

    print("Detrending and clipping light curves")
    all_flat = []
    all_trend = []
    all_mask = []
    all_times = []
    for i in range(len(light_curves)):
        clipped_flux = sigma_clip(light_curves[i][flux_id].value, sigma_upper=4., sigma_lower=12., masked=True)
        mask = clipped_flux.mask
        times = light_curves[i]['time'].value
        flatten_lc, trend_lc = flatten(times[~mask], clipped_flux.data[~mask], window_length=1.1, 
                                   return_trend=True, method='biweight', break_tolerance=0.1)
        all_flat.append(flatten_lc)
        all_trend.append(trend_lc)
        all_mask.append(mask)
        all_times.append(times[~mask])

    for i in range(len(light_curves)):
        fig, ax = plt.subplots(2,1, figsize=(10, 5), sharex=True, tight_layout=True)
        ax[0].scatter(light_curves[i]['time'].value, light_curves[i][flux_id].value, s=1, color='black')
        ax[0].plot(all_times[i], all_trend[i], linewidth=2, color='red')
        ax[0].set_xlabel('Time (days)')
        ax[0].set_ylabel('Calibrated Flux')
        ax[1].scatter(all_times[i], all_flat[i], s=1, color='black')
        ax[1].set_xlabel('Time (days)')
        ax[1].set_ylabel('Detrended flux')
        #plt.savefig('tic_'+str(ticid)+'_sector_'+sectors[i]+'_detrend_plot_poster.jpg')
        plt.show()

    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=ticid)
    print("running transit search")
    all_results = []
    for i in range(len(light_curves)):
        model = transitleastsquares(all_times[i], all_flat[i])
        results = model.power(u=ab)
        all_results.append(results)

    for i in range(len(all_results)):
        results = all_results[i]
        fig, ax = plt.subplots(1,2, figsize=(10,5), tight_layout=True)
        
        ax[0].axvline(results.period, alpha=0.4, lw=3)
        ax[0].set_xlim(np.min(results.periods), np.max(results.periods))
        for n in range(2, 10):
            ax[0].axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
            ax[0].axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
        ax[0].set_ylabel(r'SDE')
        ax[0].set_xlabel('Period (days)')
        ax[0].plot(results.periods, results.power, color='black', lw=0.5)
        ax[0].set_xlim(0, max(results.periods));
        ax[1].plot(results.model_folded_phase, results.model_folded_model, color='red')
        ax[1].scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
        ax[1].set_xlim(0.4, 0.6)
        ax[1].set_xlabel('Phase')
        ax[1].set_ylabel('Relative flux')
        #plt.savefig('tic_'+str(ticid)+'_sector_'+sectors[i]+'_tls_plot_poster.jpg')
        plt.show()

    print("saving transit search results")
    list4 = []
    for key in all_results[0].keys():
        list4.append(key)

    #with open("TIC_"+str(ticid)+"/tic_"+str(ticid)+"_tls_results.txt", "w") as file1:
    #    # Writing data to a file
    #    for i in range(len(light_curves)):
    #        file1.write("sector = "+str(sectors[i])+"\n")
    #        for j in list4:
    #            file1.write(j+" : "+str(all_results[i][j])+"\n")
    #        file1.write("-------------------- \n")
    print("Completed :)")

    return all_results, all_times, all_flat, all_trend

def save_results_file(file_name, results, params):
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
    keys = [key for key in results.keys()]
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
    key_list = ['flag', 'sigma_upper', 'sigma_lower', 'window_length', 'method', 'flux_id']
    key_list.extend(keys)
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
    #Add elements and column headers to a pandas dataframe, then save as a csv
    df = pd.DataFrame(reordered_array)
    df.columns = key_list[:34]
    df.to_csv(file_name + 'stats.csv')

    #The following lines can save the periodograms and phase-folded light curve to additional files
    #tes2 = np.column_stack([
    #    list(dict(list(results.items())[28:29]).values())[0],
    #    list(dict(list(results.items())[29:30]).values())[0],
    #    list(dict(list(results.items())[30:31]).values())[0],
    #    list(dict(list(results.items())[31:32]).values())[0],
    #    list(dict(list(results.items())[32:33]).values())[0],
    #    list(dict(list(results.items())[33:34]).values())[0]])
    #df2 = pd.DataFrame(tes2)
    #df2.columns = key_list[34:40]
    #df2.to_csv(file_name + 'power.csv')
    
    #tes3 = np.column_stack([
    #    list(dict(list(results.items())[36:37]).values())[0],
    #    list(dict(list(results.items())[37:38]).values())[0],
    #    list(dict(list(results.items())[38:39]).values())[0],
    #    list(dict(list(results.items())[39:40]).values())[0],
    #    list(dict(list(results.items())[40:41]).values())[0]])
    #df3 = pd.DataFrame(tes3)
    #df3.columns = key_list[42:47]
    #df3.to_csv(file_name + 'phase.csv')

def count(list1, l, r):
    """
    Count the number of elements in list1 that are between l and r (lower and upper bounds)
    """
    return len(list(x for x in list1 if l <= x <= r))

def flagging_criteria(all_results, save_direc='./transit_search/'):
    #This is the code for flagging whether a transit-like signal is detected
    new_results = []
    flag = False
    for i in all_results:
        if i.FAP < 1e-2 and (i.SDE > 7):
            if i.transit_count >= 2:
                new_results.append(i)
                transit_file = save_direc + 'transit_times.txt'
                file3 = open(transit_file, "a")  # append mode
                for j in range(i.transit_count):
                    file3.write(str(i.transit_times[j]) + " ")
                file3.write("\n")
                file3.close()
            else:
                low_file = save_direc + 'flagged_w_1_transit.txt'
                file2 = open(low_file, "a")  # append mode
                file2.write(str(ticid) + '\n')
                file2.close()
    new_ps = [i.period for i in new_results]
    for i in new_results:
        upper = i.period + 2*i.period_uncertainty
        lower = i.period - 2*i.period_uncertainty
        sim_count = count(new_ps, lower, upper)
        if sim_count > 1:
            flag = True
    if flag is True:
        print("Potential transit detected")
    else:
        print("No transit found")
    return flag

def determine_best_flux(light_curves):
    cap_tal = 0
    prf_tal = 0
    for i in range(len(light_curves)):
        cap_std = np.std(light_curves[i]['cal_cap_flux'].value)
        prf_std = np.std(light_curves[i]['cal_prf_flux'].value)
        if cap_std > prf_std:
            prf_tal += 1
        elif cap_std < prf_std:
            cap_tal += 1
    if cap_tal > prf_tal:
        print("Use cap")
        flux_id = 'cal_cap_flux'
    elif prf_tal > cap_tal:
        print("Use prf")
        flux_id = 'cal_prf_flux'
    else:
        cap_d = sum([np.std(light_curves[i]['cal_cap_flux'].value) for i in range(len(light_curves))])
        prf_d = sum([np.std(light_curves[i]['cal_prf_flux'].value) for i in range(len(light_curves))])
        if cap_d < prf_d:
            print("Use cap")
            flux_id = 'cal_cap_flux'
        else:
            print("Use prf")
            flux_id = 'cal_prf_flux'
    return flux_id
