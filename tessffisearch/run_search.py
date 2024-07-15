import ffisearch as ff
from transitleastsquares import catalog_info

def run_the_search(ticid, sigma_upper=4., sigma_lower=12., window_length=0.8, method='biweight'):
    light_curves, sectors = ff.retrieve_or_make_lc(ticid, lc_save_direc='./test_data/')
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
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=ticid)
    print("running transit search")
    all_results = []
    for i in range(len(light_curves)):
        results = ff.search_for_transit(all_times[i], all_flat[i], ab)
        all_results.append(results)
    print("Determine Flag")
    flag = ff.flagging_criteria(all_results, save_direc='./')
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

    n2_list = pd.read_csv("n2_targets_list.csv")
    T_mask = (n2_list['Tmag'] > 10.5) & (n2_list['Tmag'] < 15)
    n2_targ = n2_list[T_mask]
    n2_targ = n2_targ.sort_values('MES', ascending=False).reset_index()

    for tic_index in range(len(n2_targ)):
        ticid = int(n2_targ['ID'][tic_index])
        print("TIC", ticid)
        print("TESSmag", n2_targ['Tmag'][tic_index])
        run_the_search(ticid)

    end_time = time.time()

    runtime = (end_time - start_time) / 60

    print("runtime =", runtime, "minutes for", len(n2_targ), "targets.")
