import pdb
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import gsw
import sys
from numba import jit
import cartopy.crs as ccrs
import ipdb
import statsmodels.api as sm
import copy

def cmp_sigma(ds):
    ''' 
    compute potential density anomaly with reference to pressure of 0 dbar for
    Argo file
    '''

    TA = PA = SA = '0'

    if ('PRES_ADJUSTED' in ds.keys()):
        if (not np.all(np.isnan(ds.PRES_ADJUSTED.values))):
            PRES = ds.PRES_ADJUSTED
            PA = '1'
        else:
            PRES = ds.PRES
    else:
        PRES = ds.PRES

    if ('TEMP_ADJUSTED' in ds.keys()):
        if (not np.all(np.isnan(ds.TEMP_ADJUSTED.values))):
            TEMP = ds.TEMP_ADJUSTED
            TA = '1'
        else:
            TEMP = ds.TEMP
    else:
        TEMP = ds.TEMP

    if ('PSAL_ADJUSTED' in ds.keys()):
        if (not np.all(np.isnan(ds.PSAL_ADJUSTED.values))):
            PSAL = ds.PSAL_ADJUSTED
            SA = "1"
        else:
            PSAL = ds.PSAL
    else:
        PSAL = ds.PSAL


    ds['SA'] = gsw.SA_from_SP(PSAL, PRES, ds.LONGITUDE, ds.LATITUDE)
    ds.SA.attrs['units'] = 'g/kg'
    ds.SA.attrs['long_name'] = 'Absolute Salinity'
    ds.SA.attrs['standard_name'] = 'SA'

    ds['CT'] = gsw.CT_from_t(ds.SA, TEMP, PRES)
    ds.CT.attrs['units'] = 'deg C'
    ds.CT.attrs['long_name'] = 'Conservative Temperature'
    ds.CT.attrs['standard_name'] = 'CT'

    ds['sigma0'] = gsw.density.sigma0(ds.SA, ds.CT)
    ds.sigma0.attrs['units'] = 'kg/m3'
    ds.sigma0.attrs['long_name'] = 'Potential Density Anomaly'
    ds.sigma0.attrs['standard_name'] = 'sigma theta0'
    ds.sigma0.attrs['adjusted'] = 'P' + PA + "T" + TA + "S" + SA # info on which vars were adjusted

    if np.all(np.isnan(ds.sigma0.values)):
        print("WARNING: all sigma0 values are NaNs")
        pdb.set_trace()

    return ds

def cmp_spiciness(ds):
    '''
    compute spiciness0 with reference to pressure of 0 dbar for
    Argo file
    '''

    TA = PA = SA = '0'

    if 'PRES_ADJUSTED' in ds.keys():
        PRES = ds.PRES_ADJUSTED
        PA = '1'
    else:
        PRES = ds.PRES

    if 'TEMP_ADJUSTED' in ds.keys():
        TEMP = ds.TEMP_ADJUSTED
        TA = '1'
    else:
        TEMP = ds.TEMP

    if 'PSAL_ADJUSTED' in ds.keys():
        PSAL = ds.PSAL_ADJUSTED
        SA = "1"
    else:
        PSAL = ds.PSAL

    ds['spice'] = gsw.spiciness0(ds.SA, ds.CT)

    ds.spice.attrs['units'] = ' '
    ds.spice.attrs['long_name'] = 'Spiciness'
    ds.spice.attrs['standard_name'] = 'spice'

    return ds

# medfilt1 function similar to Octave's that does not bias extremes of dataset towards zero
@jit(nopython=True)  # this is to use numba to speed up calculations (by a factor 46!)
def medfilt1(data, kernel_size, endcorrection='shrinkkernel'):
    """One-dimensional median filter"""
    halfkernel = int(kernel_size / 2)
    data = np.asarray(data)

    filtered_data = np.empty(data.shape)
    filtered_data[:] = np.nan

    for n in range(len(data)):
        i1 = np.nanmax([0, n - halfkernel])
        i2 = np.nanmin([len(data), n + halfkernel + 1])
        filtered_data[n] = np.nanmedian(data[i1:i2])

    return filtered_data

@jit(nopython=True) # this is to use numba to speed up calculations (by a factor 46!)
def smooth_profile(x, y):

    def medfilt1_adp(data, kernel_size, endcorrection='shrinkkernel'):
        """One-dimensional median filter"""
        halfkernel = int(kernel_size / 2)
        data = np.asarray(data)

        filtered_data = np.empty(data.shape)
        filtered_data[:] = np.nan

        for n in range(len(data)):
            i1 = np.nanmax([0, n - halfkernel])
            i2 = np.nanmin([len(data), n + halfkernel + 1])
            filtered_data[n] = np.nanmedian(data[i1:i2])

        return filtered_data

    # compute x resolution
    xres = np.diff(x)
    xres = np.append(xres, xres[-1])  # assumes that the vertical resolution of the deepest bin
                                     # is the same as the previous one

    # initialise medfiltered array
    ymf = np.zeros(y.shape) * np.nan

    # ir_LT1 = np.where(xres < 1)[0]
    # if np.any(ir_LT1):
    #     win_LT1 = 11.
    #     ymf[ir_LT1] = medfilt1_adp(y[ir_LT1], win_LT1)
    #
    # ir_13 = np.where((xres >= 1) & (xres <= 3))[0]
    # if np.any(ir_13):
    #     win_13 = 7.
    #     ymf[ir_13] = medfilt1_adp(y[ir_13], win_13)
    #
    # ir_GT3 = np.where(xres > 3)[0]
    # if np.any(ir_GT3):
    #     win_GT3 = 5.
    #     ymf[ir_GT3] = medfilt1_adp(y[ir_GT3], win_GT3)


    # ir_LT1 = np.where(xres < 5)[0]
    # if np.any(ir_LT1):
    #     win_LT1 = 11.
    #     ymf[ir_LT1] = medfilt1_adp(y[ir_LT1], win_LT1)
    #
    # ir_13 = np.where((xres >= 5) & (xres < 50))[0]
    # if np.any(ir_13):
    #     win_13 = 7.
    #     ymf[ir_13] = medfilt1_adp(y[ir_13], win_13)
    #
    # ir_GT3 = np.where(xres > 50)[0]
    # if np.any(ir_GT3):
    #     win_GT3 = 5.
    #     ymf[ir_GT3] = medfilt1_adp(y[ir_GT3], win_GT3)
    #

    ir_LT1 = np.where(xres >0)[0]
    if np.any(ir_LT1):
        win_LT1 = 11.
        ymf[ir_LT1] = medfilt1_adp(y[ir_LT1], win_LT1)



    return ymf

# function to define adaptive median filtering based on Christina Schallemberg's suggestion for CHLA
def adaptive_medfilt1(xall, yall, PLOT=False):
    '''
    applies a median filtering with an adaptive window
        xall is PRES
        yall is variable to smooth
    '''

    # initialize output array
    filtered_y = yall * np.nan

    # check if the input variable is a vector or a matrix
    if len(xall.shape) == 2:
        # loop through all elements assuming the first axis is time
        for ijuld,yi in enumerate(yall):
            filtered_y[ijuld,:] = smooth_profile(xall[ijuld,:], yall[ijuld,:])
    else:
        filtered_y = smooth_profile(xall, yall)

    return filtered_y

def cmp_zm(ds):
    '''
    compute mixed-layer depth, zm, based on 0.03 kg/m3 sigma0 difference from surface (10 dbar)
    '''

    # see if we have adjusted vars
    PA = '0'

    if ('PRES_ADJUSTED' in ds.keys()):
        if (not np.all(np.isnan(ds.PRES_ADJUSTED.values))):
            PRES = ds.PRES_ADJUSTED
            PA = '1'
        else:
            PRES = ds.PRES
    else:
        PRES = ds.PRES

    # parameters
    SIGMA0_THRESH = 0.03 # [kg/m3]
    PRES_SURF = 10# [dbar]

    # check that we have sigma0, otherwise compute it
    if not "sigma0" in ds.keys():
        ds = cmp_sigma(ds)

    # initialize new DataArray
    ds['zm'] = xr.DataArray(data=np.nan*ds.LATITUDE.values, dims="N_PROF") # this is for storing the PRES @zm
    ds['zm_sigma0'] = xr.DataArray(data=np.nan*ds.LATITUDE.values, dims="N_PROF") # this is for storing the SIGMA0 @zm

    for ijuld, juld in enumerate(ds.JULD):
        # print(ijuld)

        # check that PRES is not all NaN for this profile
        if np.all(np.isnan(PRES.values[ijuld,:])):
            ds.zm[ijuld] = np.nan
            ds.zm_sigma0[ijuld] = np.nan
            continue

        # check that we have a deep-enough PRES array
        if np.nanmax(PRES.values[ijuld, :]) < 100:
            ds.zm[ijuld] = np.nan
            ds.zm_sigma0[ijuld] = np.nan
            continue

        # find first PRES where the difference in density from the surface is equal to the 0.03 kg/m3 threshold
        # check that we have a PRES values at the surface
        if not np.any(PRES[ijuld, :] <= PRES_SURF):
            # ds.zm[ijuld] = np.nan
            # ds.zm_sigma0[ijuld] = np.nan
            # continue
            PRES_SURF = np.nanmin(PRES[ijuld, :]) # assign new surface pressure

        iSurf = np.where(PRES[ijuld, :] <= PRES_SURF)[0][-1]  # select last element to find the deepest point in the profile where the condition is met
        delta_sigma0 = ds.sigma0[ijuld, :] - ds.sigma0[ijuld, iSurf] # sigma0 difference from sigma0[surface]

        if np.isnan(ds.sigma0[ijuld, iSurf]):
            ds.zm[ijuld] = np.nan
            ds.zm_sigma0[ijuld] = np.nan
            continue

        innan = np.where(~np.isnan(delta_sigma0))[0]
        if np.all(delta_sigma0[innan] < SIGMA0_THRESH): # if delta_sigma0 is always less than SIGMA0_THRESH
                                                        # we assume a homogeneous profile all the way to the bottom
            izm = innan[-1] # set izm equal to the deepest non-nan value of the profile
            print("------ zm set equal to max depth")
        else:
            izm = np.where(delta_sigma0 >= SIGMA0_THRESH)[0][0] # find first element of delta_sigma0 that is greater than the threshold

        ds.zm[ijuld] = PRES.values[ijuld, izm] # extract PRES corresponding to zm
        ds.zm_sigma0[ijuld] = ds.sigma0.values[ijuld, izm] # extract SIGMA0 corresponding to zm

    ds.zm.attrs['units'] = 'dbar'
    ds.zm.attrs['long_name'] = 'Mixed-Layer Depth'
    ds.zm.attrs['standard_name'] = 'zm'
    ds.zm.attrs['adjusted'] = 'P' + PA  # info on which input vars were adjusted

    ds.zm_sigma0.attrs['units'] = 'kg/m3'
    ds.zm_sigma0.attrs['long_name'] = 'sigma0 at Mixed-Layer Depth'
    ds.zm_sigma0.attrs['standard_name'] = 'zm_sigma0'

    return ds

def cmp_zm_interpolated_data(ds_in):
    '''
    compute mixed-layer depth, zm, based on 0.03 kg/m3 sigma0 difference from surface (10 dbar)
    '''

    # see if we have adjusted vars
    PA = '0'

    if ('PRES_ADJUSTED' in ds_in.keys()):
        if (not np.all(np.isnan(ds_in.PRES_ADJUSTED.values))):
            PRES = ds_in.PRES_ADJUSTED
            PA = '1'
        else:
            PRES = ds_in.PRES
    else:
        PRES = ds_in.PRES

    # parameters
    SIGMA0_THRESH = 0.03 # [kg/m3]
    PRES_SURF = 10# [dbar]

    # check that we have sigma0, otherwise compute it
    if not "sigma0" in ds_in.keys():
        ds_in = cmp_sigma(ds_in)

    # initialize new DataArray
    # ds_in['zm'] = xr.DataArray(data=np.nan * ds_in.LATITUDE.values, dims="JULD") # this is for storing the PRES @zm
    # ds_in['zm_sigma0'] = xr.DataArray(data=np.nan * ds_in.LATITUDE.values, dims="JULD") # this is for storing the SIGMA0 @zm
    ds_in['zm'] = ds_in.LATITUDE.copy(deep=True) * np.nan # this is for storing the PRES @zm
    ds_in['zm_sigma0'] = ds_in.LATITUDE.copy(deep=True) * np.nan # this is for storing the SIGMA0 @zm

    for ijuld, juld in enumerate(ds_in.JULD):
        # print(ijuld)

        # check that PRES is not all NaN for this profile
        if np.all(np.isnan(PRES.values)):
            ds_in.zm[ijuld] = np.nan
            ds_in.zm_sigma0[ijuld] = np.nan
            continue

        # check that we have a deep-enough PRES array
        if np.nanmax(PRES.values) < 100:
            ds_in.zm[ijuld] = np.nan
            ds_in.zm_sigma0[ijuld] = np.nan
            continue

        # find first PRES where the difference in density from the surface is equal to the 0.03 kg/m3 threshold
        # check that we have a PRES values at the surface
        if not np.any(PRES <= PRES_SURF):
            # ds.zm[ijuld] = np.nan
            # ds.zm_sigma0[ijuld] = np.nan
            # continue
            PRES_SURF = np.nanmin(PRES) # assign new surface pressure

        iSurf = np.where(PRES <= PRES_SURF)[0][-1]  # select last element to find the deepest point in the profile where the condition is met
        delta_sigma0 = ds_in.sigma0[ijuld, :] - ds_in.sigma0[ijuld, iSurf] # sigma0 difference from sigma0[surface]

        if np.isnan(ds_in.sigma0[ijuld, iSurf]):
            ds_in.zm[ijuld] = np.nan
            ds_in.zm_sigma0[ijuld] = np.nan
            continue

        innan = np.where(~np.isnan(delta_sigma0))[0]
        if np.all(delta_sigma0[innan] < SIGMA0_THRESH): # if delta_sigma0 is always less than SIGMA0_THRESH
                                                        # we assume a homogeneous profile all the way to the bottom
            izm = innan[-1] # set izm equal to the deepest non-nan value of the profile
            print("------ zm set equal to max depth")
        else:
            izm = np.where(delta_sigma0 >= SIGMA0_THRESH)[0][0] # find first element of delta_sigma0 that is greater than the threshold

        ds_in.zm[ijuld] = PRES.values[izm] # extract PRES corresponding to zm
        ds_in.zm_sigma0[ijuld] = ds_in.sigma0.values[ijuld, izm] # extract SIGMA0 corresponding to zm

    ds_in.zm.attrs['units'] = 'dbar'
    ds_in.zm.attrs['long_name'] = 'Mixed-Layer Depth'
    ds_in.zm.attrs['standard_name'] = 'zm'
    ds_in.zm.attrs['adjusted'] = 'P' + PA  # info on which input vars were adjusted

    ds_in.zm_sigma0.attrs['units'] = 'kg/m3'
    ds_in.zm_sigma0.attrs['long_name'] = 'sigma0 at Mixed-Layer Depth'
    ds_in.zm_sigma0.attrs['standard_name'] = 'zm_sigma0'

    return ds_in

def cmp_zeu(ds):
    '''
    compute depth of euphotic zone using CHLA data
    '''

    CHLA_THRESHOLD = 0.04 # [mg/m3] threshold above the deep CHLA value where the zeu is
                          #         found (2X the minimum surface chla)

    # intialize output vector
    ds['zeu'] = ds.LATITUDE.copy(deep=True) * np.nan
    ds['zeu'].assign_attrs({'units': 'dbar'})

    if 'CHLA' not in ds.keys():
        print('CHLA not available: no zeu')
        return ds

    # check if we have ADJUSTED variables
    if ("CHLA_ADJUSTED" in ds.keys()) & (not np.all(np.isnan(ds.CHLA_ADJUSTED.values))):
        CHLA = ds.CHLA_ADJUSTED.values
    else:
        CHLA = ds.CHLA.values

    if ('PRES_ADJUSTED' in ds.keys()) & (not np.all(np.isnan(ds.PRES_ADJUSTED.values))):
        PRES = ds.PRES_ADJUSTED.values
    else:
        PRES = ds.PRES.values

    # smooth CHLA (matrix) using median filter
    ds['CHLA_smooth'] = ds.CHLA.copy(deep=True) * 0. # initialize new array
    ds['CHLA_smooth'].values = adaptive_medfilt1(PRES, CHLA)

    # find zeu for each profile
    for ijuld, CHLA_smooth in enumerate(ds['CHLA_smooth'].values):

        # # find median value of deep (>850 dbar) CHLA
        i_deep = np.where(PRES[ijuld, :] > 850)[0]
        CHLA_smooth_deep = np.nanmedian(CHLA_smooth[i_deep])

        # from the bottom of the profile search for the first CHLA_smooth (CHLA_ZEU) that is 0.02 mg/m3 higher than the deep CHLA_smooth
        ind = np.argwhere(CHLA_smooth >= (CHLA_smooth_deep + CHLA_THRESHOLD)) # see https://stackoverflow.com/questions/49612061/how-to-find-last-k-indexes-of-vector-satisfying-condition-python-analogue
        if ind.any():
            i_zeu = ind[-1:].flatten()[0]
            # extract the PRES of the above CHLA_ZEU
            ds['zeu'][ijuld] = PRES[ijuld, i_zeu]
        else:
            ds['zeu'][ijuld] = np.nan

    return ds

def cmp_zp(ds):
    '''
    compute depth of productive zone
    '''

    # intialize output vector
    ds['zp'] = ds.LATITUDE.copy(deep=True) * np.nan
    ds['zp'].assign_attrs({'units': 'dbar'})

    # check that zm and zeu are available
    if not(('zm' in ds.keys()) and ('zeu' in ds.keys())):
        print('missing zm or zeu')
        return ds

    # compute zp for each profile
    for ijuld,tmp in enumerate(ds.JULD.values):
        ds['zp'][ijuld] = np.nanmax([ds['zm'][ijuld], ds['zeu'][ijuld]])

    return ds

def cmp_o2sat_aou(ds):
    # compute oxygen solubility
    ds['O2SOL'] = gsw.O2sol(ds.SA, ds.CT, ds.PRES, ds.LONGITUDE, ds.LATITUDE)

    # compute percent oxygen saturation
    if "DOXY_ADJUSTED" in ds.keys():
        ds['O2SAT'] = ds.O2SOL/ds.DOXY_ADJUSTED*100.
    else:
        ds['O2SAT'] = ds.O2SOL/ds.DOXY*100.

    # compute AOU
    if "DOXY_ADJUSTED" in ds.keys():
        ds['AOU'] = ds.O2SOL - ds.DOXY_ADJUSTED
    else:
        ds['AOU'] = ds.O2SOL - ds.DOXY

    ds.AOU.attrs['units'] = 'umol/kg'
    ds.AOU.attrs['long_name'] = 'Apparent Oxygen Utilisation'
    ds.AOU.attrs['standard_name'] = 'AOU'

    ds.O2SOL.attrs['units'] = 'umol/kg'
    ds.O2SOL.attrs['long_name'] = 'oxygen concentration expected at equilibrium with air at an Absolute Pressure of 101325 Pa'
    ds.O2SOL.attrs['standard_name'] = 'O2 solubility'

    ds.O2SAT.attrs['units'] = '%'
    ds.O2SAT.attrs['long_name'] = 'percent oxygen saturation'
    ds.O2SAT.attrs['standard_name'] = 'O2 saturation'

    return ds

def bin_on_sigma_layers(ds4bin, sigma0_bins=np.arange(27.2, 27.6, 0.1), np_functs=['nanmean', 'nanstd', 'counts']):
    '''
    compute statistics of variables needed for respiration on sigma layers
    '''

    def ini_df(n_rows, vrbls):
        '''this function create an empty dataframe that is then used to initialize the dictionary'''

        # initialize columns of dataframes
        col = {ivar: np.nan * np.ones(n_rows) for ivar in vrbls}

        # initialise template of pandas dataframe where a given stat will be stored
        df_tmp = pd.DataFrame(data=col)
        df_tmp.columns = vrbls

        return df_tmp

    # first groupby in bins of density
    group_binned_by_sigma0 = ds4bin.groupby_bins(group=ds4bin.sigma0, bins=sigma0_bins, right=True, labels=None,
                                                 include_lowest=True, squeeze=True, restore_coord_dims=True)

    # extract variables from ds4bin
    vrbls = [ii for ii in ds4bin.data_vars.keys()]
    # vrbls.remove('JULD')

    # initialize output dictionary
    df_binned = {}

    # loop through sigma bins
    print("Binning on sigma layers...")
    for isigma, grouped_in_1_layer in group_binned_by_sigma0:
        print(isigma)

        # create name of dictionary for this layer
        nm_layer = f's0_{isigma.left:.02f}'.replace('.', '') + "_" + f'{isigma.right:.02f}'.replace('.', '')

        # find unique JULD
        juld_uq = np.unique(grouped_in_1_layer['JULD'].values)

        # initialise dictionaries for this layer
        # df_binned[nm_layer] = {
        #     'mean': df_tmp.copy(deep=True),
        #     'std': df_tmp.copy(deep=True),
        #     'N': df_tmp.copy(deep=True),
        # }
        df_tmp = ini_df(len(juld_uq), vrbls)
        df_binned[nm_layer] = {fun: df_tmp.copy(deep=True) for fun in
                               np_functs}  # add empty df_tmp for each function to df_binned

        # add index to each function
        df_binned[nm_layer]['nanmean'].index = juld_uq
        df_binned[nm_layer]['nanstd'].index = juld_uq
        df_binned[nm_layer]['counts'].index = juld_uq

        # loop through juld_uq and compute stats for each variable
        for ii, juld in enumerate(juld_uq):
            ijuld = np.where(grouped_in_1_layer['JULD'].values == juld)[0]

            # loop through variables to bin and compute stats for given variable
            for var in vrbls:
                if not np.all(np.isnan(grouped_in_1_layer[var].values[ijuld])):
                    df_binned[nm_layer]['nanmean'][var][ii] = np.nanmean(grouped_in_1_layer[var].values[ijuld])
                    df_binned[nm_layer]['counts'][var][ii] = len(
                        np.where(~np.isnan(grouped_in_1_layer[var].values[ijuld]))[0])
                    if df_binned[nm_layer]['counts'][var][ii] > 2:
                        df_binned[nm_layer]['nanstd'][var][ii] = np.nanstd(grouped_in_1_layer[var].values[ijuld])
                    else:
                        df_binned[nm_layer]['nanstd'][var][ii] = np.nan

    print("...done")
    #         print(df_binned[nm_layer])
    # sort keys in binned dataframe
    myKeys = list(df_binned.keys())  # extract list of keys
    myKeys.sort()  # sort list of keys

    sorted_df_binned = {i: df_binned[i] for i in myKeys}  # create new
    #                                                          # dictionary with sorted keys

    return sorted_df_binned

def plot_binned(df_binned, var, np_fun='nanmean', stride=1, LEGEND=False):
    # plot time series of binned variable var for each sigma layer

    # check that variable is in df_binned
    lay1 = list(df_binned.keys())[0]
    if var not in df_binned[lay1][np_fun]:
        print('ERROR: Variable not in df_binned')
        return

    fig, ax = plt.subplots(1, figsize=(10,3))
    for isigmalay in list(df_binned.keys())[::stride]:
        df_binned[isigmalay][np_fun].plot(y=var, marker='o', ms=4, ls='-',
                                          lw=0.2, # figsize=(13, 4),
                                          label=isigmalay, ax=ax,
                                          ylabel=var+"-"+np_fun)
        ax.grid('on', linestyle='--')
        if LEGEND:
            ax.legend(loc='best', bbox_to_anchor=(1.05, 1.0), fontsize=10)
        else:
            ax.get_legend().remove()

        if (var == "AOU") & (np_fun == "nanmean"):
            ks = list(df_binned.keys())[-1]
            maxAOU = np.round(np.nanmax(df_binned[ks]['nanmean']['AOU']) + 5)
            ax.set_ylim([0, maxAOU+10])

    return fig, ax

def plot_map(ds, figsize=(4, 2.5)):
    '''
    plot global map
    ds: xarray dataset with 'LATITUDE' and 'LONGITUDE' variables
    figsize: parameter to specify figsize=(width, height)
    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=200))

    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()

    ax.stock_img()
    ax.coastlines()

    ax.plot(ds.LONGITUDE, ds.LATITUDE, transform=ccrs.PlateCarree(), c='m', lw=2);

    return fig, ax

def year(t):
    '''
    matlab style function that takes as input a numpy.datetime64 and returns the year
    '''
    # check if type is numpy.datetime64
    if type(t[0]).__module__ + '.' + type(t[0]).__name__ != "numpy.datetime64":
        print('input type should be numpy.datetime64, found: ' + str(type(t)))
        return t.astype(float) * np.nan

    year = t.astype('datetime64[Y]').astype(int) + 1970

    return year

def month(t):
    '''
    matlab style function that takes as input a numpy.datetime64 and returns the month
    '''
    # check if type is numpy.datetime64
    if type(t[0]).__module__ + '.' + type(t[0]).__name__ != "numpy.datetime64":
        print('input type should be numpy.datetime64, found: ' + str(type(t)))
        return t.astype(float) * np.nan

    month = t.astype('datetime64[M]').astype(int) % 12 + 1

    return month

def day(t):
    '''
    matlab style function that takes as input a numpy.datetime64 and returns the day of the month
    '''
    # check if type is numpy.datetime64
    if type(t[0]).__module__ + '.' + type(t[0]).__name__ != "numpy.datetime64":
        print('input type should be numpy.datetime64, found: ' + str(type(t)))
        return t.astype(float) * np.nan

    day = (t - t.astype('datetime64[M]')).astype(int) + 1

    return day

def define_sigma_extremes(ds):
    ### this is NOT YET for under-ice profiles ###

    if 'PRES_ADJUSTED' in ds:
        PRES = ds.PRES_ADJUSTED.values
    else:
        PRES = ds.PRES.values

    ## first the bottom ##
    # find time index of the maximum winter mixed-layer depth (zm_max)
    i_1st_half = np.where(ds.JULD.values <= ds.JULD.values[0] + np.timedelta64(6 * 30,
                                                                                  'D'))  # indices of the first part of the year over which we look for zm_max
    it_zm_max = np.where(ds.zm.values[i_1st_half] >= np.nanmax(ds.zm.values[i_1st_half]))[0][0]

    # find PRES index of zm_max
    iP_zm_max = np.where(abs(PRES[it_zm_max, :] - ds.zm.values[it_zm_max]) == np.nanmin(
        abs(PRES[it_zm_max, :] - ds.zm.values[it_zm_max])))[0][0]

    # at it_zm_max, find isopycnal surface just below zm_max, add a delta-sigma0 and define this as the bottom limit above which we estimate respiration
    sigma0_max = np.round(ds.sigma0.values[it_zm_max, iP_zm_max], 2) + 0.5

    ## then the top ##
    #     # I need to make sure that this is below 150-200 dbar
    #     it_zm_min = np.where(ds.zm.values == np.nanmin(ds.zm.values))[0][0]  # find time index of minimum mixed-layer depth

    # here's my first try
    PRES_MAX = 150  # [dbar]
    # find first sigma0 below PRES_MAX
    sigma_PRES_MAX = np.asarray(
        [ds.sigma0.values[ipre, np.where(pre > PRES_MAX)[0][0]] for ipre, pre in enumerate(ds.PRES.values)])
    # compute mean value of sigma0 below PRES_MAX
    sigma_MIN = np.nanmin(sigma_PRES_MAX)
    it_zm_min = np.where(ds.sigma0.values == sigma_MIN)[0][
        0]  # find isopycnal that is "always" below PRES_MAX  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TIME IS NOT USED ANYWHERE

    # find PRES index of zm_min
    iP_zm_min = np.where(abs(PRES[it_zm_min, :] - ds.zm.values[it_zm_min]) == np.nanmin(
        abs(PRES[it_zm_min, :] - ds.zm.values[it_zm_min])))[0][0]

    # at it_zm_min, find isopycnal surface just below zm_min and define this as the (provisional) top limit below which we estimate respiration
    sigma0_min = np.round(ds.sigma0.values[it_zm_min, iP_zm_min],
                          2)  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< THIS IS WHAT IS USED

    return it_zm_max, sigma0_max, it_zm_min, sigma0_min

def extract_unique_year(ds_, i_year, FIRST_MONTH, N_DAYS=365):
    ## find beginning of the float year
    i_yr_start = np.where((ds_['YEAR'].values == i_year) & (ds_['MONTH'].values == FIRST_MONTH))[0] # find indices of first month of the float year
    if i_yr_start.size > 0:
        i_yr_start = i_yr_start[0] # index of first day of float year
    else:
        print("incomplete year (start)")
        ds_yr = []
        return ds_yr

    ## find end of the float year
    i_yr_end = np.where( ds_['JULD'].values >= ds_['JULD'].values[i_yr_start] + np.timedelta64(N_DAYS, 'D') ) [0] # find indices of profiles after 365 days from the first day of the float year
    if i_yr_end.size > 0:
        i_yr_end = i_yr_end[0] # index of last day of float year
    else:
        print("incomplete year (end)")
        ds_yr = []
        return ds_yr

    ## extract data from this float year
    ds_yr = ds_.sel(JULD=slice(ds_['JULD'].values[i_yr_start], ds_['JULD'].values[i_yr_end]))

    return ds_yr

def filter_binned(df_yr_binned, MIN_N_POINTS_JULD=3, MAX_AOU_STD=2, MAX_PSAL_STD=0.01, MIN_PRES=150, MIN_N_POINTS_SERIES=4):

    # create copy of input dataset
    df_yr_binned_filt_ = copy.deepcopy(df_yr_binned)
    layrs = list(df_yr_binned_filt_.keys())
    funcs = list(df_yr_binned_filt_[layrs[0]].keys())
    vrs = list(df_yr_binned_filt_[layrs[0]]['nanmean'].keys())

    #     print(df_yr_binned['s0_2742_2752']['counts'].head())

    for layr in layrs:
        # remove negative AOUs
        i2rm = np.where(df_yr_binned_filt_[layr]['nanmean']['AOU'] < 0)[0]
        #         print(len(i2rm)/len(df_yr_binned_filt_[layr]['nanmean']['AOU'].values))
        if i2rm.size > 0:
            for func in funcs:
                for vr in vrs:
                    df_yr_binned_filt_[layr][func][vr].values[i2rm] = np.nan
            print("negative AOUs ", layr, ": ", len(i2rm), " points removed")
        #             print(df_yr_binned['s0_2742_2752']['counts'].head())

        # remove AOUs with too few counts
        i2rm = np.where(df_yr_binned_filt_[layr]['counts']['AOU'] < MIN_N_POINTS_JULD)[0]
        #         print(len(i2rm)/len(df_yr_binned_filt_[layr]['counts']['AOU'].values))
        if i2rm.size > 0:
            for func in funcs:
                for vr in vrs:
                    df_yr_binned_filt_[layr][func][vr].values[i2rm] = np.nan
            print("AOUs with too few counts ", layr, ": ", len(i2rm), " points removed")
        #             print(df_yr_binned['s0_2742_2752']['counts'].head())
        # remove AOUs with nanstd too high
        # MAX_AOU_STD = 2  # [umol/kg]
        i2rm = np.where(df_yr_binned_filt_[layr]['nanstd']['AOU'] > MAX_AOU_STD)[0]
        #         print(len(i2rm)/len(df_yr_binned_filt_[layr]['nanstd']['AOU'].values))
        if i2rm.size > 0:
            for func in funcs:
                for vr in vrs:
                    df_yr_binned_filt_[layr][func][vr].values[i2rm] = np.nan
            print("AOUs with nanstd too high ", layr, ": ", len(i2rm), " points removed")
        #             print(df_yr_binned['s0_2742_2752']['counts'].head())
        # remove PSAL with nanstd too high
        # MAX_PSAL_STD = 0.01  # [-]
        i2rm = np.where(df_yr_binned_filt_[layr]['nanstd']['PSAL'] > MAX_PSAL_STD)[0]
        #         print(len(i2rm)/len(df_yr_binned_filt_[layr]['nanstd']['PSAL'].values))
        if i2rm.size > 0:
            for func in funcs:
                for vr in vrs:
                    df_yr_binned_filt_[layr][func][vr].values[i2rm] = np.nan
            print("PSAL with nanstd too high ", layr, ": ", len(i2rm), " points removed")
        #             print(df_yr_binned['s0_2742_2752']['counts'].head())
        # remove data shallower than 200 dbar
        # MIN_PRES = 150  # [dbar]
        i2rm = np.where(df_yr_binned_filt_[layr]['nanmean']['PRES'] < MIN_PRES)[0]
        #         print(len(i2rm)/len(df_yr_binned_filt_[layr]['nanmean']['PRES'].values))
        if i2rm.size > 0:
            for func in funcs:
                for vr in vrs:
                    df_yr_binned_filt_[layr][func][vr].values[i2rm] = np.nan
            print("data shallower than 200 dbar ", layr, ": ", len(i2rm), " points removed")
        #             print(df_yr_binned['s0_2742_2752']['counts'].head())
        # remove layers with too few points
        # MIN_N_POINTS_SERIES = 4
        n_points = len(np.where(~np.isnan(df_yr_binned_filt_[layr]['nanmean']['AOU'].values))[0])
        if n_points < MIN_N_POINTS_SERIES:
            del df_yr_binned_filt_[layr]
    #             print("layers with too few points ", layr, "(", n_points, ")")
    #     print(df_yr_binned['s0_2742_2752']['counts'].head())
    return df_yr_binned_filt_

def fit_linear(x, y):
    x = sm.add_constant(x)
    ols = sm.OLS(y, exog=x)
    ols_result = ols.fit()
    ols_result.summary()

    rlm = sm.RLM(y, x, sm.robust.norms.TrimmedMean(0.5))
    rlm_result = rlm.fit(maxiter=50,
                         tol=1e-08,
                         scale_est='mad',
                         init=None,
                         cov='H1',
                         update_scale=True,
                         conv='dev',
                         start_params=None
                         )
    rlm_result.summary()

    return ols, ols_result, rlm, rlm_result

def umolO2_kg_d_TO_mmolC_m3_yr(R, R_ERR=False, SIGMA=1027):
    '''convert R from umolO2/kg/d to mmolC/m3/yr
       if provided, convert also the uncertainty in R using the standard law for the propagation of uncertainties'''

    # conversion constants and units
    Oxy2_to_Carb = 170 / 117  # [umolO2/umolC] Anderson and Sarmiento (1994)
    DAYSinYEAR = 365.25  # [d]
    # SIGMA = 1027 # [kg/m3]
    umol2mmol = 1000  # [umol/mmol]

    ### the three equations below are the "Measurement equation" for the SLPU
    RC = R / Oxy2_to_Carb  # [umolO2/kg/d * umolC/umolO2] = [umolC/kg/d]
    RC = RC * SIGMA / umol2mmol  # [umolC/kg/d * kg/m3 / (umol/mmol)] = [mmolC/m3/d]
    RC = RC * DAYSinYEAR  # [mmolC/m3/d * d/yr] = [mmolC/m3/yr]

    if np.any(R_ERR):
        dRC_dR = 1. / Oxy2_to_Carb * SIGMA / umol2mmol * DAYSinYEAR  # this is the "sensitivity" (i.e. the derivation of the Measurement Equation with repsect to the input variable(s) that introduce uncertainties in the final outut)
        RC_ERR = np.sqrt((R_ERR * dRC_dR) ** 2)  # this is the SLPU for this
        return RC, RC_ERR

    else:
        return RC, np.nan




if __name__ == '__main__':
    cmp_sigma(sys.argv[0])
