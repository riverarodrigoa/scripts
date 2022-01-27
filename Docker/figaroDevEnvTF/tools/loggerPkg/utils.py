import os
import numpy as np
import pandas as pd
import string
import datetime
from tqdm import tqdm
from loggerPkg.graphics import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from scipy.signal import savgol_filter, medfilt, find_peaks

from pathlib import Path
from itertools import combinations, product, chain

from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score
from sklearn import linear_model
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split

# QuantileTransformer, MinMaxScaler,
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import skill_metrics as sm


def filter_vars(df_ff, variables, window=211):
    df_ff = df_ff.loc[:, variables].copy()
    for n, i in enumerate(variables):
        if i not in ['P_BMP180', 'P_BMP280']:
            ix = df_ff[(df_ff[i] >= 0) & (df_ff[i] <= 100)].index
            ixx = ~df_ff.index.isin(ix)
            df_ff.loc[ixx, i] = np.nan
        vv = df_ff.loc[:, [i]].copy()
        dd = pd.DataFrame()
        dd = vv.dropna()
        dd = pd.DataFrame(dd)
        values = list(chain(*dd.values))
        dd[i] = savgol_filter(values, window, 3)
        dd[i] = medfilt(dd[i].values, window)
        dd.columns = [i+'_filter']
        dd = pd.DataFrame(dd)
        vv = vv.join(dd)
        df_ff = df_ff.join(dd)
    return df_ff


def ts_correct(df, variable, ix_time_shift):
    ddf = df.copy()
    for n, (time, ts) in enumerate(ix_time_shift):
        if n == 0:
            t = ddf.loc[time[0]:time[1], variable].shift(ts)
        else:
            tt = ddf.loc[time[0]:time[1], variable].shift(ts)
            t = t.append(tt)
    return t


def computeStats(yTrR, yTeR, yTrM, yTeM, kobayashiDec=False):
    msdTrain = mse(yTrR, yTrM)
    msdTest = mse(yTeR, yTeM)
    rmseTrain = np.sqrt(msdTrain)
    rmseTest = np.sqrt(msdTest)
    biasTrain = np.mean(yTrM - yTrR)
    biasTest = np.mean(yTeM - yTeR)
    sdTrain = 100 * (np.std(yTrM) / np.std(yTrR))
    sdTest = 100 * (np.std(yTeM) / np.std(yTeR))
    corrTrain = np.corrcoef(yTrM.ravel(), yTrR.ravel())[0, 1]
    corrTest = np.corrcoef(yTeM.ravel(), yTeR.ravel())[0, 1]
    sdDeltaTrain = np.std(yTrM - yTrR)
    sdDeltaTest = np.std(yTeM - yTeR)
    fomTrain = np.trapz(yTrM.reshape(
        1, -1)[0]) / np.trapz(yTrR.reshape(1, -1)[0])
    fomTest = np.trapz(yTeM.reshape(
        1, -1)[0]) / np.trapz(yTeR.reshape(1, -1)[0])
    if kobayashiDec:
        # Train
        sbTr = (np.mean(yTrM) - np.mean(yTrR)) ** 2
        sd_s = np.sqrt(np.sum((yTrM - np.mean(yTrM)) ** 2) / len(yTrM))
        sd_m = np.sqrt(np.sum((yTrR - np.mean(yTrR)) ** 2) / len(yTrR))
        rho = (np.sum((yTrM - np.mean(yTrM)) *
               (yTrR - np.mean(yTrR))) / len(yTrM)) / (sd_m * sd_s)
        sdsdTr = (sd_s - sd_m) ** 2
        lcsTr = 2 * sd_s * sd_m * (1 - rho)
        msvTr = sdsdTr + lcsTr
        msdTr = sbTr + sdsdTr + lcsTr
        # Test
        sbTe = (np.mean(yTeM) - np.mean(yTeR)) ** 2
        sd_s = np.sqrt(np.sum((yTeM - np.mean(yTeM)) ** 2) / len(yTeM))
        sd_m = np.sqrt(np.sum((yTeR - np.mean(yTeR)) ** 2) / len(yTeR))
        rho = (np.sum((yTeM - np.mean(yTeM)) *
               (yTeR - np.mean(yTeR))) / len(yTeM)) / (sd_m * sd_s)
        sdsdTe = (sd_s - sd_m) ** 2
        lcsTe = 2 * sd_s * sd_m * (1 - rho)
        msvTe = sdsdTe + lcsTe
        msdTe = sbTe + sdsdTe + lcsTe
        return msdTrain, msdTest, rmseTrain, rmseTest, biasTrain, biasTest, sdTrain, sdTest, corrTrain, corrTest, sdDeltaTrain, sdDeltaTest, fomTrain, fomTest, msdTr, sbTr, sdsdTr, lcsTr, msdTe, sbTe, sdsdTe, lcsTe
    else:
        return msdTrain, msdTest, rmseTrain, rmseTest, biasTrain, biasTest, sdTrain, sdTest, corrTrain, corrTest, sdDeltaTrain, sdDeltaTest, fomTrain, fomTest


def plot_H2O_regression(dr, model, xvar, yvar, xlabel, ylabel, logger, r_2_score, mode=1, filename=None):
    if mode == 1:
        with plt.style.context('seaborn-whitegrid'):
            fig, ax = plt.subplots(
                ncols=1, nrows=1, figsize=(10, 10), squeeze=False)
            ax[0, 0] = dr.plot(x=xvar, y=yvar, ax=ax[0, 0], style='.', ms=1)
            ax[0, 0] = dr.plot(x=xvar, y='y_pred',
                               ax=ax[0, 0], style='-', ms=1)
            # '$\mathrm{H_{2}O}$ Mole fraction (%)'
            ax[0, 0].set_xlabel(xlabel)
            ax[0, 0].set_ylabel(ylabel)  # 'TGS 2611-C00 (V)'
            ax[0, 0].lines[0].set_color('grey')
            ax[0, 0].lines[1].set_color('red')
            ax[0, 0].legend('')

            ax[0, 0] = logger.format_ax0(ax[0, 0], time=False)
            props = dict(boxstyle='round', alpha=0.5)

            ax[0, 0].text(0.1, 0.9,
                          f"y = {round(model.named_steps['linearregression'].coef_[0][0],3)}x + {round(model.named_steps['linearregression'].intercept_[0],3)}\nR2: {round(r_2_score,3)}",
                          transform=ax[0, 0].transAxes, size=20, weight='normal', bbox=props)
            fig.align_ylabels(ax[:, 0])
            plt.subplots_adjust(left=None, bottom=None,
                                right=None, top=None, wspace=None, hspace=0.4)
            plt.xticks(ha='center')
    else:
        with plt.style.context('seaborn-whitegrid'):
            fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(
                20, 12), sharex=True, squeeze=False)
            ax[0, 0] = dr.loc[:, [yvar, 'Baseline_pred']].plot(
                ax=ax[0, 0], style='.', ms=1)
            ax[1, 0] = dr.loc[:, ['BS_rem']].plot(ax=ax[1, 0], style='.', ms=1)
            ax[0, 0].set_ylabel(ylabel)
            ax[1, 0].set_ylabel(ylabel)
            ax[0, 0].lines[0].set_color('grey')
            ax[0, 0].lines[1].set_color('seagreen')
            ax[1, 0].lines[0].set_color('seagreen')
            ax[0, 0].legend(['Raw obs.', 'H2O cross sensitivities'],
                            markerscale=8, loc='upper left', frameon=True)
            ax[1, 0].legend(['Offset correction'], markerscale=8,
                            loc='upper left', frameon=True)

            ax[0, 0] = logger.format_ax0(ax[0, 0], time=True)
            ax[1, 0] = logger.format_ax0(ax[1, 0], time=True)

            ax[0, 0].text(-0.08, 1.0, '('+string.ascii_lowercase[0]+')',
                          transform=ax[0, 0].transAxes, size=20, weight='bold')
            ax[1, 0].text(-0.08, 1.0, '('+string.ascii_lowercase[1]+')',
                          transform=ax[1, 0].transAxes, size=20, weight='bold')
            fig.align_ylabels(ax[:, 0])
            plt.subplots_adjust(left=None, bottom=None,
                                right=None, top=None, wspace=None, hspace=0.1)
            plt.xticks(ha='center')

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')


def plot_detection(df_spk, logger, variable, binary):
    if len(df_spk[df_spk[binary] == 1]) == len(df_spk):
        print('All obs. detected as spikes')
        with plt.style.context('seaborn-whitegrid'):
            fig, ax = plt.subplots(
                ncols=1, nrows=1, figsize=(20, 8), squeeze=False)
            ax[0, 0] = df_spk[df_spk[binary] == 1].loc[:,
                                                       variable].plot(ax=ax[1, 0], style='.', ms=1)
            ax[0, 0].set_ylabel('TGS 2611-C00 (V)')

            ax[0, 0].lines[0].set_color('gray')
            ax[0, 0].lines[0].set_label('')

            ax[0, 0] = logger.format_ax0(ax[0, 0], time=True)

            fig.align_ylabels(ax[:, 0])
            plt.subplots_adjust(left=None, bottom=None,
                                right=None, top=None, wspace=None, hspace=0.4)
            plt.xticks(ha='center')
    else:
        with plt.style.context('seaborn-whitegrid'):
            fig, ax = plt.subplots(
                ncols=1, nrows=2, figsize=(20, 8), squeeze=False)
            ax[0, 0] = df_spk[df_spk[binary] == 0].loc[:,
                                                       variable].plot(ax=ax[0, 0], style='.', ms=1)
            ax[1, 0] = df_spk[df_spk[binary] == 1].loc[:,
                                                       variable].plot(ax=ax[1, 0], style='.', ms=1)
            ax[0, 0].set_ylabel('TGS 2611-C00 (V)')
            ax[1, 0].set_ylabel('TGS 2611-C00 (V)')

            ax[0, 0].lines[0].set_color('gray')
            ax[1, 0].lines[0].set_color('gray')
            ax[0, 0].lines[0].set_label('')

            ax[0, 0] = logger.format_ax0(ax[0, 0], time=True)
            ax[1, 0] = logger.format_ax0(ax[1, 0], time=True)

            fig.align_ylabels(ax[:, 0])
            plt.subplots_adjust(left=None, bottom=None,
                                right=None, top=None, wspace=None, hspace=0.4)
            plt.xticks(ha='center')


def plot_reconstruction(df1, df2, variables, start, end, logger, style='.-'):
    colors = ['r', 'tab:blue', 'tab:orange',
              'tab:green', 'tab:purple', 'tab:brown']
    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(
            20, 10), sharex=True, squeeze=False)
        ax[0, 0] = df1.loc[start:end, variables].plot(
            ax=ax[0, 0], style=style, ms=1)
        ax[1, 0] = df2.loc[start:end, variables].plot(
            ax=ax[1, 0], style=style, ms=1)
        ax[0, 0].set_ylabel('$\mathrm{CH_{4}}$ (ppm)')
        ax[1, 0].set_ylabel('$\mathrm{CH_{4}}$ (ppm)')

        for i, _ in enumerate(variables):
            ax[0, 0].lines[i].set_color(colors[i])
            ax[1, 0].lines[i].set_color(colors[i])

        ax[0, 0] = logger.format_ax0(ax[0, 0], time=True, date_format='1A')
        ax[1, 0] = logger.format_ax0(ax[1, 0], time=True, date_format='1A')

        handles, l = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, l, loc='lower center',
                   ncol=2, fontsize=14, markerscale=2.0)
        ax[0, 0].legend('')
        ax[1, 0].legend('')
        fig.align_ylabels(ax[:, 0])
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=None, hspace=0.2)
        # plt.xticks(ha='center')


def plot_reconstruction2(df1, df2, variables, start, end, style='o-', filename=None):
    colors = ['k', 'tab:blue', 'tab:orange',
              'tab:green', 'tab:purple', 'tab:brown']
    # df2b = df2.copy()
    # df2b.index = pd.to_timestamp(df2b.index.values)
    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(
            ncols=1, nrows=2, figsize=(20, 10), squeeze=False)
        ax[0, 0] = df1.loc[start:end, variables[0]].plot(
            ax=ax[0, 0], style=style, ms=8, rot=0, grid=False)
        ax[1, 0] = df2.loc[start:end, variables[0]].plot(
            ax=ax[1, 0], style=style, ms=8, rot=0, grid=False)
        ax[0, 0] = df1.loc[start:end, variables[1:]].plot(
            ax=ax[0, 0], style=style, ms=4, rot=0, grid=False)
        ax[1, 0] = df2.loc[start:end, variables[1:]].plot(
            ax=ax[1, 0], style=style, ms=4, rot=0, grid=False)
        ax[0, 0].set_ylabel('$\mathrm{CH_{4}}$ (ppm)')
        ax[1, 0].set_ylabel('$\mathrm{CH_{4}}$ (ppm)')

        for i, _ in enumerate(variables):
            ax[0, 0].lines[i].set_color(colors[i])
            ax[1, 0].lines[i].set_color(colors[i])

        ax[0, 0].xaxis.set_major_locator(
            mdates.MinuteLocator(byminute=range(0, 65, 5)))
        ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[1, 0].xaxis.set_major_locator(
            mdates.MinuteLocator(byminute=range(0, 60, 5)))
        ax[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax[0, 0].xaxis.label.set_size(16)
        ax[0, 0].yaxis.label.set_size(16)
        ax[1, 0].xaxis.label.set_size(16)
        ax[1, 0].yaxis.label.set_size(16)

        handles, l = ax[0, 0].get_legend_handles_labels()
        ax[0, 0].text(0.01, 1.00, '2 sec', transform=ax[0, 0].transAxes, fontsize=20, verticalalignment='top',
                      weight='bold')
        ax[1, 0].text(0.01, 1.00, '1 min', transform=ax[1, 0].transAxes, fontsize=20, verticalalignment='top',
                      weight='bold')
        fig.legend(handles, l, loc='lower center',
                   ncol=6, fontsize=20, markerscale=2.0)
        ax[0, 0].legend('')
        ax[1, 0].legend('')
        ax[0, 0].spines['left'].set_linewidth(2)
        ax[0, 0].spines['left'].set_color('gray')
        ax[0, 0].spines['bottom'].set_linewidth(2)
        ax[0, 0].spines['bottom'].set_color('gray')

        ax[1, 0].spines['left'].set_linewidth(2)
        ax[1, 0].spines['left'].set_color('gray')
        ax[1, 0].spines['bottom'].set_linewidth(2)
        ax[1, 0].spines['bottom'].set_color('gray')
        fig.suptitle(f'Release from {start} to {end}',
                     weight='bold', fontsize=26)
        fig.align_ylabels(ax[:, 0])
        plt.subplots_adjust(left=None, bottom=0.1, right=None,
                            top=0.95, wspace=None, hspace=0.15)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')


def spikes(logger, df, variable, binary, prom, window, it):
    df1 = logger.find_peaks_it(df.copy(
    ), variable=variable, prom=prom, window=window, it=it, baseline=False)  # 0.007
    df[binary] = df1.loc[:, 'Binary'].astype(int)
    plot_detection(df, logger=logger, variable=variable, binary=binary)
    return df


def plot_CV(wpath, tr, folder_path=('A', 'A1', 'C', '20'), filename=None):
    CV = pd.read_pickle(
        wpath + f'{folder_path[0]}/_save/{folder_path[1]}/{folder_path[1]}_{folder_path[2]}_Tr{tr}_Te{np.abs(100 - tr)}_S{folder_path[3]}/' + 'Roll_CV.pkl')
    with plt.style.context('seaborn-whitegrid'):
        colors = plt.cm.get_cmap('Set2')
        fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(
            16, 18), squeeze=False, sharex=True)
        ax[0, 0].axhline(y=0.2, color='k')
        ax[1, 0].axhline(y=0, color='k')
        ax[2, 0].axhline(y=100, color='k')
        ax[3, 0].axhline(y=1, color='k')
        ax[4, 0].axhline(y=1, color='k')

        # ax[0,0] = CV.loc[:,['MSD_Train', 'MSD_Test'  ]].plot(style='s', ax=ax[0,0], rot=0)
        ax[0, 0] = CV.loc[:, ['RMSE_Train', 'RMSE_Test']].plot(
            style='s', ax=ax[0, 0], rot=0, legend=False)
        ax[1, 0] = CV.loc[:, ['Bias_Train', 'Bias_Test']].plot(
            style='s', ax=ax[1, 0], rot=0, legend=False)
        ax[2, 0] = CV.loc[:, ['SD_Train', 'SD_Test']].plot(
            style='s', ax=ax[2, 0], rot=0, legend=False)
        ax[3, 0] = CV.loc[:, ['CORR_Train', 'CORR_Test']].plot(
            style='s', ax=ax[3, 0], rot=0, legend=False)
        ax[4, 0] = CV.loc[:, ['FoM_Train', 'FoM_Test']].plot(
            style='s', ax=ax[4, 0], rot=0, legend=False)

        for i, *_ in enumerate(ax):
            ax[i, 0].lines[1].set_color('r')
            ax[i, 0].lines[2].set_color('b')
            ax[i, 0].yaxis.set_major_locator(plt.MaxNLocator(5), )

        ax[0, 0].legend(['Target error (RMSE = 0.3 [ppm])', f'Train Error ({tr}% Obs.)',
                         f'Test Error ({np.abs(100 - tr)}% Obs.)'],
                        markerscale=3, prop={'size': 16}, loc='best', frameon=True, fancybox=True)
        ax[0, 0].set_ylabel('RMSE [$ppm$]', fontdict={'size': 16})
        ax[1, 0].set_ylabel('Bias [$ppm$]', fontdict={'size': 16})
        ax[2, 0].set_ylabel(
            '$\sigma_{Model}/\sigma_{Data}$ [%]', fontdict={'size': 16})
        ax[3, 0].set_ylabel(r'$\rho$', fontdict={'size': 16})
        ax[4, 0].set_ylabel('Figure Of Merit', fontdict={'size': 16})
        ax[-1, 0].set_xlabel('Index for the train & test period',
                             fontdict={'size': 16})

        ax[-1, 0].xaxis.set_major_locator(plt.MaxNLocator(20))
        ax[-1, 0].set_xticklabels([i for i in range(1, 21)])
        ax[-1, 0].set_xlim([1, 20])

        fig.align_ylabels(ax[:, 0])
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')


def target_diagram(models, label, limits=(-0.2, 0.2, 0.05), norm=True, taylor=False):
    lmin, lmax, step = limits
    if taylor:
        sdev = []
        crmsd = []
        ccoef = []
    else:
        bias = []
        crmsd = []
        rmsd = []

    for i, x in enumerate(models):
        x1 = pd.DataFrame()
        x1 = x.y_test.copy()
        x1['Model'] = x.y_test_pred.reshape(-1, 1)
        x1.columns = ['Reference', 'Model']
        if norm:
            x1['ReferenceN'] = x1['Reference'] / x1['Reference'].std()
            x1['ModelN'] = x1['Model'] / x1['Reference'].std()
        else:
            x1['ReferenceN'] = x1['Reference']
            x1['ModelN'] = x1['Model']
        if taylor:
            stats = sm.taylor_statistics(
                x1['ModelN'].values, x1['ReferenceN'].values, 'data')
            if i == 0:
                sdev.append(stats['sdev'][0])
                sdev.append(stats['sdev'][1])
                crmsd.append(stats['crmsd'][0])
                crmsd.append(stats['crmsd'][1])
                ccoef.append(stats['ccoef'][0])
                ccoef.append(stats['ccoef'][1])
            else:
                sdev.append(stats['sdev'][1])
                crmsd.append(stats['crmsd'][1])
                ccoef.append(stats['ccoef'][1])
        else:
            stats = sm.target_statistics(
                x1['ModelN'].values, x1['ReferenceN'].values, 'data')
            bias.append(stats['bias'])
            rmsd.append(stats['rmsd'])
            crmsd.append(stats['crmsd'])
    if taylor:
        sdev = np.array(sdev)
        crmsd = np.array(crmsd)
        ccoef = np.array(ccoef)
    else:
        bias = np.array(bias)
        crmsd = np.array(crmsd)
        rmsd = np.array(rmsd)

    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(
            14, 14), sharex=False, sharey=False, squeeze=False)
        if taylor:
            ax[0, 0] = sm.taylor_diagram(sdev, crmsd, ccoef, MarkerDisplayed='marker', markerLabel=label,
                                         markerLegend='on', checkStats='on',
                                         alpha=0.0, tickSTD=np.round(np.arange(0, 3 + 0.2, 0.2), 2), axismax=3,
                                         tickRMS=range(0, 4, 1),
                                         showlabelsSTD='on', titleSTD='on', widthRMS=2.0, titleRMS='on',
                                         titleRMSDangle=160, markerSize=10,
                                         titleOBS='REF', markerObs='o', titleCOR='on', styleOBS='-')
        else:
            ax[0, 0] = sm.target_diagram(bias, crmsd, rmsd, markerLabel=label, MarkerDisplayed='marker', colormap='on',
                                         ticks=np.round(
                                             np.arange(lmin, lmax + 0.05, step), 2),
                                         titleColorbar='RMSD', cmapzdata=rmsd, markerLegend='on', markerSize=14,
                                         equalAxes='on', axismax=lmax, circleLineSpec='--', circleLineWidth=1.5,
                                         overlay='off')
        plt.grid(b=None)


def get_true_y(df, yvars, tr, samples):
    ytr = pd.DataFrame()
    yte = pd.DataFrame()
    for i in range(0, samples, 1):
        TRAIN, TEST = split_dataset(df, train_test_ratio=np.abs(
            100 - tr) / 100, offset_data=0, sample=i, n_samples=samples)
        ytr.loc[:, str(i+1)] = TRAIN.loc[:, yvars].values
        yte.loc[:, str(i+1)] = TEST.loc[:, yvars].values
    return ytr, yte


def get_true_y_ix(df, yvars, tr, samples, ix):
    TRAIN, TEST = split_dataset(df, train_test_ratio=np.abs(
        100 - tr) / 100, offset_data=0, sample=ix, n_samples=samples)
    ytr = pd.DataFrame(TRAIN.loc[:, yvars].copy())
    yte = pd.DataFrame(TEST.loc[:, yvars].copy())
    ytr.columns = ['Reference']
    yte.columns = ['Reference']
    return ytr, yte


def target_diagram2(label, yvars, df, wpath, folder_path, tr, limitsRMS=(-0.2, 0.2, 0.05), limitsSTD=(-0.2, 0.2, 0.05), limitsTGT=(-0.2, 0.2, 0.05), norm=True, taylor=False, figsize=(12, 12)):
    CVtr = pd.read_pickle(wpath +
                          f'{folder_path[0]}/_save/{folder_path[1]}/{folder_path[1]}_{folder_path[2]}_Tr{tr}_Te{np.abs(100 - tr)}_S{folder_path[3]}/' +
                          'Roll_CV_train.pkl')
    CVte = pd.read_pickle(wpath +
                          f'{folder_path[0]}/_save/{folder_path[1]}/{folder_path[1]}_{folder_path[2]}_Tr{tr}_Te{np.abs(100 - tr)}_S{folder_path[3]}/' +
                          'Roll_CV_test.pkl')
    ytr, yte = get_true_y(df, yvars, tr, CVtr.shape[1])

    rlmin, rlmax, rstep = limitsRMS
    slmin, slmax, sstep = limitsSTD
    tlmin, tlmax, tstep = limitsTGT
    if taylor:
        sdevTr = []
        crmsdTr = []
        ccoefTr = []
        sdevTe = []
        crmsdTe = []
        ccoefTe = []
    else:
        biasTr = []
        crmsdTr = []
        rmsdTr = []
        biasTe = []
        crmsdTe = []
        rmsdTe = []

    modelsTr = []
    modelsTe = []
    for i in range(ytr.shape[1]):
        mm = pd.DataFrame()
        mm['Reference'] = ytr[str(i+1)].values
        mm['Model'] = CVtr[str(i + 1)].values
        modelsTr.append(mm)
        mm = pd.DataFrame()
        mm['Reference'] = yte[str(i + 1)].values
        mm['Model'] = CVte[str(i + 1)].values
        modelsTe.append(mm)

    for i in range(0, len(modelsTr)):
        xTr, xTe = modelsTr[i], modelsTe[i]
        if norm and taylor:
            xTr['ReferenceN'] = xTr['Reference'] / xTr['Reference'].std()
            xTr['ModelN'] = xTr['Model'] / xTr['Reference'].std()
            xTe['ReferenceN'] = xTe['Reference'] / xTe['Reference'].std()
            xTe['ModelN'] = xTe['Model'] / xTe['Reference'].std()
        else:
            xTr['ReferenceN'] = xTr['Reference']
            xTr['ModelN'] = xTr['Model']
            xTe['ReferenceN'] = xTe['Reference']
            xTe['ModelN'] = xTe['Model']
        if taylor:
            statsTr = sm.taylor_statistics(
                xTr['ModelN'].values, xTr['ReferenceN'].values, 'data')
            statsTe = sm.taylor_statistics(
                xTe['ModelN'].values, xTe['ReferenceN'].values, 'data')
            if i == 0:
                sdevTr.append(statsTr['sdev'][0])
                sdevTr.append(statsTr['sdev'][1])
                crmsdTr.append(statsTr['crmsd'][0])
                crmsdTr.append(statsTr['crmsd'][1])
                ccoefTr.append(statsTr['ccoef'][0])
                ccoefTr.append(statsTr['ccoef'][1])

                sdevTe.append(statsTe['sdev'][0])
                sdevTe.append(statsTe['sdev'][1])
                crmsdTe.append(statsTe['crmsd'][0])
                crmsdTe.append(statsTe['crmsd'][1])
                ccoefTe.append(statsTe['ccoef'][0])
                ccoefTe.append(statsTe['ccoef'][1])
            else:
                sdevTr.append(statsTr['sdev'][1])
                crmsdTr.append(statsTr['crmsd'][1])
                ccoefTr.append(statsTr['ccoef'][1])

                sdevTe.append(statsTe['sdev'][1])
                crmsdTe.append(statsTe['crmsd'][1])
                ccoefTe.append(statsTe['ccoef'][1])
        else:
            statsTr = sm.target_statistics(
                xTr['ModelN'].values, xTr['ReferenceN'].values, 'data', norm=norm)
            statsTe = sm.target_statistics(
                xTe['ModelN'].values, xTe['ReferenceN'].values, 'data', norm=norm)
            biasTr.append(statsTr['bias'])
            rmsdTr.append(statsTr['rmsd'])
            crmsdTr.append(statsTr['crmsd'])

            biasTe.append(statsTe['bias'])
            rmsdTe.append(statsTe['rmsd'])
            crmsdTe.append(statsTe['crmsd'])

    if taylor:
        sdevTr = np.array(sdevTr)
        crmsdTr = np.array(crmsdTr)
        ccoefTr = np.array(ccoefTr)

        sdevTe = np.array(sdevTe)
        crmsdTe = np.array(crmsdTe)
        ccoefTe = np.array(ccoefTe)
    else:
        biasTr = np.array(biasTr)
        crmsdTr = np.array(crmsdTr)
        rmsdTr = np.array(rmsdTr)

        biasTe = np.array(biasTe)
        crmsdTe = np.array(crmsdTe)
        rmsdTe = np.array(rmsdTe)

    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots(
            ncols=1, nrows=1, figsize=figsize, sharex=False, sharey=False, squeeze=False)
        plt.axes()
        if taylor:
            sm.taylor_diagram(sdevTr, crmsdTr, ccoefTr, checkStats='on', markercolor='r', alpha=0.0,
                              tickSTD=np.round(np.arange(slmin, slmax + sstep, sstep), 2), axismax=slmax,
                              tickRMS=np.round(
                                  np.arange(rlmin, rlmax + rstep, rstep), 2),
                              showlabelsSTD='on', titleSTD='on', widthRMS=2.0, titleRMS='off',
                              markerSize=16, titleOBS='REF', markerObs='o', titleCOR='on', styleOBS='-')

            sm.taylor_diagram(sdevTe, crmsdTe, ccoefTe, checkStats='on', markercolor='b', alpha=0.0,
                              tickSTD=np.round(np.arange(slmin, slmax + sstep, sstep), 2), axismax=slmax,
                              tickRMS=np.round(
                                  np.arange(rlmin, rlmax + rstep, rstep), 2),
                              showlabelsSTD='on', titleSTD='on', widthRMS=2.0, titleRMS='off',
                              markerSize=16, titleOBS='REF', markerObs='o', titleCOR='on', styleOBS='-',
                              overlay='on', markerLabel=label)
            plt.grid(b=None)
        else:
            sm.target_diagram(biasTr, crmsdTr, rmsdTr, markercolor='r', alpha=0.0,
                              ticks=np.round(
                                  np.arange(tlmin, tlmax + tstep, tstep), 2),
                              markerSize=14, equalAxes='on', axismax=tlmax, circleLineSpec='--',
                              circleLineWidth=1.5, overlay='on')
            sm.target_diagram(biasTe, crmsdTe, rmsdTe, markerLabel=label, markercolor='b', alpha=0.0,
                              ticks=np.round(
                                  np.arange(tlmin, tlmax + tstep, tstep), 2),
                              markerSize=14, equalAxes='on', axismax=tlmax, circleLineSpec='--',
                              circleLineWidth=1.5, overlay='off')
    fig.legend(markerscale=5)


def target_diagram3(label, yvars, df, wpath, folder_path, tr, limitsRMS=(-0.2, 0.2, 0.05), limitsSTD=(-0.2, 0.2, 0.05), limitsTGT=(-0.2, 0.2, 0.05), norm=True, taylor=False, figsize=(12, 12)):
    colors = ['b', 'orange', 'crimson', 'indigo',
              'slategray', 'teal', 'cornflowerblue', 'forestgreen']
    labels = dict(zip(label, colors))
    # with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize,
                           sharex=False, sharey=False, squeeze=False)
    for n, folder in enumerate(folder_path[1]):
        CVte = pd.read_pickle(wpath +
                              f'{folder_path[0]}/_save/{folder}/{folder_path[2][n]}_{folder_path[3]}_Tr{tr}_Te{np.abs(100 - tr)}_S{folder_path[4]}/' +
                              'Roll_CV_test.pkl')
        _, yte = get_true_y(df, yvars, tr, CVte.shape[1])
        # print(f'{folder_path[0]}/_save/{folder}/{folder_path[2][n]}_{folder_path[3]}_Tr{tr}_Te{np.abs(100 - tr)}_S{folder_path[4]}/')
        rlmin, rlmax, rstep = limitsRMS
        slmin, slmax, sstep = limitsSTD
        tlmin, tlmax, tstep = limitsTGT
        if taylor:
            sdevTe = []
            crmsdTe = []
            ccoefTe = []
        else:
            biasTe = []
            crmsdTe = []
            rmsdTe = []

        modelsTe = []
        for i in range(yte.shape[1]):
            mm = pd.DataFrame()
            mm['Reference'] = yte[str(i + 1)].values
            mm['Model'] = CVte[str(i + 1)].values
            modelsTe.append(mm)

        for i in range(0, len(modelsTe)):
            xTe = modelsTe[i]
            if norm and taylor:
                xTe['ReferenceN'] = xTe['Reference'] / xTe['Reference'].std()
                xTe['ModelN'] = xTe['Model'] / xTe['Reference'].std()
            else:
                xTe['ReferenceN'] = xTe['Reference']
                xTe['ModelN'] = xTe['Model']
            if taylor:
                statsTe = sm.taylor_statistics(
                    xTe['ModelN'].values, xTe['ReferenceN'].values, 'data')
                if i == 0:
                    sdevTe.append(statsTe['sdev'][0])
                    sdevTe.append(statsTe['sdev'][1])
                    crmsdTe.append(statsTe['crmsd'][0])
                    crmsdTe.append(statsTe['crmsd'][1])
                    ccoefTe.append(statsTe['ccoef'][0])
                    ccoefTe.append(statsTe['ccoef'][1])
                else:
                    sdevTe.append(statsTe['sdev'][1])
                    crmsdTe.append(statsTe['crmsd'][1])
                    ccoefTe.append(statsTe['ccoef'][1])
            else:
                statsTe = sm.target_statistics(
                    xTe['ModelN'].values, xTe['ReferenceN'].values, 'data', norm=norm)
                biasTe.append(statsTe['bias'])
                rmsdTe.append(statsTe['rmsd'])
                crmsdTe.append(statsTe['crmsd'])

        if taylor:
            sdevTe = np.array(sdevTe)
            crmsdTe = np.array(crmsdTe)
            ccoefTe = np.array(ccoefTe)
        else:
            biasTe = np.array(biasTe)
            crmsdTe = np.array(crmsdTe)
            rmsdTe = np.array(rmsdTe)

        if taylor:
            if n == len(folder_path[1])-1:
                sm.taylor_diagram(sdevTe, crmsdTe, ccoefTe, checkStats='on', markercolor=colors[n], alpha=0.0,
                                  tickSTD=np.round(np.arange(slmin, slmax + sstep, sstep), 2), axismax=slmax,
                                  tickRMS=np.round(
                                      np.arange(rlmin, rlmax + rstep, rstep), 2),
                                  showlabelsSTD='on', titleSTD='off', widthRMS=2.0, titleRMS='off',
                                  markerSize=20, titleOBS='REF', markerObs='o', titleCOR='off', styleOBS='-',
                                  colCOR='black', colRMS='black',  # 'mediumseagreen'
                                  overlay='on')
                plt.grid(b=None)
            elif n == 0:
                sm.taylor_diagram(sdevTe, crmsdTe, ccoefTe, checkStats='on', markercolor=colors[n], alpha=0.0,
                                  tickSTD=np.round(np.arange(slmin, slmax + sstep, sstep), 2), axismax=slmax,
                                  tickRMS=np.round(
                                      np.arange(rlmin, rlmax + rstep, rstep), 2),
                                  showlabelsSTD='off', widthRMS=2.0, titleRMS='off',
                                  markerSize=20, titleOBS='REF', markerObs='o', titleCOR='off', styleOBS='-',
                                  overlay='off', colCOR='black', colRMS='mediumseagreen')

            else:
                sm.taylor_diagram(sdevTe, crmsdTe, ccoefTe, checkStats='on', markercolor=colors[n], alpha=0.0,
                                  tickSTD=np.round(np.arange(slmin, slmax + sstep, sstep), 2), axismax=slmax,
                                  tickRMS=np.round(
                                      np.arange(rlmin, rlmax + rstep, rstep), 2),
                                  showlabelsSTD='on', titleSTD='off', widthRMS=2.0, titleRMS='off',
                                  markerSize=20, titleOBS='REF', markerObs='o', titleCOR='off', styleOBS='-',
                                  colCOR='black', colRMS='mediumseagreen',
                                  overlay='on', markerLabel=labels)
        else:
            if n == len(folder_path[1])-1:
                sm.target_diagram(biasTe, crmsdTe, rmsdTe, markerLabel=labels, markercolor=colors[n], alpha=0.0,
                                  ticks=np.round(
                                      np.arange(tlmin, tlmax + tstep, tstep), 2),
                                  markerSize=20, equalAxes='off', axismax=tlmax, circleLineSpec='--',
                                  circleLineWidth=1.5, overlay='on')
            elif n == 0:
                sm.target_diagram(biasTe, crmsdTe, rmsdTe, markerLabel=labels, markercolor=colors[n], alpha=0.0,
                                  ticks=np.round(
                                      np.arange(tlmin, tlmax + tstep, tstep), 2),
                                  markerSize=20, equalAxes='off', axismax=tlmax, circleLineSpec='--',
                                  circleLineWidth=1.5, overlay='off')
            else:
                sm.target_diagram(biasTe, crmsdTe, rmsdTe, markerLabel=labels, markercolor=colors[n], alpha=0.0,
                                  ticks=np.round(
                                      np.arange(tlmin, tlmax + tstep, tstep), 2),
                                  markerSize=20, equalAxes='off', axismax=tlmax, circleLineSpec='--',
                                  circleLineWidth=1.5, overlay='off')


if __name__ == '__main__':
    pass
