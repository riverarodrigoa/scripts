from __future__ import print_function, division, with_statement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.axislines import SubplotZero

import seaborn as sns
import string
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score
from sklearn import linear_model
from sklearn.metrics import r2_score as r2

from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

from scipy import stats
from scipy.signal import savgol_filter, medfilt, find_peaks
from tqdm import tqdm
from itertools import chain
import warnings
warnings.filterwarnings('ignore')


def initialize():

    # defining plt default params
    fsize = 15
    tsize = 14

    tdir = 'in'

    major = 5.0
    minor = 3.0
    lwidth = 2.0
    lhandle = 2.0

    style = 'default'

    plt.style.use(style)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['axes.linewidth'] = lwidth
    plt.rcParams['legend.handlelength'] = lhandle
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    colors = ['#ad3211', '#c46d21', '#cba553', '#d7d696',
              '#a4cda5', '#669393', '#3f5a75', '#292929']


def plot_comp_all_vars(da, vars_comp, start=None, end=None, qq=(0.0, 1.0), sec=None, ylabs=None,
                       legend_labs=None, bars=None, cmap=None,
                       ylims=None, mask_date=None, vline=None, vspan=None, file_name=None, figsize=(30, 23),
                       alpha=1.0, fontsize=16, interplotspace=(None, None), comp_in_subplot=False, cmp_colors=None,
                       reverse=(), k_ticks=None, style=None, grid_plot=True, marker_size=4, date_format=None, break_axes=None):
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    if start is None:
        start = da.index[0]
    if end is None:
        end = da.index[-1]

    if file_name is None:
        save = False
    else:
        save = True

    if sec is None:
        keys = None
    else:
        keys = list(sec.keys())

    if bars is None:
        bars = ['']
    else:
        bars_dict = bars
        bars = list(bars_dict.keys())

    if style is None:
        style = '.'

    if date_format is None:
        date_format = '1W'

    if date_format == 'W' or date_format == 'D':  # For compatibility with old notebooks
        date_format = '1'+date_format
    elif len(date_format) > 1:
        date_interval = int(date_format[0])
        date_type = date_format[1]
    else:
        date_type = date_format

    if break_axes is None:
        break_axes = [0]

    da = da.loc[start:end, :]
    d = da.copy()
    if k_ticks is not None:
        t_keys = list(k_ticks.keys())
        for i in t_keys:
            d.loc[:, i] = d.loc[:, i] / k_ticks[i]
    dqq = d.quantile(q=qq, axis=0)
    if cmap is None:
        cmap = 'Set2'
    n = len(ylabs)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n + 1))
    if comp_in_subplot:
        if cmp_colors is None:
            c = len(max(vars_comp))
            cmp_colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, c))
        else:
            pass

    with plt.style.context('seaborn-whitegrid'):
        if len(break_axes) > 1:
            fig, ax = plt.subplots(nrows=n, ncols=len(
                break_axes), sharey='row', sharex='col', figsize=figsize, squeeze=False)
            for i in range(0, len(ylabs)):
                for n, t in enumerate(break_axes):
                    j = n % len(break_axes)
                    # print(f'i:{i}, j:{j}, vars:{vars_comp[i]}, {t[0]}:{t[1]}')
                    ax[i, j] = d.loc[t[0]:t[1], vars_comp[i]].plot(ax=ax[i, j], style=style,
                                                                   grid=False, rot=0,
                                                                   ms=marker_size, alpha=alpha,
                                                                   x_compat=True)
                    if i == 0:
                        ax[i, j].lines[0].set_color('r')
                    else:
                        if i == 0:
                            ax[i, j].lines[0].set_color('r')
                        elif i == 1:
                            ax[i, j].lines[0].set_color('b')
                        elif i == 2:
                            ax[i, j].lines[0].set_color('g')
                        else:
                            ax[i, j].lines[0].set_color(colors[i])

                    if j == len(break_axes)-1:
                        if legend_labs is not None:
                            ax[i, j].legend(legend_labs[i], markerscale=3, prop={'size': fontsize},
                                            loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fancybox=True)
                        else:
                            ax[i, j].legend(markerscale=3, prop={'size': fontsize},
                                            loc='center left', bbox_to_anchor=(1, 0.5))
                    ax[i, j].set_xlabel('')
                    if j == 0:
                        ax[i, 0].set_ylabel(
                            ylabs[i], fontdict={'size': fontsize})
                        ax[i, 0].yaxis.set_major_locator(plt.MaxNLocator(5), )
                        ax[i, 0].ticklabel_format(
                            axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
                        ax[i, 0].yaxis.set_tick_params(labelsize=fontsize)
                    # else:
                    #    ax[i, j].yaxis.set_ticklabels([])
                    ax[i, j].xaxis.set_tick_params(
                        labelsize=fontsize, rotation=0, pad=0.01)

                    ax[i, 0].spines['left'].set_linewidth(2)
                    ax[i, 0].spines['left'].set_color('gray')
                    ax[i, j].spines['bottom'].set_linewidth(2)
                    ax[i, j].spines['bottom'].set_color('gray')
                    space_ba = 8
                    if j == len(break_axes)-1:
                        ax[i, j].spines['left'].set_visible(True)
                        ax[i, j].spines['left'].set_linestyle(
                            (0, (space_ba, space_ba)))
                        # ax[i, j].spines['left'].set_color('gray')
                    elif j == 0:
                        ax[i, j].spines['right'].set_visible(True)
                        ax[i, j].spines['right'].set_linestyle(
                            (0, (space_ba, space_ba)))
                        # ax[i, j].spines['right'].set_color('gray')
                    else:
                        ax[i, j].spines['left'].set_visible(True)
                        ax[i, j].spines['right'].set_visible(True)
                        ax[i, j].spines['left'].set_linestyle(
                            (0, (space_ba, space_ba)))
                        ax[i, j].spines['right'].set_linestyle(
                            (0, (space_ba, space_ba)))
                        # ax[i, j].spines['left'].set_color('gray')
                        # ax[i, j].spines['right'].set_color('gray')
                    bl = 0.01
                    kwargs = dict(
                        transform=ax[i, j].transAxes, color='gray', clip_on=False)
                    if j == 0:
                        ax[i, j].plot((1 - bl, 1 + bl), (-bl, +bl), **kwargs)
                        ax[i, j].plot((1 - bl, 1 + bl),
                                      (1 - bl, 1 + bl), **kwargs)
                    elif j == len(break_axes)-1:
                        ax[i, j].plot((-bl, +bl), (1 - bl, 1 + bl), **kwargs)
                        ax[i, j].plot((-bl, +bl), (-bl, +bl), **kwargs)
                    else:
                        ax[i, j].plot((1 - bl, 1 + bl), (-bl, +bl), **kwargs)
                        ax[i, j].plot((1 - bl, 1 + bl),
                                      (1 - bl, 1 + bl), **kwargs)
                        ax[i, j].plot((-bl, +bl), (1 - bl, 1 + bl), **kwargs)
                        ax[i, j].plot((-bl, +bl), (-bl, +bl), **kwargs)

                    if date_type == 'W':
                        locator = mdates.WeekdayLocator(
                            byweekday=0, interval=date_interval)
                        minlocator = mdates.DayLocator()
                    elif date_type == 'D':
                        locator = mdates.DayLocator(interval=date_interval)
                        minlocator = mdates.HourLocator()
                    elif date_type == 'A':
                        locator = mdates.AutoDateLocator(
                            minticks=1, maxticks=1)
                        minlocator = mdates.AutoDateLocator(
                            minticks=1, maxticks=2)
                    else:
                        locator = mdates.HourLocator(interval=20)
                        minlocator = mdates.MinuteLocator(interval=10)
                    # formatter = mdates.ConciseDateFormatter(locator)
                    # formatter.formats = ['%y', '%b', '%d-%b', '%H:%M', '%H:%M\n%d-%m', '%S.%2f']
                    # formatter.zero_formats = ['', '%y', '%d-%b', '%d-%b', '%H:%M', '%H:%M']
                    # formatter.offset_formats = ['', '', '', '', '', '']

                    ax[i, j].xaxis.set_major_locator(locator)
                    ff = '%H:%M\n%b-%d'
                    if t[2] == 1:
                        ax[i, j].xaxis.set_major_formatter(
                            mdates.DateFormatter(ff))
                    else:
                        ax[i, j].xaxis.set_major_formatter(
                            mdates.DateFormatter(ff))

                    ax[i, j].xaxis.set_minor_locator(minlocator)
                    ax[i, j].tick_params(which='minor', length=4, color='gray')
                    ax[i, j].tick_params(
                        which='major', length=8, color='gray', pad=0)
                    for tick in ax[i, j].xaxis.get_major_ticks():
                        tick.label1.set_horizontalalignment('center')
                    if i != len(ylabs) - 1:
                        ax[i, j].set_xticklabels('')

            plt.subplots_adjust(left=None, bottom=None, right=None,
                                top=None, wspace=0.4, hspace=interplotspace[1])
            # plt.xticks(ha='center')
            fig.align_ylabels(ax[:, 0])

        else:
            fig, ax = plt.subplots(nrows=n, sharex=True,
                                   figsize=figsize, squeeze=False)
            for i in range(0, n):
                if vars_comp[i][0] in bars:
                    ax[i, 0] = d.loc[:, vars_comp[i]].plot(ax=ax[i, 0],
                                                           style=style,
                                                           grid=grid_plot,
                                                           rot=0, ms=marker_size, alpha=alpha, x_compat=True)
                    binary_ix = bars_dict[vars_comp[i][0]]
                    d_line = d.loc[:, binary_ix]
                    d_bin = d[d.loc[:, binary_ix] == 1].dropna(how='all')
                    dd_bin = d.loc[d_bin.index, vars_comp[i]]
                    ax[i, 0] = dd_bin.plot(
                        style='x', ax=ax[i, 0], ms=marker_size)
                else:
                    if keys is not None:
                        if vars_comp[i][0] in keys:  # Secondary axes
                            ax[i, 0] = d.loc[:, vars_comp[i][0]].plot(ax=ax[i, 0],
                                                                      style=style, grid=grid_plot,
                                                                      rot=0, ms=marker_size, alpha=alpha, x_compat=True)
                            for j in range(1, len(sec[keys[0]])):
                                ax[i, 0] = d.loc[:, vars_comp[i][j]].plot(ax=ax[i, 0],
                                                                          secondary_y=True,
                                                                          style=style, grid=grid_plot,
                                                                          rot=0, ms=marker_size, alpha=alpha, x_compat=True)
                    else:
                        ax[i, 0] = d.loc[:, vars_comp[i]].plot(ax=ax[i, 0],
                                                               style=style,
                                                               grid=grid_plot,
                                                               rot=0, ms=marker_size, alpha=alpha, x_compat=True)

                if comp_in_subplot:
                    for k, color in enumerate(cmp_colors):
                        ax[i, 0].lines[k].set_color(color)
                elif vars_comp[i][0] in bars:
                    if i == 0:
                        ax[i, 0].lines[1].set_color('b')
                        ax[i, 0].lines[0].set_color('r')
                    else:
                        ax[i, 0].lines[1].set_color('gray')
                        ax[i, 0].lines[0].set_color(colors[i])
                else:
                    if i == 0:
                        ax[i, 0].lines[0].set_color('r')
                    elif i == 1:
                        ax[i, 0].lines[0].set_color('b')
                    elif i == 2:
                        ax[i, 0].lines[0].set_color('g')
                    else:
                        ax[i, 0].lines[0].set_color(colors[i])

                if legend_labs is not None:
                    ax[i, 0].legend(legend_labs[i], markerscale=3, prop={'size': fontsize},
                                    loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    ax[i, 0].legend(markerscale=3, prop={'size': fontsize},
                                    loc='center left', bbox_to_anchor=(1, 0.5))

                if ylims is not None:
                    print(ylims.keys(), i)
                    if i in ylims.keys():

                        a, b = ylims[i]
                    else:
                        a = dqq.loc[:, vars_comp[i]].values.min()
                        b = dqq.loc[:, vars_comp[i]].values.max()

                else:
                    a = dqq.loc[:, vars_comp[i]].values.min()
                    b = dqq.loc[:, vars_comp[i]].values.max()

                ax[i, 0].set_ylim(a, b)

                ax[i, 0].set_xlabel('')
                ax[i, 0].set_ylabel(ylabs[i], fontdict={'size': fontsize})
                ax[i, 0].yaxis.set_major_locator(plt.MaxNLocator(5), )
                ax[i, 0].ticklabel_format(
                    axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
                ax[i, 0].yaxis.set_tick_params(labelsize=fontsize)
                ax[i, 0].xaxis.set_tick_params(labelsize=fontsize, rotation=0)

                if date_type == 'W':
                    locator = mdates.WeekdayLocator(
                        byweekday=0, interval=date_interval)
                    minlocator = mdates.DayLocator()
                elif date_type == 'D':
                    locator = mdates.DayLocator(interval=date_interval)
                    minlocator = mdates.HourLocator()
                else:
                    locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
                    minlocator = mdates.AutoDateLocator(
                        minticks=5, maxticks=10)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats = ['%y', '%b', '%d-%b',
                                     '%H:%M:%S', '%H:%M:%S\n%b-%d',       '%S.%f']
                formatter.zero_formats = [
                    '', '%y', '%d-%b',    '%d-%b', '%H:%M:%S\n%b-%d', '%H:%M:%S']
                formatter.offset_formats = [
                    '',   '',      '',    '%b %Y',         '',        '%d %b %Y']

                ax[i, 0].xaxis.set_major_locator(locator)
                ax[i, 0].xaxis.set_major_formatter(formatter)

                ax[i, 0].xaxis.set_minor_locator(minlocator)
                ax[i, 0].tick_params(which='minor', length=4, color='k')
                ax[i, 0].tick_params(
                    which='major', length=8, color='k', pad=10)

                ax[i, 0].spines['left'].set_linewidth(2)
                ax[i, 0].spines['left'].set_color('gray')
                ax[i, 0].spines['bottom'].set_linewidth(2)
                ax[i, 0].spines['bottom'].set_color('gray')

                if len(reverse) != 0:
                    if reverse[i] in vars_comp[i]:
                        ax[i, 0].invert_yaxis()

                if vline is not None:
                    for l, k, a in vline:
                        if i == a:
                            ax[i, 0].axvline(x=l, color=k, linestyle='-')
                if vspan is not None:
                    for v, k, a in vspan:
                        if i == a:
                            ax[i, 0].axvspan(
                                v[0], v[1], facecolor=k, alpha=0.15)

                if i != n - 1:
                    ax[i, 0].set_xticklabels('')

                plt.xticks(ha='center')

                fig.align_ylabels(ax[:, 0])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=interplotspace[0], hspace=interplotspace[1])
    if save:
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0.1, dpi=300)


def calculate_partial_correlation(input_df):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables,
    controlling for all other remaining variables

    Parameters
    ----------
    input_df : array-like, shape (n, p)
        Array with the different variables. Each column is taken as a variable.

    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of input_df[:, i] and input_df[:, j]
        controlling for all other remaining variables.
    """
    partial_corr_matrix = np.zeros((input_df.shape[1], input_df.shape[1]))
    for i, column1 in enumerate(input_df):
        for j, column2 in enumerate(input_df):
            control_variables = np.delete(np.arange(input_df.shape[1]), [i, j])
            if i == j:
                partial_corr_matrix[i, j] = 1
                continue
            data_control_variable = input_df.iloc[:, control_variables]
            data_column1 = input_df[column1].values
            data_column2 = input_df[column2].values
            fit1 = linear_model.LinearRegression(fit_intercept=True)
            fit2 = linear_model.LinearRegression(fit_intercept=True)
            fit1.fit(data_control_variable, data_column1)
            fit2.fit(data_control_variable, data_column2)
            residual1 = data_column1 - \
                (np.dot(data_control_variable, fit1.coef_) + fit1.intercept_)
            residual2 = data_column2 - \
                (np.dot(data_control_variable, fit2.coef_) + fit2.intercept_)
            partial_corr_matrix[i, j] = stats.spearmanr(
                residual1, residual2)[0]  # pearsonr
    return pd.DataFrame(partial_corr_matrix, columns=input_df.columns, index=input_df.columns)


def get_period(x, ix, rate=0.3, offset=0, samples=50):
    n = len(ix)  # 100
    n_val = int(len(ix) * rate)  # 100*0.3= 30 len test set
    imin = offset  # 0
    imax = n - offset - n_val  # 100-0-30=70
    n_sample = (imax - imin) / samples  # (70-0)/2=35

    start = int(imin + x * n_sample)  # 0+0*35= 0
    end = int(start + n_val)  # 0+35=35

    return ix[start], ix[end]


def split_dataset(data, train_test_ratio, offset_data, sample, n_samples):
    start, end = get_period(x=sample, ix=data.index, rate=train_test_ratio,
                            offset=offset_data, samples=n_samples)  # Test set
    test_set = data.loc[start:end, :]
    train_set = data.loc[(data.index < start) | (data.index > end), :]
    return train_set, test_set


# Mean hourly
def msd_hourly(y_true, y_pred):
    yy_true = [np.mean(y_true[i:i+60]) for i in range(0, y_true.shape[0], 60)]
    yy_pred = [np.mean(y_pred[i:i+60]) for i in range(0, y_pred.shape[0], 60)]
    error = mse(yy_true, yy_pred)
    return error


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)


def get_reconstructed_ts(data, cv_tr, cv_te, i, yvar='CH4d_ppm', ix='Index', n_samples=50, train_test_ratio=0.3):
    train, test = split_dataset(
        data, train_test_ratio=train_test_ratio, offset_data=0, sample=i, n_samples=n_samples)
    yvars = [yvar]
    y = data.loc[:, yvars]
    y_train = train.loc[:, yvars]
    y_test = test.loc[:, yvars]

    y_tr = pd.DataFrame(cv_tr.loc[:, str(i+1)].values,
                        index=y_train.index, columns=[str(i+1)])
    y_te = pd.DataFrame(cv_te.loc[:, str(i+1)].values,
                        index=y_test.index, columns=[str(i+1)])

    y_all = y_tr.append(y_te)
    y_all.sort_values(by=ix, inplace=True)
    y_all['REF'] = y.values
    y_all.columns = ['Model', 'REF']
    return y_all, y_train, y_test, y_tr, y_te


def mean_by_time(x, time_str):
    # return x.groupby(pd.Grouper(freq=time_str)).mean()
    return x.resample(time_str).mean()


def h2o_mole_fraction(rh, t, p):
    a1 = (rh / 100) * np.exp(13.7 - 5120 / (t + 273.15))
    mf = 100 * (a1 / ((p / 100000) - a1))
    return mf


def h2o_mole_fraction2(data, relH, temp, press):
    rh = data.loc[:, relH].values
    t = data.loc[:, temp].values
    p = data.loc[:, press].values
    a1 = (rh / 100) * np.exp(13.7 - 5120 / (t + 273.15))
    mf = 100 * (a1 / ((p / 100000) - a1))
    return mf


def data_over_percentile(d, col_name='CH4_dry', percentile=0.9, labeled=False):
    qq = d.quantile(q=percentile, axis=0)
    if labeled:
        d_p = d.copy()
        d_p['Binary'] = d.loc[:, col_name] > qq.loc[col_name]
        d_p['Binary'] = d_p['Binary'].astype(int)
    else:
        d_p = d[d.loc[:, col_name] > qq.loc[col_name]]
        d_p.dropna(inplace=True)
    return d_p


# Detect spikes (sd over background)
def find_spikes(x, alpha, C_unf, n):  # Detect spikes in a window
    # n = 1
    sigma = np.std(x)
    # C_unf = x[0]
    threshold = C_unf + alpha * sigma + np.sqrt(n) * sigma
    spike = []
    # spike.append(0)
    for i in range(0, len(x)):
        if x[i] >= threshold:
            spike.append(1)
            n += 1
        else:
            spike.append(0)
            C_unf = x[i]
            n = 0
        threshold = C_unf + alpha * sigma + np.sqrt(n) * sigma
    return spike, C_unf, n


def find_spikes_onts(X, window, alpha):  # Detect spikes in a TS
    spike = pd.DataFrame(columns=['spikes'])
    for i in range(0, len(X), window):
        start = i
        end = start + window
        x = X[start:end]
        if i == 0:
            C_unf = x[0]
            n = 0
            spike['spikes'], C_unf_last, n_last = find_spikes(
                x, alpha=alpha, C_unf=C_unf, n=n)
            # print(C_unf_last, "\t", n_last)
        else:
            spike_t = pd.DataFrame(columns=['spikes'])
            spike_t['spikes'], C_unf_last, n_last = find_spikes(
                x, alpha=alpha, C_unf=C_unf_last, n=n_last)
            # print(C_unf_last, "\t", n_last)
            spike = spike.append(spike_t, ignore_index=True)
    return spike


def detect_spikes(x, window, alpha, backwards):  # Do the detection FWD & BCKWD
    spike_f = find_spikes_onts(x, window, alpha)
    if backwards:
        X_b = list(reversed(x))
        spike_b = find_spikes_onts(X_b, window, alpha)
        spike = pd.DataFrame(columns=['F', 'B', 'SS'])
        spike['F'] = spike_f.loc[:, 'spikes'].values
        spike['B'] = list(reversed(spike_b.loc[:, 'spikes'].values))
        spike.loc[spike['F'] != spike['B'], 'SS'] = 1
        ix = spike[spike['F'] == spike['B']].index
        spike.loc[ix, 'SS'] = spike.loc[ix, 'F']
    else:
        spike = pd.DataFrame(columns=['SS'])
        spike['SS'] = spike_f.loc[:, 'spikes'].values
    return spike


def spike_detection_it(data, variable, window, alpha, backwards):
    detection = dict()
    it = len(alpha)
    for i in tqdm(range(0, it)):
        if i == 0:
            data_ns = data
        else:
            data_ns = ddata[ddata.loc[:, 'Binary'] == 0].copy()
            del ddata, spike

        X_f = data_ns.loc[:, variable].copy()
        spike = detect_spikes(X_f, window[i], alpha[i], backwards)
        ddata = pd.DataFrame(columns=[variable, 'Binary'])
        ddata[variable] = data_ns.loc[:, variable]
        ddata.iloc[:, 1] = spike.loc[:, 'SS'].values
        detection[i] = ddata

    data_f = data.copy()
    for i in tqdm(range(0, it)):
        ix = detection[i][detection[i].loc[:, 'Binary'] == 1].index
        data_f.loc[ix, 'Binary'] = 1

    return ddata, detection, data_f


def find_peaks_it(df, variable, it, prom, window=100, baseline=True):
    dfs = pd.DataFrame(df.loc[:, variable])
    dfs['Binary'] = [False]*len(dfs)
    for i in range(0, it):
        dd = dfs[dfs.loc[:, 'Binary'] == False]
        y = dd.loc[:, variable].values
        spikes = find_peaks(y, prominence=prom, wlen=window)
        dd.iloc[spikes[0], 1] = True
        ix = dd[dd.loc[:, 'Binary'] == True].index
        dfs.loc[ix, 'Binary'] = True
    if not baseline:
        ix = dfs[dfs.loc[:, 'Binary'] == False].index
        dfs['Binary'] = [False]*len(dfs)
        dfs.loc[ix, 'Binary'] = True
    dfs['Binary'] = dfs.loc[:, 'Binary'].astype(int)
    # Sanity check
    binary = dfs['Binary'].values
    vprev = binary[0]
    for i in range(1, len(binary)-1):
        vactual = binary[i]
        vnext = binary[i+1]
        if vprev == 1 and vnext == 1 and vactual == 0:
            binary[i] = 1
        if vprev == 0 and vnext == 0 and vactual == 1:
            binary[i] = 0
    dfs['Binary'] = binary
    return dfs


def remove_baseline(x, variable, binary_ix, interpolation_method='pad', inverted=False):
    y = pd.DataFrame(columns=['Raw', 'Binary', 'Baseline'])
    y['Raw'] = x.loc[:, variable].copy()
    y['Baseline'] = y['Raw']
    y['Binary'] = binary_ix
    ix = y[y['Binary'] == 1].index
    y.loc[ix, 'Baseline'] = np.nan
    y['Interpolation'] = y.loc[:, 'Baseline'].interpolate(
        method=interpolation_method)
    if inverted:
        y['Corrected'] = y['Interpolation'] - y['Raw']
    else:
        y['Corrected'] = y['Raw'] - y['Interpolation']
    return y.loc[:, ['Interpolation', 'Corrected']]


def spike_change(time_ix, spike_ix):
    lst_change = []
    value_old = 0.0
    for ix, v in enumerate(spike_ix):
        if value_old != v:
            lst_change.append(time_ix[ix])
        value_old = v
    return lst_change


def align_ts(x, start, end, var1, var2):
    xpre = x.loc[:start, :]
    xx = x.loc[start:end, :]
    xaft = x.loc[end:, :]

    # Correct shift
    if xx.empty:
        return x
    else:
        m1 = xx.loc[:, [var1, var2]].idxmax()
        shift = m1[0] - m1[1]
        if shift < pd.Timedelta(0):
            shift = m1[1] - m1[0]
            xx.loc[:, var2] = xx.loc[:, var2].shift(-(shift // 5).seconds)
        else:
            xx.loc[:, var2] = xx.loc[:, var2].shift((shift // 5).seconds)

        # Reconstruct df
        xf = xpre.append(xx)
        xf = xf.append(xaft)
        return xf


def remove_noisy_baseline(df, variable, n, ratio, window):
    start = 0
    end = window
    remaining_obs = len(df)
    flag = []
    while remaining_obs > 0:
        remaining_obs -= window
        ######

        obs = pd.DataFrame(df.loc[:, variable].copy())
        obs = obs.iloc[start:end, :]
        obs['x'] = list(range(0, len(obs)))
        # obs.dropna(inplace=True)
        m_reg = make_pipeline(RobustScaler(quantile_range=(1.0, 99.0)),
                              linear_model.LinearRegression())
        m_reg.fit(obs.loc[:, 'x'].values.reshape(len(obs), 1),
                  obs.loc[:, variable].values.reshape(len(obs), 1))
        obs['y_pred'] = m_reg.predict(
            obs.loc[:, 'x'].values.reshape(len(obs), 1))
        # threshold = n*(obs['y_pred'].mean() + obs['y_pred'].std())
        threshold = n * (obs[variable].mean() + obs[variable].std())
        # ix1 = obs[obs[variable] > threshold].index
        ix0 = obs[obs[variable] <= threshold].index

        if len(ix0) > ratio * len(obs):
            obs['flag'] = 1
        else:
            obs['flag'] = 0
        # else:
        #     obs.loc[ix0,'flag'] = 0
        #     obs.loc[ix1,'flag'] = 1
        flag.append(obs['flag'].values)
        # if obs['y_pred'].mean() > n*obs['y_pred'].std(): # obs[variable].max() >= obs['y_pred_sd'].max()
        #    flag.append([1 for i in range(0,len(obs))])
        # else:
        #    flag.append([0 for i in range(0,len(obs))])

        #######
        start = end
        if remaining_obs < window:
            end += remaining_obs
        else:
            end += window
    flag = list(chain(*flag))
    df['Flag'] = flag
    return df


def delete_from_list(df, to_remove):
    for date in tqdm(to_remove):
        ix = df.loc[date[0]:date[1], :].index
        ixx = ~df.index.isin(ix)
        df = df.loc[df.index[ixx], :]
        print(len(df))
    return df


def obs_to_select(df, to_select):
    for n, date in enumerate(to_select):
        if n == 0:
            ddf = df.loc[date[0]:date[1], :].copy()
        else:
            ddf.append(df.loc[date[0]:date[1], :].copy())
    return ddf


def filter_releases(df, release, inside=True):
    ix = []
    for i in release:
        ix.append(df.loc[i[0]:i[1], :].index)
    ix = list(chain(*ix))
    if inside:
        return df.loc[ix, :]
    else:
        ixx = df.index[~df.index.isin(ix)]
        return df.loc[ixx, :]


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


if __name__ == '__main__':
    pass
