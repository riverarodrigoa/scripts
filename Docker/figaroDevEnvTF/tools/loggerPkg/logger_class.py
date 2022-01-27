from loggerPkg.utils import *
from loggerPkg.graphics import *
import os
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
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


# from dateutil import parser
# from pandas import DataFrame
# from pandas.io.parsers import TextFileReader

import warnings
import pickle

warnings.filterwarnings('ignore')


class Logger(object):
    def __init__(self, df, instrument_names, labels, variables=None, path=None, prefix='Logger'):
        self.df = df
        self.variables = dict()
        # Dict: [Key] = Value;  Key:Var_name;  Value:Legend_lab, ylabels
        if instrument_names is not None:
            for i, j in enumerate(list(self.df.columns)):
                self.variables[j] = [[instrument_names[i]], labels[i]]
        else:
            self.variables = variables

        self.variables_to_plot = []
        self.path = Path(os.getcwd()) if path is None else Path(path)
        self.prefix = prefix
        self.sufix = '.png'

        self.X_train = []
        self.X_test = []
        self.y_test = []
        self.y_train = []
        self.y_train_pred = []
        self.y_test_pred = []
        self.model = []
        self.tested = False

        self.count_figs = 0

        self.fontsize = 16

    def save_logger(self):
        wpath = self.path / self.prefix / '_save' / \
            str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        self.df.to_pickle(wpath/'df.pkl')

        with open(wpath / 'variables.pkl', 'wb+') as f:
            pickle.dump(self.variables, f, pickle.HIGHEST_PROTOCOL)
        print(f'[INFO] Logger saved succesfully on: {wpath}')

    @classmethod
    def load_logger(cls, savepath, path, prefix='Logger'):
        wpath = Path(savepath)
        df = pd.read_pickle(wpath / 'df.pkl')

        with open(wpath / 'variables.pkl', 'rb') as f:
            variables = pickle.load(f)

        instrument_n = [i[1][0][0] for i in variables.items()]
        labels = [i[1][1] for i in variables.items()]
        return cls(df=df, instrument_names=instrument_n, labels=labels, path=path, prefix=prefix)

    def copy(self, variables_to_copy=None, prefix=None, path=None):
        if variables_to_copy is not None:
            new_variables = {}
            for i in variables_to_copy:
                new_variables[i] = self.variables[i]

            dd = self.df.loc[:, variables_to_copy].copy()
        else:
            new_variables = self.variables
            dd = self.df.copy()

        if prefix is None:
            prefix = self.prefix

        if path is None:
            path = self.path
        return Logger(df=dd, instrument_names=None, variables=new_variables, labels=None, path=path, prefix=prefix)

    def rename_cols(self, new_cols):
        self.df.columns = new_cols
        keys = list(self.variables.keys())
        for i, j in enumerate(keys):
            self.variables[new_cols[i]] = self.variables.pop(j)

    def add_cols(self, other, newcols):
        for i in newcols:
            self.df[i[0]] = other.loc[:, i[0]]
            self.variables[i[0]] = [[i[1]], i[2]]

    def add_cols2(self, other, newcols):
        for i in newcols:
            self.df = self.df.join(other.loc[:, i[0]])
            self.variables[i[0]] = [[i[1]], i[2]]

    def remove_data_after_reboot(self, reboot_ix, time_to_remove):
        for i in reboot_ix:
            start = pd.to_datetime(i)
            end = start + pd.Timedelta(minutes=time_to_remove)
            r = pd.date_range(start, end, freq='5S')
            self.df.drop(r, errors='ignore', inplace=True)

    def find_discontinuities(self, min_gaps=15):
        gaps = []
        prev = self.df.index[0]
        for i in self.df.index[1:]:
            # print(f'[DEBUG] {i} - {prev} : {i-prev}, Threshold: {min_gaps} in time {pd.Timedelta(seconds=min_gaps)}')
            if i - prev >= pd.Timedelta(seconds=min_gaps):
                gaps.append([prev, i])
            prev = i
        return gaps

    def invert_signal_dates(self, gaps, freq='5s'):
        # da: Dataframe of gaps
        # dg: Dataframe of observations
        # dd : Dataframe of deltas(inverted)
        da = pd.DataFrame(gaps)
        da['diff'] = da[1] - da[0]
        da['diff_inv'] = da['diff'].values[::-1]

        dg = pd.DataFrame(gaps)
        dg.loc[len(dg), :] = [np.nan, np.nan]
        dg = dg.shift(1)
        dg.columns = ['end', 'start']
        dg.loc[0, 'start'] = self.df.index[0]
        dg.loc[len(dg), 'end'] = self.df.index[-1]
        dg.loc[:, 'start'] = dg.loc[:, 'start'].shift(1)
        dg.dropna(how='all', inplace=True)
        dg['delta'] = dg['end'] - dg['start']
        dg['delta_inv'] = dg['delta'].values[::-1]
        dg.index = [i for i in range(0, len(dg))]
        dd = pd.concat([dg['delta_inv'], da['diff_inv']], axis=1)

        ix = []
        start_index = self.df.index[0]
        print(dd.values.shape)
        for n, i in tqdm(enumerate(dd.values)):
            if n != 0:
                start_index = ix[-1] + i[1]
                ix = ix + list(pd.date_range(start=start_index,
                               end=start_index + i[0], freq=freq))
            else:
                ix = ix + list(pd.date_range(start=start_index,
                               end=start_index + i[0], freq=freq))
            # print(f'N: {n}, index: {ix}')
        return ix

    def v_to_r(self, variable, r_load=5000):
        vout = self.df.loc[:, variable].values
        rs = ((5*r_load)/vout) - r_load
        df = pd.DataFrame()
        df[variable+'_R'] = rs
        df.index = self.df.index
        self.add_cols(df, newcols=[
                      [variable+'_R', self.variables[variable][0][0], 'Resistance ($\mathrm{\Omega}$)']])

    def filter_data(self, variable, window, kernel, add=True, fix_values=False):
        df = pd.DataFrame()
        if fix_values:
            dvals = list(chain(*self.df.loc[:, variable].values))
        else:
            dvals = self.df.loc[:, variable].values
        df[variable+'_F'] = savgol_filter(dvals, window, 3)
        df[variable+'_F'] = medfilt(df.loc[:, variable+'_F'].values, kernel)
        if add:
            self.add_cols(df, newcols=[
                          [variable+'_F', self.variables[variable][0][0], self.variables[variable][1]]])
        else:
            return df[variable+'_F']

    def filter_threshold(self, variable, threshold, mode='outside', closed=True, save=False):
        df = pd.DataFrame()
        df[variable+'_FT'] = self.df.loc[:, variable].copy()
        df = pd.DataFrame(df)
        if closed:
            if len(threshold) > 1:
                if mode == "inside":
                    df = df[(threshold[1] <= df[variable+'_FT']) &
                            (df[variable+'_FT'] <= threshold[0])]
                elif mode == 'outside':
                    df = df[(threshold[0] <= df[variable+'_FT']) &
                            (df[variable+'_FT'] <= threshold[1])]
            else:
                if mode == 'inside':
                    df = df[df[variable+'_FT'] <= threshold[0]]
                elif mode == 'outside':
                    df = df[df[variable+'_FT'] >= threshold[0]]
        else:
            if len(threshold) > 1:
                if mode == "inside":
                    df = df[(threshold[1] < df[variable+'_FT']) &
                            (df[variable+'_FT'] < threshold[0])]
                elif mode == 'outside':
                    df = df[(threshold[0] < df[variable+'_FT']) &
                            (df[variable+'_FT'] < threshold[1])]
            else:
                if mode == 'inside':
                    df = df[df[variable+'_FT'] < threshold[0]]
                elif mode == 'outside':
                    df = df[df[variable+'_FT'] > threshold[0]]

        if save:
            self.add_cols(df, newcols=[
                          [variable+'_FT', self.variables[variable][0][0], self.variables[variable][1]]])
        else:
            return df

    def ewma(self, variable, alpha, ts):
        df = pd.DataFrame()
        df[variable +
            '_EWMA'] = self.df[variable].ewm(alpha=alpha, adjust=False).mean()
        df[variable + '_EWMA'] = df[variable+'_EWMA'].shift(-ts)
        df = pd.DataFrame(df)
        self.add_cols(df, newcols=[
                      [variable + '_EWMA', self.variables[variable][0][0]+' EWMA', self.variables[variable][1]]])

    def time_shift_correction(self, variable, lags):
        df = pd.DataFrame()
        df[variable + '_TS'] = self.df[variable].shift(lags)
        df = pd.DataFrame(df)
        self.add_cols(df, newcols=[
                      [variable + '_TS', self.variables[variable][0][0] + ' TS', self.variables[variable][1]]])

    def drift_time_correction(self, variable, variable_reference, window, n=10, timescale='s'):
        # defining time scale
        if timescale == 's':
            ts = 1
        elif timescale == 'm':
            ts = 12
        elif timescale == 'h':
            ts = 12*60
        else:
            ts = 1

        df = self.df.loc[:, [variable, variable_reference]].copy()
        ddf = pd.DataFrame()
        start = df.index[0]
        end = start + pd.Timedelta(hours=window)
        it = 1
        while end <= df.index[-1]:
            end = start + pd.Timedelta(hours=window)
            date_range = [i for i in df.index if start <= i < end]

            df1 = df.loc[date_range, :].copy()
            dcor = pd.DataFrame(columns=['Corr'])
            corrs = list(range(-n, n + 1))
            for i in corrs:
                df2 = df1.copy()
                df2[variable] = df2[variable].shift(i*ts)
                dcor.loc[i, 'Corr'] = np.abs(df2.corr().iloc[0, 1])
            ix = np.argmax(dcor)
            shifted_var = pd.DataFrame(df1[variable].shift(corrs[ix]).dropna())
            if it == 1:
                ddf = shifted_var
            else:
                ddf = ddf.append(shifted_var)
            # print(start, end, len(date_range), ddf.shape, it)
            start = end
            it += 1
        ddf.columns = [variable + '_DTC']
        ddf = pd.DataFrame(ddf)
        self.add_cols(ddf, newcols=[
                      [variable + '_DTC', self.variables[variable][0][0], self.variables[variable][1]]])

    def find_peaks_it(self, df=None, variable=None, it=1, prom=(None, 0.1), window=100, baseline=True):
        if df is None:
            df = self.df.copy()
        if variable is None:
            variable = list(self.variables.keys())[0]
        dfs = pd.DataFrame(df.loc[:, variable])
        dfs['Binary'] = [False] * len(dfs)
        for i in range(0, it):
            dd = dfs[dfs.loc[:, 'Binary'] == False]
            y = dd.loc[:, variable].values
            spikes = find_peaks(y, prominence=prom, wlen=window)
            dd.iloc[spikes[0], 1] = True
            ix = dd[dd.loc[:, 'Binary'] == True].index
            dfs.loc[ix, 'Binary'] = True
        if not baseline:
            ix = dfs[dfs.loc[:, 'Binary'] == False].index
            dfs['Binary'] = [False] * len(dfs)
            dfs.loc[ix, 'Binary'] = True
        dfs['Binary'] = dfs.loc[:, 'Binary'].astype(int)  # Sanity check
        # Sanity check
        binary = dfs['Binary'].values
        vprev = binary[0]
        # print('[DEBUG] sanity check')
        for i in range(1, len(binary) - 1):
            vactual = binary[i]
            vnext = binary[i + 1]
            if vprev == 1 and vnext == 1 and vactual == 0:
                binary[i] = 1
            if vprev == 0 and vnext == 0 and vactual == 1:
                binary[i] = 0
        dfs['Binary'] = binary
        return dfs

    @staticmethod
    def remove_baseline(x, variable, binary_ix, interpolation_method='pad', inverted=False, filter_int=None):
        y = pd.DataFrame(columns=['Raw', 'Binary', 'Baseline'])
        y['Raw'] = x.loc[:, variable].copy()
        y['Baseline'] = y['Raw']
        y['Binary'] = binary_ix
        ix = y[y['Binary'] == 1].index
        y.loc[ix, 'Baseline'] = np.nan
        y['Interpolation'] = y.loc[:, 'Baseline'].interpolate(
            method=interpolation_method)
        if filter_int is not None:
            y.dropna(how='any', inplace=True)
            y['Interpolation'] = savgol_filter(
                y['Interpolation'].values, filter_int[0], 3)
            y['Interpolation'] = medfilt(
                y['Interpolation'].values, filter_int[1])
        if inverted:
            y['Corrected'] = y['Interpolation'] - y['Raw']
        else:
            y['Corrected'] = y['Raw'] - y['Interpolation']
        return y.loc[:, ['Interpolation', 'Corrected']]

    def split_dataset_conf(self, xvars, yvars, train_test_ratio=0.5, shuffle=False, debug=0):
        # train_test_ratio: Test size
        ddf = self.df.loc[:, xvars+yvars]
        ddf.dropna(how='any', inplace=True)
        X = ddf.loc[:, xvars]
        y = ddf.loc[:, yvars]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=train_test_ratio, random_state=0, shuffle=shuffle)
        if debug == 1:
            print(self.X_train.shape, self.X_test.shape)

    def split_dataset_time(self, xvars, yvars, train_start, test_start, train_duration=1, test_duration=1, train_end=None, test_end=None):
        ddf = self.df.loc[:, xvars + yvars]
        ddf.dropna(how='any', inplace=True)
        if train_end is None:
            train_end = pd.to_datetime(
                train_start) + pd.Timedelta(days=train_duration)
        if test_end is None:
            test_end = pd.to_datetime(test_start) + \
                pd.Timedelta(days=test_duration)
        self.X_train = ddf.loc[train_start:train_end, xvars]
        self.y_train = ddf.loc[train_start:train_end, yvars]
        self.X_test = ddf.loc[test_start:test_end, xvars]
        self.y_test = ddf.loc[test_start:test_end, yvars]
        print(self.X_train.shape, self.X_test.shape,
              self.y_train.shape, self.y_test.shape)

    def split_dataset_time2(self, xvars, yvars, train_start, train_end):
        ddf = self.df.loc[:, xvars + yvars]
        ddf.dropna(how='any', inplace=True)

        self.X_train = ddf.loc[train_start:train_end, xvars]
        self.y_train = ddf.loc[train_start:train_end, yvars]
        ix = self.X_train.index
        ixx = ~ddf.index.isin(ix)
        self.X_test = ddf.loc[ixx, xvars]
        self.y_test = ddf.loc[ixx, yvars]
        print(self.X_train.shape, self.X_test.shape,
              self.y_train.shape, self.y_test.shape)

    def split_dataset_time3(self, xvars, yvars, test_start, test_end):
        ddf = self.df.loc[:, xvars + yvars]
        ddf.dropna(how='any', inplace=True)

        self.X_test = ddf.loc[test_start:test_end, xvars]
        self.y_test = ddf.loc[test_start:test_end, yvars]
        ix = self.X_test.index
        ixx = ~ddf.index.isin(ix)
        self.X_train = ddf.loc[ixx, xvars]
        self.y_train = ddf.loc[ixx, yvars]
        print(self.X_train.shape, self.X_test.shape,
              self.y_train.shape, self.y_test.shape)

    def train_model(self, model, return_model=False):
        self.model = model
        self.model.fit(self.X_train, self.y_train)
        self.tested = False
        if return_model:
            return self.model

    def rmse_by_delta(self, levels):

        train = self.y_train.copy()
        test = self.y_test.copy()

        xtr_pred = self.model.predict(self.X_train).flatten()
        xte_pred = self.model.predict(self.X_test).flatten()

        train['pred'] = xtr_pred
        test['pred'] = xte_pred

        rmse_delta = pd.DataFrame(
            columns=['RMSE_tr', 'R2_tr', 'RMSE_te', 'R2_te', 'count_tr', 'count_te'])
        for i, level in enumerate(levels):
            if level[0] is None:
                train_ss = train[train[self.y_train.columns[0]] <= level[1]]
                test_ss = test[test[self.y_test.columns[0]] <= level[1]]
            elif level[1] is None:
                train_ss = train[level[0] <= train[self.y_train.columns[0]]]
                test_ss = test[level[0] <= test[self.y_test.columns[0]]]
            else:
                train_ss = train[(level[0] < train[self.y_train.columns[0]]) & (
                    train[self.y_train.columns[0]] < level[1])]
                test_ss = test[(level[0] < test[self.y_test.columns[0]]) & (
                    test[self.y_test.columns[0]] < level[1])]

            count_tr = len(train_ss)
            count_te = len(test_ss)
            rmse_tr = np.sqrt(mse(
                y_true=train_ss[self.y_train.columns[0]].values, y_pred=train_ss['pred'].values))
            rmse_te = np.sqrt(
                mse(y_true=test_ss[self.y_test.columns[0]].values, y_pred=test_ss['pred'].values))
            r2_tr = r2(
                y_true=train_ss[self.y_train.columns[0]].values, y_pred=train_ss['pred'].values)
            r2_te = r2(
                y_true=test_ss[self.y_test.columns[0]].values, y_pred=test_ss['pred'].values)
            rmse_delta.loc[i, :] = [rmse_tr, r2_tr,
                                    rmse_te, r2_te, count_tr, count_te]
        return rmse_delta

    def test_model(self, scatter=True, dates_sample=None, save=False, train_or_test='11', style='o-', show_metrics=False, debug=0, plot=True, return_df=False):
        if not self.tested:
            if train_or_test == '11':
                self.y_train_pred = self.model.predict(self.X_train).flatten()
                self.y_test_pred = self.model.predict(self.X_test).flatten()
                self.tested = True
            elif train_or_test == '01':
                self.y_test_pred = self.model.predict(self.X_test).flatten()
                self.tested = True
        if plot:
            if train_or_test == '11':
                msd_train = mse(self.y_train.values, self.y_train_pred)
                msd_test = mse(self.y_test.values, self.y_test_pred)
                rmse_train = np.sqrt(msd_train)
                rmse_test = np.sqrt(msd_test)
                train_rp = pd.DataFrame(index=self.y_train.index, columns=[
                                        'Reference', 'Model'])
                test_rp = pd.DataFrame(index=self.y_test.index, columns=[
                                       'Reference', 'Model'])
                train_rp['Reference'] = self.y_train.values
                train_rp['Model'] = self.y_train_pred
                test_rp['Reference'] = self.y_test.values
                test_rp['Model'] = self.y_test_pred
                a = np.trapz(self.y_train_pred)
                b = np.trapz(self.y_train.values.reshape(
                    1, len(self.y_train))[0])
                c = np.abs(a - b)
                fom_train = a / (a + c)
                a = np.trapz(self.y_test_pred)
                b = np.trapz(self.y_test.values.reshape(
                    1, len(self.y_test))[0])
                c = np.abs(a - b)
                fom_test = a / (a + c)
            elif train_or_test == '01':
                msd_test = mse(self.y_test.values, self.y_test_pred)
                rmse_test = np.sqrt(msd_test)
                test_rp = pd.DataFrame(index=self.y_test.index, columns=[
                                       'Reference', 'Model'])
                test_rp['Reference'] = self.y_test.values
                test_rp['Model'] = self.y_test_pred
            # fom_train = np.mean(self.y_train.values/self.y_train_pred.reshape(len(self.y_train_pred), 1))
            # fom_test = np.mean(self.y_test.values / self.y_test_pred.reshape(len(self.y_test_pred), 1))
                a = np.trapz(self.y_test_pred)
                b = np.trapz(self.y_test.values.reshape(
                    1, len(self.y_test))[0])
                c = np.abs(a - b)
                fom_test = a / (a + c)

            if scatter:
                with plt.style.context('seaborn-whitegrid'):
                    if train_or_test == '11':
                        fig, ax = plt.subplots(
                            ncols=2, figsize=(20, 8), squeeze=False)
                        # ax[0, 0] = sns.histplot(train_rp, x='Model', y='Reference', bins=50, cbar=True, ax=ax[0, 0], palette='Pastel2')
                        # ax[0, 1] = sns.histplot(test_rp, x='Model', y='Reference', bins=50, cbar=True, ax=ax[0, 1], palette='Pastel2')
                        ax[0, 0] = train_rp.plot(
                            x='Model', y='Reference', ax=ax[0, 0], style='.',)
                        ax[0, 1] = test_rp.plot(
                            x='Model', y='Reference', ax=ax[0, 1], style='.')

                        ax[0, 0].set_xlabel('Model (ppm)')
                        ax[0, 0].set_ylabel('Reference (ppm)')
                        ax[0, 1].set_xlabel('Model (ppm)')
                        ax[0, 1].set_ylabel('Reference (ppm)')

                        ax[0, 0].legend([f'RMSE: {round(rmse_train, 3)} (ppm)\nFoM: {round(fom_train, 3)}'], markerscale=1, prop={
                                        'size': self.fontsize}, loc='best', frameon=True, fancybox=True)
                        ax[0, 1].legend([f'RMSE: {round(rmse_test , 3)} (ppm)\nFoM: {round(fom_test, 3)}'], markerscale=1, prop={
                                        'size': self.fontsize}, loc='best', frameon=True, fancybox=True)

                        ax[0, 0].set_title(
                            'Train set', {'fontsize': '20', 'fontweight': 'bold'})
                        ax[0, 1].set_title(
                            'Test set', {'fontsize': '20', 'fontweight': 'bold'})
                        tr_mx = int(max(train_rp.max()))
                        te_mx = int(max(test_rp.max()))
                        tr_mi = int(min(train_rp.min()))
                        te_mi = int(min(test_rp.min()))
                        ax[0, 0].plot(np.linspace(tr_mi, tr_mx, 1000),
                                      np.linspace(tr_mi, tr_mx, 1000), 'r')
                        ax[0, 1].plot(np.linspace(te_mi, te_mx, 1000),
                                      np.linspace(te_mi, te_mx, 1000), 'r')
                        ax[0, 0].set_xlim([tr_mi, tr_mx])
                        ax[0, 0].set_ylim([tr_mi, tr_mx])
                        ax[0, 1].set_xlim([te_mi, te_mx])
                        ax[0, 1].set_ylim([te_mi, te_mx])

                        if save:
                            plt.savefig(self.figure_path(),
                                        dpi=300, bbox_inches='tight')
                    elif train_or_test == '01':
                        fig, ax = plt.subplots(
                            ncols=1, figsize=(12, 8), squeeze=False)
                        # ax[0, 0] = sns.histplot(train_rp, x='Model', y='Reference', bins=50, cbar=True, ax=ax[0, 0], palette='Pastel2')
                        # ax[0, 1] = sns.histplot(test_rp, x='Model', y='Reference', bins=50, cbar=True, ax=ax[0, 1], palette='Pastel2')

                        ax[0, 0] = test_rp.plot(
                            x='Model', y='Reference', ax=ax[0, 0], style='.')

                        ax[0, 0].set_xlabel('Model (ppm)')
                        ax[0, 0].set_ylabel('Reference (ppm)')

                        ax[0, 0].legend([f'RMSE: {round(rmse_test, 3)} (ppm)'],
                                        markerscale=1, prop={'size': self.fontsize}, loc='best', frameon=True,
                                        fancybox=True)

                        ax[0, 0].set_title(
                            'Test set', {'fontsize': '20', 'fontweight': 'bold'})
                        te_mx = int(max(test_rp.max()))
                        te_mi = int(min(test_rp.min()))
                        ax[0, 0].plot(np.linspace(te_mi, te_mx, 1000),
                                      np.linspace(te_mi, te_mx, 1000), 'r')
                        ax[0, 0].set_xlim([te_mi, te_mx])
                        ax[0, 0].set_ylim([te_mi, te_mx])

                        if save:
                            plt.savefig(self.figure_path(),
                                        dpi=300, bbox_inches='tight')
            else:
                yy_train = self.y_train.copy()
                yy_train['Model'] = self.y_train_pred
                yy_train.columns = ['Reference', 'Model']
                yy_test = self.y_test.copy()
                yy_test['Model'] = self.y_test_pred
                yy_test.columns = ['Reference', 'Model']
                in_metric = [f'RMSE: {round(rmse_train, 3)} (ppm)\nFoM: {round(fom_train, 3)}',
                             f'RMSE: {round(rmse_test , 3)} (ppm)\nFoM: {round(fom_test, 3)}']

                if dates_sample is not None:
                    if 'train' in list(dates_sample.keys()):
                        tr_start, tr_end = dates_sample['train']
                        d0 = yy_train.loc[tr_start:tr_end, :]
                    else:
                        d0 = yy_train
                    if 'test' in list(dates_sample.keys()):
                        te_start, te_end = dates_sample['test']
                        d1 = yy_test.loc[te_start:te_end, :]
                    else:
                        d1 = yy_test
                else:
                    d0 = yy_train
                    d1 = yy_test

                props = dict(boxstyle='round', alpha=0.5)
                with plt.style.context('seaborn-whitegrid'):
                    if train_or_test == '11':
                        fig, ax = plt.subplots(
                            nrows=2, ncols=1, sharex=False, sharey=False, figsize=(20, 8), squeeze=False)
                        ax[0, 0] = d0.plot(
                            ax=ax[0, 0], style=style, grid=True, rot=0, ms=5, legend=False, x_compat=True)
                        ax[1, 0] = d1.plot(
                            ax=ax[1, 0], style=style, grid=True, rot=0, ms=5, legend=False, x_compat=True)

                        ax[0, 0].lines[0].set_color('r')
                        ax[0, 0].lines[1].set_color('b')

                        ax[1, 0].lines[0].set_color('r')
                        ax[1, 0].lines[1].set_color('b')

                        ax[0, 0] = set_ax_conf(
                            ax[0, 0], leg=None, ylabl="$\mathrm{CH_{4}}$ (ppm)", fontsize=14, loc='A')
                        ax[1, 0] = set_ax_conf(
                            ax[1, 0], leg=None, ylabl="$\mathrm{CH_{4}}$ (ppm)", fontsize=14, loc='A')
                        if show_metrics:
                            ax[0, 0].text(0.01, 0.98, msd_train, transform=ax[0, 0].transAxes,
                                          fontsize=15, verticalalignment='top', bbox=props)
                            ax[1, 0].text(0.01, 0.98, msd_test, transform=ax[1, 0].transAxes,
                                          fontsize=15, verticalalignment='top', bbox=props)

                        ax[0, 0].text(0.01, 1.00, 'Train Set', transform=ax[0, 0].transAxes,
                                      fontsize=18, verticalalignment='top', weight='bold')
                        ax[1, 0].text(0.01, 1.00, 'Test Set', transform=ax[1, 0].transAxes,
                                      fontsize=18, verticalalignment='top', weight='bold')
                    if train_or_test == '10':
                        fig, ax = plt.subplots(
                            nrows=1, ncols=1, sharex=False, sharey=False, figsize=(20, 8), squeeze=False)
                        ax[0, 0] = d0.plot(
                            ax=ax[0, 0], style=style, grid=True, rot=0, ms=5, legend=False, x_compat=True)
                        ax[0, 0].lines[0].set_color('r')
                        ax[0, 0].lines[1].set_color('b')
                        ax[0, 0] = set_ax_conf(
                            ax[0, 0], leg=None, ylabl="$\mathrm{CH_{4}}$ (ppm)", fontsize=14, loc='A')
                        if show_metrics:
                            ax[0, 0].text(0.01, 0.98, msd_train, transform=ax[0, 0].transAxes,
                                          fontsize=15, verticalalignment='top', bbox=props)
                        ax[0, 0].text(0.01, 1.00, 'Train Set', transform=ax[0, 0].transAxes,
                                      fontsize=18, verticalalignment='top', weight='bold')
                    if train_or_test == '01':
                        fig, ax = plt.subplots(
                            nrows=1, ncols=1, sharex=False, sharey=False, figsize=(20, 8), squeeze=False)
                        ax[0, 0] = d1.plot(
                            ax=ax[0, 0], style=style, grid=True, rot=0, ms=5, legend=False, x_compat=True)
                        ax[0, 0].lines[0].set_color('r')
                        ax[0, 0].lines[1].set_color('b')
                        ax[0, 0] = set_ax_conf(
                            ax[0, 0], leg=None, ylabl="$\mathrm{CH_{4}}$ (ppm)", fontsize=18, loc='A')
                        if show_metrics:
                            ax[0, 0].text(0.01, 0.98, msd_test, transform=ax[0, 0].transAxes,
                                          fontsize=15, verticalalignment='top', bbox=props)
                        ax[0, 0].text(0.01, 1.00, 'Test Set', transform=ax[0, 0].transAxes,
                                      fontsize=18, verticalalignment='top', weight='bold')

                    handles, _ = ax[0, 0].get_legend_handles_labels()
                    fig.legend(handles, [
                               'Reference', 'Model'], loc='lower center', ncol=2, fontsize=14, markerscale=2.0)
                    plt.xticks(ha='center')
                    plt.subplots_adjust(
                        left=None, bottom=0.15, right=None, top=None, wspace=0.2, hspace=0.4)
                if save:
                    fig.savefig(self.figure_path(),
                                bbox_inches='tight', pad_inches=0.1, dpi=300)

        if return_df:
            if train_or_test == '11':
                return train_rp, test_rp
            elif train_or_test == '01':
                return test_rp

    def cross_valid(self, model, xvars, yvars, samples, ratio, save=False, prefix=None):
        # ratio: Test set length (0.0 - 1.0)
        ddf = self.df.loc[:, xvars + yvars]
        ddf.dropna(how='any', inplace=True)

        # X = ddf.loc[:, xvars]
        # y = ddf.loc[:, yvars]

        CV = pd.DataFrame(columns=['MSD_Train', 'MSD_Test',
                                   'RMSE_Train', 'RMSE_Test',
                                   'Bias_Train', 'Bias_Test',
                                   'SD_Train', 'SD_Test',
                                   'CORR_Train', 'CORR_Test',
                                   'SD_delta_Train', 'SD_delta_Test',
                                   'FoM_Train', 'FoM_Test'])
        CV_P_tr = pd.DataFrame(columns=[str(i) for i in range(1, samples+1)])
        CV_P_te = pd.DataFrame(columns=[str(i) for i in range(1, samples+1)])
        print('[INFO] Start CV.')
        for i in range(0, samples):
            print(f'[INFO] CV (Iteration: {i+1}/{samples}).')
            TRAIN, TEST = split_dataset(
                ddf, train_test_ratio=ratio, offset_data=0, sample=i, n_samples=samples)
            # train_test_ratio : Test set size
            # X = ddf.loc[:, xvars]
            # y = ddf.loc[:, yvars]
            X_train = TRAIN.loc[:, xvars]
            y_train = TRAIN.loc[:, yvars]
            X_test = TEST.loc[:, xvars]
            y_test = TEST.loc[:, yvars]

            model.fit(X_train, y_train)

            xtr_pred = model.predict(X_train).flatten()
            xte_pred = model.predict(X_test).flatten()

            msd_train = mse(y_train.values, xtr_pred)
            msd_test = mse(y_test.values, xte_pred)
            rmse_train = np.sqrt(msd_train)
            rmse_test = np.sqrt(msd_test)
            bias_train = np.mean(xtr_pred.reshape(
                len(xtr_pred), 1) - y_train.values)
            bias_test = np.mean(xte_pred.reshape(
                len(xte_pred), 1) - y_test.values)
            sd_train = 100 * (np.std(xtr_pred) / np.std(y_train.values))
            sd_test = 100 * (np.std(xte_pred) / np.std(y_test.values))
            corr_train = np.corrcoef(
                xtr_pred, y_train.values.reshape(len(y_train.values), ))[0, 1]
            corr_test = np.corrcoef(
                xte_pred, y_test.values.reshape(len(y_test.values), ))[0, 1]
            CV_P_tr.loc[:, str(i + 1)] = xtr_pred
            CV_P_te.loc[:, str(i + 1)] = xte_pred
            sd_delta_train = np.std(xtr_pred.reshape(
                len(xtr_pred), 1) - y_train.values)
            sd_delta_test = np.std(xte_pred.reshape(
                len(xte_pred), 1) - y_test.values)
            fom_train = np.trapz(
                xtr_pred) / np.trapz(y_train.values.reshape(1, len(y_train))[0])
            fom_test = np.trapz(
                xte_pred) / np.trapz(y_test.values.reshape(1, len(y_test))[0])

            CV.loc[i + 1, :] = [msd_train, msd_test,
                                rmse_train, rmse_test,
                                bias_train, bias_test,
                                sd_train, sd_test,
                                corr_train, corr_test,
                                sd_delta_train, sd_delta_test,
                                fom_train, fom_test]
        print('[INFO] End CV.')
        if save:
            if prefix is None:
                name = 'CV'
                wpath = self.path / self.prefix / '_save' / name / str(
                    datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            else:
                name = prefix
                wpath = self.path / self.prefix / '_save' / name
            if not os.path.exists(wpath):
                os.makedirs(wpath)
            print('[INFO] Saving results.')
            CV.to_pickle(path=wpath / 'Roll_CV.pkl')
            CV_P_tr.to_pickle(path=wpath / 'Roll_CV_train.pkl')
            CV_P_te.to_pickle(path=wpath / 'Roll_CV_test.pkl')
            print(f'[INFO] Results saved in {wpath}')

    def cross_valid_hybrid(self, model1, model2, xvars, yvars, samples, ratio, thresholdRatio, save=False, prefix=None):
        # ratio: Test set length (0.0 - 1.0)
        ddf = self.df.loc[:, xvars + yvars]
        ddf.dropna(how='any', inplace=True)
        # X = ddf.loc[:, xvars]
        # y = ddf.loc[:, yvars]
        CV = pd.DataFrame(columns=['MSD_Train', 'MSD_Test',
                                   'RMSE_Train', 'RMSE_Test',
                                   'Bias_Train', 'Bias_Test',
                                   'SD_Train', 'SD_Test',
                                   'CORR_Train', 'CORR_Test',
                                   'SD_delta_Train', 'SD_delta_Test',
                                   'FoM_Train', 'FoM_Test'
                                   ])

        CV_P_tr = pd.DataFrame(columns=[str(i) for i in range(1, samples + 1)])
        CV_P_te = pd.DataFrame(columns=[str(i) for i in range(1, samples + 1)])

        # CV1 = CV.copy()
        # CV2 = CV.copy()
        # CV1_P_tr = CV_P_tr.copy()
        # CV1_P_te = CV_P_te.copy()
        # CV2_P_tr = CV_P_tr.copy()
        # CV2_P_te = CV_P_te.copy()
        print('[INFO] Start CV.')
        for i in range(0, samples):
            print(f'[INFO] CV (Iteration: {i + 1}/{samples}).')
            TRAIN, TEST = split_dataset(
                ddf, train_test_ratio=ratio, offset_data=0, sample=i, n_samples=samples)
            # train_test_ratio : Test set size
            #dqq = d.quantile(q=qq, axis=0)
            threshold = TRAIN.loc[:, yvars].max()[0]*thresholdRatio

            ixTr1 = TRAIN[TRAIN.loc[:, yvars] <
                          threshold].loc[:, yvars].dropna().index
            ixTr2 = TRAIN[TRAIN.loc[:, yvars] >=
                          threshold].loc[:, yvars].dropna().index
            ixTe1 = TEST[TEST[yvars] < threshold].loc[:, yvars].dropna().index
            ixTe2 = TEST[TEST[yvars] >= threshold].loc[:, yvars].dropna().index
            # print(len(ixTe2))
            X_trainM1 = TRAIN.loc[ixTr1, xvars]
            y_trainM1 = TRAIN.loc[ixTr1, yvars]
            X_testM1 = TEST.loc[ixTe1, xvars]
            y_testM1 = TEST.loc[ixTe1, yvars]

            X_trainM2 = TRAIN.loc[ixTr2, xvars]
            y_trainM2 = TRAIN.loc[ixTr2, yvars]
            if len(ixTe2) > 0:
                X_testM2 = TEST.loc[ixTe2, xvars]
                y_testM2 = TEST.loc[ixTe2, yvars]

            model1.fit(X_trainM1, y_trainM1)
            model2.fit(X_trainM2, y_trainM2)

            xtr_predM1 = model1.predict(X_trainM1).flatten()
            xte_predM1 = model1.predict(X_testM1).flatten()
            xtr_predM2 = model2.predict(X_trainM2).flatten()
            if len(ixTe2) > 0:
                xte_predM2 = model2.predict(X_testM2).flatten()

            ytr1 = y_trainM1.copy()
            ytr1['Model'] = xtr_predM1
            yte1 = y_testM1.copy()
            yte1['Model'] = xte_predM1
            ytr2 = y_trainM2.copy()
            ytr2['Model'] = xtr_predM2
            if len(ixTe2) > 0:
                yte2 = y_testM2.copy()
                yte2['Model'] = xte_predM2

            ytr = ytr1.copy()
            ytr = ytr.append(ytr2).sort_index()
            yte = yte1.copy()
            if len(ixTe2) > 0:
                yte = yte.append(yte2).sort_index()
            ytr.columns = ['Reference', 'Model']
            yte.columns = ['Reference', 'Model']
            # print(ytr, yte)
            #
            CV_P_tr.loc[:, str(i + 1)] = ytr['Model'].values
            CV_P_te.loc[:, str(i + 1)] = yte['Model'].values

            yTrR = ytr['Reference'].values.reshape(-1, 1)
            yTrM = ytr['Model'].values.reshape(-1, 1)
            yTeR = yte['Reference'].values.reshape(-1, 1)
            yTeM = yte['Model'].values.reshape(-1, 1)

            CV.loc[i + 1, :] = computeStats(yTrR, yTeR, yTrM, yTeM)

        print('[INFO] End CV.')
        if save:
            if prefix is None:
                name = 'CV'
                wpath = self.path / self.prefix / '_save' / name / str(
                    datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            else:
                name = prefix
                wpath = self.path / self.prefix / '_save' / name
            if not os.path.exists(wpath):
                os.makedirs(wpath)
            print('[INFO] Saving results...')
            CV.to_pickle(path=wpath / 'Roll_CV.pkl')
            CV_P_tr.to_pickle(path=wpath / 'Roll_CV_train.pkl')
            CV_P_te.to_pickle(path=wpath / 'Roll_CV_test.pkl')

            # CV1.to_pickle(path=wpath / 'Roll_CV1.pkl')
            # CV1_P_tr.to_pickle(path=wpath / 'Roll_CV1_train.pkl')
            # CV1_P_te.to_pickle(path=wpath / 'Roll_CV1_test.pkl')
            # CV2.to_pickle(path=wpath / 'Roll_CV2.pkl')
            # CV2_P_tr.to_pickle(path=wpath / 'Roll_CV2_train.pkl')
            # CV2_P_te.to_pickle(path=wpath / 'Roll_CV2_test.pkl')
            print(f'[INFO] Results saved in {wpath}')

    def cross_validation(self, model, xvars, yvars, train_length, test_length, mode='1', save=True, debug=False):
        ddf = self.df.loc[:, xvars + yvars]
        ddf.dropna(how='any', inplace=True)

        CV = pd.DataFrame(columns=['RMSE_Train', 'RMSE_Test', 'Bias_Train', 'Bias_Test',
                                   'SD_Train', 'SD_Test', 'CORR_Train', 'CORR_Test',
                                   'SD_delta_Train', 'SD_delta_Test', 'fom_Train', 'fom_Test', 'N_train', 'N_Test'])
        if mode == '1':
            train_start = 0
            train_end = train_start + train_length
            test_start = train_length + 1
            test_end = test_start + test_length

            dtrain = ddf.iloc[train_start:train_end, :]
            X_train = dtrain.loc[:, xvars]
            y_train = dtrain.loc[:, yvars]
            # Train model
            model.fit(X_train, y_train)
            # Train prediction
            xtr_pred = model.predict(X_train).flatten()
            # Train metrics
            msd_train = mse(y_train.values, xtr_pred)
            rmse_train = np.sqrt(msd_train)
            bias_train = np.mean(xtr_pred.reshape(
                len(xtr_pred), 1) - y_train.values)
            sd_train = 100 * (np.std(xtr_pred) / np.std(y_train.values))
            corr_train = np.corrcoef(
                xtr_pred, y_train.values.reshape(len(y_train.values), ))[0, 1]
            sd_delta_train = np.std(xtr_pred.reshape(
                len(xtr_pred), 1) - y_train.values)
            fom_train = np.trapz(
                xtr_pred) / np.trapz(y_train.values.reshape(1, len(y_train))[0])
            ntrain = len(y_train)

            remaining_obs = len(ddf) - train_length
            c = 0
            while remaining_obs > 0:
                remaining_obs -= test_length
                ##
                dtest = ddf.iloc[test_start:test_end, :]
                X_test = dtest.loc[:, xvars]
                y_test = dtest.loc[:, yvars]
                xte_pred = model.predict(X_test).flatten()
                msd_test = mse(y_test.values, xte_pred)
                rmse_test = np.sqrt(msd_test)
                bias_test = np.mean(xte_pred.reshape(
                    len(xte_pred), 1) - y_test.values)
                sd_test = 100 * (np.std(xte_pred) / np.std(y_test.values))
                corr_test = np.corrcoef(
                    xte_pred, y_test.values.reshape(len(y_test.values), ))[0, 1]
                sd_delta_test = np.std(xte_pred.reshape(
                    len(xte_pred), 1) - y_test.values)
                fom_test = np.trapz(
                    xte_pred) / np.trapz(y_test.values.reshape(1, len(y_test))[0])
                ntest = len(y_test)
                CV.loc[c + 1, :] = [rmse_train, rmse_test,
                                    bias_train, bias_test,
                                    sd_train, sd_test,
                                    corr_train, corr_test,
                                    sd_delta_train, sd_delta_test,
                                    fom_train, fom_test,
                                    ntrain, ntest]
                c += 1
                ##
                test_start = test_end
                if remaining_obs < test_length:
                    test_end += remaining_obs
                else:
                    test_end += test_length

        if save and not debug:
            name = 'Time_CV'
            wpath = self.path / self.prefix / '_save' / name / mode / \
                str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            if not os.path.exists(wpath):
                os.makedirs(wpath)
            print('[INFO] Saving results.')
            CV.to_pickle(path=wpath / 'Time_CV.pkl')
            print(f'[INFO] Results saved in {wpath}')

    def cvCalib(self, pK, xvars, yvars, pctByCluster, models, labels, thresholdRatio=0.9, random_seed=0, save=True, prefix=None):
        if len(pctByCluster) != len(self.df['Cluster'].unique()):
            raise ValueError(
                f"Length of 'pctByCluster'({len(pctByCluster)}) does not correspond to the number of classes in the Logger class ({len(self.df['Cluster'].unique())}).")
        if len(models) != len(labels):
            raise ValueError(
                f"Length of 'models'({len(models)}) does not correspond to the number of labels ({len(labels)}).")
        if len(yvars) > 1:
            raise ValueError(
                f"Number of target variable 'yvars' ({len(yvars)}) is too large, please select only 1 target variable.")
        # Select index of spikes
        for n, pct in enumerate(pctByCluster):
            if n == 0:
                pKSampleTr = pK[pK['Cluster'] == n + 1].loc[:, ['Start', 'End']].sample(frac=pct,
                                                                                        random_state=random_seed)
                ytemp = pK[pK['Cluster'] == n +
                           1].loc[:, ['Start', 'End']].copy()
                pKSampleTe = ytemp[~ ytemp.index.isin(pKSampleTr.index)]
                #print(n + 1, pKSampleTr.shape)
            else:
                pKSampleTr = pKSampleTr.append(
                    pK[pK['Cluster'] == n + 1].loc[:, ['Start', 'End']].sample(frac=pct, random_state=random_seed))
                ytemp = pK[pK['Cluster'] == n +
                           1].loc[:, ['Start', 'End']].copy()
                pKSampleTe = pKSampleTe.append(
                    ytemp[~ ytemp.index.isin(pKSampleTr.index)])
                #print(n + 1, pKSampleTr.shape, pKSampleTe.shape)
        #print(pKSampleTr.shape, pKSampleTe.shape)

        # Select observations corresponding to the indexes
        for i in range(len(pKSampleTr)):
            if i == 0:
                dfTr = self.df.loc[pKSampleTr.iloc[i, 0]:pKSampleTr.iloc[i, 1], xvars + yvars].copy()
            else:
                dfTr = dfTr.append(
                    self.df.loc[pKSampleTr.iloc[i, 0]:pKSampleTr.iloc[i, 1], xvars + yvars].copy())

        for i in range(len(pKSampleTe)):
            if i == 0:
                dfTe = self.df.loc[pKSampleTe.iloc[i, 0]:pKSampleTe.iloc[i, 1], xvars + yvars].copy()
            else:
                dfTe = dfTe.append(
                    self.df.loc[pKSampleTe.iloc[i, 0]:pKSampleTe.iloc[i, 1], xvars + yvars].copy())
        dfTr.sort_index(inplace=True)
        dfTe.sort_index(inplace=True)
        # Train Model
        X_train = dfTr.loc[:, xvars]
        y_train = pd.DataFrame(dfTr.loc[:, yvars])
        # print(y_train)
        X_test = dfTe.loc[:, xvars]
        y_test = pd.DataFrame(dfTe.loc[:, yvars])

        CV = pd.DataFrame(columns=['MSD_Train', 'MSD_Test',
                                   'RMSE_Train', 'RMSE_Test',
                                   'Bias_Train', 'Bias_Test',
                                   'SD_Train', 'SD_Test',
                                   'CORR_Train', 'CORR_Test',
                                   'SD_delta_Train', 'SD_delta_Test',
                                   'FoM_Train', 'FoM_Test',
                                   'msdTr', 'sbTr', 'sdsdTr', 'lcsTr',
                                   'msdTe', 'sbTe', 'sdsdTe', 'lcsTe'
                                   ])
        cvPTr = pd.DataFrame(
            columns=[i for i in ['Reference'] + labels], index=y_train.index)
        cvPTe = pd.DataFrame(
            columns=[i for i in ['Reference'] + labels], index=y_test.index)
        # print(cvPTr)

        cvPTr['Reference'] = y_train
        cvPTe['Reference'] = y_test
        # print(cvPTr)
        for label, model in zip(labels, models):
            if 'RF-h' in label:
                threshold = dfTr.loc[:, yvars].max()[0] * thresholdRatio

                ixTr1 = dfTr[dfTr.loc[:, yvars] <
                             threshold].loc[:, yvars].dropna().index
                ixTr2 = dfTr[dfTr.loc[:, yvars] >=
                             threshold].loc[:, yvars].dropna().index
                ixTe1 = dfTe[dfTe[yvars] <
                             threshold].loc[:, yvars].dropna().index
                ixTe2 = dfTe[dfTe[yvars] >=
                             threshold].loc[:, yvars].dropna().index
                # print(len(ixTe2))
                X_trainM1 = dfTr.loc[ixTr1, xvars]
                y_trainM1 = dfTr.loc[ixTr1, yvars]
                X_testM1 = dfTe.loc[ixTe1, xvars]
                y_testM1 = dfTe.loc[ixTe1, yvars]

                X_trainM2 = dfTr.loc[ixTr2, xvars]
                y_trainM2 = dfTr.loc[ixTr2, yvars]
                if len(ixTe2) > 0:
                    X_testM2 = dfTe.loc[ixTe2, xvars]
                    y_testM2 = dfTe.loc[ixTe2, yvars]

                model[0].fit(X_trainM1, y_trainM1)
                model[1].fit(X_trainM2, y_trainM2)

                xtr_predM1 = model[0].predict(X_trainM1).flatten()
                xte_predM1 = model[0].predict(X_testM1).flatten()
                xtr_predM2 = model[1].predict(X_trainM2).flatten()
                if len(ixTe2) > 0:
                    xte_predM2 = model[1].predict(X_testM2).flatten()

                ytr1 = y_trainM1.copy()
                ytr1['Model'] = xtr_predM1
                yte1 = y_testM1.copy()
                yte1['Model'] = xte_predM1
                ytr2 = y_trainM2.copy()
                ytr2['Model'] = xtr_predM2
                if len(ixTe2) > 0:
                    yte2 = y_testM2.copy()
                    yte2['Model'] = xte_predM2

                ytr = ytr1.copy()
                ytr = ytr.append(ytr2).sort_index()
                yte = yte1.copy()
                if len(ixTe2) > 0:
                    yte = yte.append(yte2).sort_index()
            else:
                model.fit(X_train, y_train)
                xtrPred = model.predict(X_train).flatten()
                xtePred = model.predict(X_test).flatten()
                ytr = y_train.copy()
                ytr['Model'] = xtrPred
                yte = y_test.copy()
                yte['Model'] = xtePred
            ytr.columns = ['Reference', 'Model']
            yte.columns = ['Reference', 'Model']
            cvPTr.loc[:, label] = ytr['Model'].values
            cvPTe.loc[:, label] = yte['Model'].values
            # Compute stats
            yTrR = ytr['Reference'].values.reshape(-1, 1)
            yTrM = ytr['Model'].values.reshape(-1, 1)
            yTeR = yte['Reference'].values.reshape(-1, 1)
            yTeM = yte['Model'].values.reshape(-1, 1)
            CV.loc[label, :] = computeStats(
                yTrR, yTeR, yTrM, yTeM, kobayashiDec=True)

        if save:
            if prefix is None:
                name = 'CV'
                wpath = self.path / self.prefix / '_save' / name / str(
                    datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            else:
                name = prefix
                wpath = self.path / self.prefix / '_save' / name
            if not os.path.exists(wpath):
                os.makedirs(wpath)
            print('[INFO] Saving results...')
            CV.to_pickle(path=wpath / 'clusterCV.pkl')
            cvPTr.to_pickle(path=wpath / 'clusterCVTrain.pkl')
            cvPTe.to_pickle(path=wpath / 'clusterCVTest.pkl')
            print(f'[INFO] Results saved in {wpath}')
        else:
            return CV, cvPTr, cvPTe

    def cross_valid_time(self, model, xvars, yvars, train_duration, mode, model2=None, thresholdRatio=None, train_start=None, test_duration=None, save=False, prefix=None, debug=False):
        ddf = self.df.loc[:, xvars + yvars]
        ddf.dropna(how='any', inplace=True)
        ix = ddf.index
        if train_start is None:
            train_start = ix[0]

        # mode 1 => Fixed train set | Increasing test set
        # mode 2 => Increasing train set | Fixed test set (at the end?)
        # mode 3 => train set (Fixed) | Shifted test set
        # mode 4 => Train set defined by user | Shifted test set
        # mode 5 => Train set is defined at the lengths defined by train durarion

        if mode == '1' or mode not in ['2', '3', '4', '5', '6']:
            train_start = train_start
            train_end = train_start + pd.Timedelta(days=train_duration)
            test_start = train_end + pd.Timedelta(hours=1)
            test_end = test_start + pd.Timedelta(days=test_duration)
            ixtr = [[train_start, train_end]]
            ixte = [[test_start, test_end]]
            while test_end < ix[-1]:
                if debug:
                    print(
                        f'[DEBUG] MODE {mode} S: {test_start}, E: {test_end}')
                test_end = test_end + pd.Timedelta(days=test_duration)
                # ixtr.append([train_start, train_end])
                ixte.append([test_start,  test_end])
        elif mode == '2':
            train_start = train_start
            train_end = train_start + pd.Timedelta(days=train_duration)
            test_start = ix[-1] - pd.Timedelta(days=test_duration)
            test_end = ix[-1]
            ixtr = [[train_start, train_end]]
            ixte = [[test_start, test_end]]
            while train_end < test_start:
                if debug:
                    print(
                        f'[DEBUG] MODE {mode} S: {train_start}, E: {train_end}')
                train_end = train_end + pd.Timedelta(days=train_duration)
                ixtr.append([train_start, train_end])
                # ixte.append([test_start, test_end])
        elif mode == '3':
            train_start = train_start
            train_end = train_start + pd.Timedelta(days=train_duration)
            test_start = train_end + pd.Timedelta(hours=1)
            test_end = test_start + pd.Timedelta(days=test_duration)
            ixtr = [[train_start, train_end]]
            ixte = [[test_start, test_end]]
            while test_end < ix[-1]:
                if debug:
                    print(
                        f'[DEBUG] MODE {mode} S: {test_start}, E: {test_end}')
                test_start = test_start + pd.Timedelta(days=1)
                test_end = test_start + pd.Timedelta(days=test_duration)
                # ixtr.append([train_start, train_end])
                ixte.append([test_start, test_end])
        elif mode == '4':
            ls = []
            ixtr = []
            for n, index in enumerate(train_duration):
                ls = ls + list(pd.date_range(start=pd.to_datetime(index[0]), end=pd.to_datetime(
                    index[0]) + pd.Timedelta(days=index[1]), freq='5s'))
                ixtr.append([pd.to_datetime(index[0]), pd.to_datetime(
                    index[0]) + pd.Timedelta(days=index[1])])
            ddf_tr = ddf[ddf.index.isin(ls)]
            ddf_te = ddf[~ddf.index.isin(ls)]

            test_start = ddf_te.index[0]
            test_end = ddf_te.index[0] + pd.Timedelta(days=test_duration)
            ixte = [[test_start, test_end]]
            while test_end <= ddf_te.index[-1]:
                if debug:
                    print(
                        f'[DEBUG] MODE {mode} S: {test_start}, E: {test_end}')
                test_start = test_start + pd.Timedelta(days=1)
                test_end = test_start + pd.Timedelta(days=test_duration)
                ixte.append([test_start, test_end])
        elif mode == '5':
            ixtr = []
            for train in train_duration:
                train_end = train_start + pd.Timedelta(days=train)
                if debug:
                    print(
                        f'[DEBUG] MODE {mode} S: {train_start}, E: {train_end}')
                ixtr.append([train_start, train_end])
        elif mode == '6':
            ixtr = []
            ixte = []
            for train in train_duration:
                train_end = train_start + pd.Timedelta(days=train)
                ixtr.append([train_start, train_end])
                trainIxLst = list(pd.date_range(
                    start=train_start, end=train_end, freq='5s'))
                te = []
                ddfTeix = ddf[~ddf.index.isin(
                    ddf.loc[train_start:train_end, :].index)].index
                testStart = ddfTeix[0]
                testEnd = testStart + pd.Timedelta(days=test_duration)
                while testEnd in trainIxLst:
                    testEnd += pd.Timedelta(days=1)
                te.append([testStart, testEnd])

                while testEnd <= ddf.index[-1]:
                    testStart = testEnd
                    testEnd = testStart + pd.Timedelta(days=test_duration)
                    while testEnd in trainIxLst:
                        testEnd += pd.Timedelta(days=1)
                    te.append([testStart, testEnd])
                ixte.append(te)

        if mode == '5':
            CV = pd.DataFrame(columns=['RMSE_Train', 'RMSE_Test',
                                       'Bias_Train', 'Bias_Test',
                                       'SD_Train', 'SD_Test',
                                       'CORR_Train', 'CORR_Test',
                                       'SD_delta_Train', 'SD_delta_Test',
                                       'fom_Train', 'fom_Test', 'N_train', 'N_Test',
                                       'CvMAE_Train', 'CvMAE_test'])
        elif mode == '6':
            CV = pd.DataFrame(columns=['TrainDuration', 'TestWindow',
                                       'RMSE_Train', 'RMSE_Test',
                                       'Bias_Train', 'Bias_Test',
                                       'SD_Train', 'SD_Test',
                                       'CORR_Train', 'CORR_Test',
                                       'SD_delta_Train', 'SD_delta_Test',
                                       'fom_Train', 'fom_Test', 'N_train', 'N_Test',
                                       'CvMAE_Train', 'CvMAE_test'])
        else:
            CV = pd.DataFrame(columns=['RMSE_Train', 'RMSE_Test',
                                       'Bias_Train', 'Bias_Test',
                                       'SD_Train', 'SD_Test',
                                       'CORR_Train', 'CORR_Test',
                                       'SD_delta_Train', 'SD_delta_Test',
                                       'fom_Train', 'fom_Test', 'N_train', 'N_Test',
                                       ])
        # n = max(len(ixtr), len(ixte))
        # CV_P_tr = pd.DataFrame(columns=[str(i) for i in range(1, n+1)])
        # CV_P_te = pd.DataFrame(columns=[str(i) for i in range(1, n+1)])
        # print(n, len(ixtr), len(ixte))

        if not debug:
            print('[INFO] Start time CV.')
            # CASE 1: Update train and test set
            # CASE 2: Fixed train and update test set

            if mode in ['1', '3', '4']:
                if mode == '4':
                    X_train = ddf_tr.loc[:, xvars]
                    y_train = ddf_tr.loc[:, yvars]
                else:
                    X_train = ddf.loc[ixtr[0][0]:ixtr[0][1], xvars]
                    y_train = ddf.loc[ixtr[0][0]:ixtr[0][1], yvars]
                # Train model
                model.fit(X_train, y_train)
                # Train prediction
                xtr_pred = model.predict(X_train).flatten()
                # Train metrics
                msd_train = mse(y_train.values, xtr_pred)
                rmse_train = np.sqrt(msd_train)
                bias_train = np.mean(xtr_pred.reshape(
                    len(xtr_pred), 1) - y_train.values)
                sd_train = 100 * (np.std(xtr_pred) / np.std(y_train.values))
                corr_train = np.corrcoef(
                    xtr_pred, y_train.values.reshape(len(y_train.values), ))[0, 1]
                sd_delta_train = np.std(xtr_pred.reshape(
                    len(xtr_pred), 1) - y_train.values)
                a = np.trapz(xtr_pred)
                b = np.trapz(y_train.values.reshape(1, len(y_train))[0])
                # c = np.abs(a - b)
                fom_train = a / b  # a / (a + c)
                # fom_train = np.mean(y_train.values/xtr_pred.reshape(len(xtr_pred), 1))
                ntrain = len(y_train)

                # Test prediction
                for c, i in tqdm(enumerate(ixte)):
                    # print(f'[INFO] Test set {c+1}/{len(ixte)}')
                    if mode == '4':
                        X_test = ddf_te.loc[i[0]:i[1], xvars]
                        y_test = ddf_te.loc[i[0]:i[1], yvars]
                    else:
                        X_test = ddf.loc[i[0]:i[1], xvars]
                        y_test = ddf.loc[i[0]:i[1], yvars]
                    xte_pred = model.predict(X_test).flatten()
                    msd_test = mse(y_test.values, xte_pred)
                    rmse_test = np.sqrt(msd_test)
                    bias_test = np.mean(xte_pred.reshape(
                        len(xte_pred), 1) - y_test.values)
                    sd_test = 100 * (np.std(xte_pred) / np.std(y_test.values))
                    corr_test = np.corrcoef(
                        xte_pred, y_test.values.reshape(len(y_test.values), ))[0, 1]
                    sd_delta_test = np.std(xte_pred.reshape(
                        len(xte_pred), 1) - y_test.values)
                    a = np.trapz(xte_pred)
                    b = np.trapz(y_test.values.reshape(1, len(y_test))[0])
                    # c = np.abs(a - b)
                    fom_test = a / b  # a / (a + c)
                    # fom_test = np.mean(y_test.values/xte_pred.reshape(len(xte_pred), 1))
                    ntest = len(y_test)
                    CV.loc[c + 1, :] = [rmse_train, rmse_test,
                                        bias_train, bias_test,
                                        sd_train, sd_test,
                                        corr_train, corr_test,
                                        sd_delta_train, sd_delta_test,
                                        fom_train, fom_test,
                                        ntrain, ntest]
                print(f'[INFO] End test')
            elif mode == '2':
                for c, i in tqdm(enumerate(ixtr)):
                    # print(f'[INFO] Train set {c + 1}/{len(ixte)}')
                    X_train = ddf.loc[i[0]:i[1], xvars]
                    y_train = ddf.loc[i[0]:i[1], yvars]
                    # Train model
                    model.fit(X_train, y_train)
                    # Train prediction
                    xtr_pred = model.predict(X_train).flatten()
                    # Train metrics
                    msd_train = mse(y_train.values, xtr_pred)
                    rmse_train = np.sqrt(msd_train)
                    bias_train = np.mean(xtr_pred.reshape(
                        len(xtr_pred), 1) - y_train.values)
                    sd_train = 100 * (np.std(xtr_pred) /
                                      np.std(y_train.values))
                    corr_train = np.corrcoef(
                        xtr_pred, y_train.values.reshape(len(y_train.values), ))[0, 1]
                    sd_delta_train = np.std(xtr_pred.reshape(
                        len(xtr_pred), 1) - y_train.values)
                    a = np.trapz(xtr_pred)
                    b = np.trapz(y_train.values.reshape(1, len(y_train))[0])
                    # c = np.abs(a - b)
                    fom_train = a / b  # a / (a + c)
                    # fom_train = np.mean(y_train.values / xtr_pred.reshape(len(xtr_pred), 1))
                    ntrain = len(y_train)

                    # Test prediction
                    X_test = ddf.loc[ixte[0][0]:ixte[0][1], xvars]
                    y_test = ddf.loc[ixte[0][0]:ixte[0][1], yvars]
                    xte_pred = model.predict(X_test).flatten()
                    msd_test = mse(y_test.values, xte_pred)
                    rmse_test = np.sqrt(msd_test)
                    bias_test = np.mean(xte_pred.reshape(
                        len(xte_pred), 1) - y_test.values)
                    sd_test = 100 * (np.std(xte_pred) / np.std(y_test.values))
                    corr_test = np.corrcoef(
                        xte_pred, y_test.values.reshape(len(y_test.values), ))[0, 1]
                    sd_delta_test = np.std(xte_pred.reshape(
                        len(xte_pred), 1) - y_test.values)
                    a = np.trapz(xte_pred)
                    b = np.trapz(y_test.values.reshape(1, len(y_test))[0])
                    # c = np.abs(a - b)
                    fom_test = a / b  # a / (a + c)
                    # fom_test = np.mean(y_test.values / xte_pred.reshape(len(xte_pred), 1))
                    ntest = len(y_test)
                    CV.loc[c + 1, :] = [rmse_train, rmse_test,
                                        bias_train, bias_test,
                                        sd_train, sd_test,
                                        corr_train, corr_test,
                                        sd_delta_train, sd_delta_test,
                                        fom_train, fom_test,
                                        ntrain, ntest]
                print(f'[INFO] End test')
            elif mode == '5':
                for c, i in enumerate(ixtr):
                    # print(f'[INFO] Train set {c + 1}/{len(ixte)}')
                    if model2 is not None:
                        ddfTr = ddf.loc[i[0]:i[1], xvars+yvars]
                        ddfTe = ddf[~ddf.index.isin(ddfTr.index)]
                        threshold = ddf.loc[i[0]:i[1], yvars].max()[
                            0] * thresholdRatio

                        ixTr1 = ddfTr[ddfTr.loc[:, yvars] <
                                      threshold].loc[:, yvars].dropna().index
                        ixTr2 = ddfTr[ddfTr.loc[:, yvars] >=
                                      threshold].loc[:, yvars].dropna().index
                        ixTe1 = ddfTe[ddfTe[yvars] <
                                      threshold].loc[:, yvars].dropna().index
                        ixTe2 = ddfTe[ddfTe[yvars] >=
                                      threshold].loc[:, yvars].dropna().index

                        X_trainM1 = ddfTr.loc[ixTr1, xvars]
                        y_trainM1 = ddfTr.loc[ixTr1, yvars]
                        X_testM1 = ddfTe.loc[ixTe1, xvars]
                        y_testM1 = ddfTe.loc[ixTe1, yvars]

                        X_trainM2 = ddfTr.loc[ixTr2, xvars]
                        y_trainM2 = ddfTr.loc[ixTr2, yvars]
                        if len(ixTe2) > 0:
                            X_testM2 = ddfTe.loc[ixTe2, xvars]
                            y_testM2 = ddfTe.loc[ixTe2, yvars]

                        model.fit(X_trainM1, y_trainM1)
                        model2.fit(X_trainM2, y_trainM2)

                        xtr_predM1 = model.predict(X_trainM1).flatten()
                        xte_predM1 = model.predict(X_testM1).flatten()
                        xtr_predM2 = model2.predict(X_trainM2).flatten()
                        if len(ixTe2) > 0:
                            xte_predM2 = model2.predict(X_testM2).flatten()

                        ytr1 = y_trainM1.copy()
                        ytr1['Model'] = xtr_predM1
                        yte1 = y_testM1.copy()
                        yte1['Model'] = xte_predM1
                        ytr2 = y_trainM2.copy()
                        ytr2['Model'] = xtr_predM2
                        if len(ixTe2) > 0:
                            yte2 = y_testM2.copy()
                            yte2['Model'] = xte_predM2

                        ytr = ytr1.copy()
                        ytr = ytr.append(ytr2).sort_index()
                        yte = yte1.copy()
                        if len(ixTe2) > 0:
                            yte = yte.append(yte2).sort_index()
                        ytr.columns = ['Reference', 'Model']
                        yte.columns = ['Reference', 'Model']
                        # print(ytr, yte)

                        yTrR = ytr['Reference'].values.reshape(-1, 1)
                        yTrM = ytr['Model'].values.reshape(-1, 1)
                        yTeR = yte['Reference'].values.reshape(-1, 1)
                        yTeM = yte['Model'].values.reshape(-1, 1)
                        msdTrain, msdTest, rmseTrain, rmseTrain, biasTrain, biasTest, sdTrain, sdTest, corrTrain, corrTest,\
                            sdDeltaTrain, sdDeltaTest, fomTrain, fomTest = computeStats(
                                yTrR, yTeR, yTrM, yTeM)
                        ntrain = len(yTrR)
                        ntest = len(yTeR)
                        CvMAE_train = np.sum(
                            np.abs(yTrR - yTrM - biasTrain)) / np.sum(yTrR)
                        CvMAE_test = np.sum(
                            np.abs(yTeR - yTeM - biasTest)) / np.sum(yTeR)

                        CV.loc[c + 1, :] = [rmseTrain, rmseTrain,
                                            biasTrain, biasTest,
                                            sdTrain, sdTest,
                                            corrTrain, corrTest,
                                            sdDeltaTrain, sdDeltaTest,
                                            fomTrain, fomTest,
                                            ntrain, ntest,
                                            CvMAE_train, CvMAE_test]
                    else:
                        X_train = ddf.loc[i[0]:i[1], xvars]
                        y_train = ddf.loc[i[0]:i[1], yvars]
                        # Train model
                        model.fit(X_train, y_train)
                        # Train prediction
                        xtr_pred = model.predict(X_train).flatten()
                        # Train metrics
                        msd_train = mse(y_train.values, xtr_pred)
                        rmse_train = np.sqrt(msd_train)
                        bias_train = np.mean(xtr_pred.reshape(
                            len(xtr_pred), 1) - y_train.values)
                        sd_train = 100 * (np.std(xtr_pred) /
                                          np.std(y_train.values))
                        corr_train = np.corrcoef(
                            xtr_pred, y_train.values.reshape(len(y_train.values), ))[0, 1]
                        sd_delta_train = np.std(xtr_pred.reshape(
                            len(xtr_pred), 1) - y_train.values)
                        a = np.trapz(xtr_pred)
                        b = np.trapz(y_train.values.reshape(
                            1, len(y_train))[0])
                        # c = np.abs(a - b)
                        fom_train = a / b  # a / (a + c)
                        # fom_train = np.mean(y_train.values / xtr_pred.reshape(len(xtr_pred), 1))
                        ntrain = len(y_train)
                        CvMAE_train = np.sum(
                            np.abs(y_train.values - xtr_pred.reshape(len(xtr_pred), 1) - bias_train)) / np.sum(
                            y_train.values)

                        # Test prediction
                        ddf_te = ddf[~ddf.index.isin(X_train.index)]
                        X_test = ddf_te.loc[:, xvars]
                        y_test = ddf_te.loc[:, yvars]
                        xte_pred = model.predict(X_test).flatten()
                        msd_test = mse(y_test.values, xte_pred)
                        rmse_test = np.sqrt(msd_test)
                        bias_test = np.mean(xte_pred.reshape(
                            len(xte_pred), 1) - y_test.values)
                        sd_test = 100 * (np.std(xte_pred) /
                                         np.std(y_test.values))
                        corr_test = np.corrcoef(
                            xte_pred, y_test.values.reshape(len(y_test.values), ))[0, 1]
                        sd_delta_test = np.std(xte_pred.reshape(
                            len(xte_pred), 1) - y_test.values)
                        a = np.trapz(xte_pred)
                        b = np.trapz(y_test.values.reshape(1, len(y_test))[0])
                        fom_test = a / b  # a / (a + c)
                        CvMAE_test = np.sum(
                            np.abs(y_test.values - xte_pred.reshape(len(xte_pred), 1) - bias_test)) / np.sum(
                            y_test.values)

                        ntest = len(y_test)
                        CV.loc[c + 1, :] = [rmse_train, rmse_test,
                                            bias_train, bias_test,
                                            sd_train, sd_test,
                                            corr_train, corr_test,
                                            sd_delta_train, sd_delta_test,
                                            fom_train, fom_test,
                                            ntrain, ntest,
                                            CvMAE_train, CvMAE_test]
            elif mode == '6':
                count = 0
                for c, ix in enumerate(zip(ixtr, ixte)):
                    if model2 is not None:
                        ddfTr = ddf.loc[ix[0][0]:ix[0][1], xvars + yvars]
                        threshold = ddf.loc[ix[0][0]:ix[0][1], yvars].max()[
                            0] * thresholdRatio
                        ixTr1 = ddfTr[ddfTr.loc[:, yvars] <
                                      threshold].loc[:, yvars].dropna().index
                        ixTr2 = ddfTr[ddfTr.loc[:, yvars] >=
                                      threshold].loc[:, yvars].dropna().index
                        X_trainM1 = ddfTr.loc[ixTr1, xvars]
                        y_trainM1 = ddfTr.loc[ixTr1, yvars]
                        X_trainM2 = ddfTr.loc[ixTr2, xvars]
                        y_trainM2 = ddfTr.loc[ixTr2, yvars]
                        model.fit(X_trainM1, y_trainM1)
                        model2.fit(X_trainM2, y_trainM2)
                        xtr_predM1 = model.predict(X_trainM1).flatten()
                        xtr_predM2 = model2.predict(X_trainM2).flatten()
                        ytr1 = y_trainM1.copy()
                        ytr1['Model'] = xtr_predM1
                        ytr2 = y_trainM2.copy()
                        ytr2['Model'] = xtr_predM2
                        ytr = ytr1.copy()
                        ytr = ytr.append(ytr2).sort_index()
                        ytr.columns = ['Reference', 'Model']
                        yTrR = ytr['Reference'].values.reshape(-1, 1)
                        yTrM = ytr['Model'].values.reshape(-1, 1)
                        for cc, ii in enumerate(ix[1]):
                            ddfTest = ddf[~ddf.index.isin(
                                ddf.loc[ix[0][0]:ix[0][1], :].index)]
                            ddfTe = ddfTest.loc[ii[0]:ii[1], xvars+yvars]
                            ixTe1 = ddfTe[ddfTe[yvars] <
                                          threshold].loc[:, yvars].dropna().index
                            ixTe2 = ddfTe[ddfTe[yvars] >=
                                          threshold].loc[:, yvars].dropna().index
                            X_testM1 = ddfTe.loc[ixTe1, xvars]
                            y_testM1 = ddfTe.loc[ixTe1, yvars]
                            if len(ixTe2) > 0:
                                X_testM2 = ddfTe.loc[ixTe2, xvars]
                                y_testM2 = ddfTe.loc[ixTe2, yvars]
                            if len(X_testM1) > 0:
                                xte_predM1 = model.predict(X_testM1).flatten()
                                if len(ixTe2) > 0:
                                    xte_predM2 = model2.predict(
                                        X_testM2).flatten()

                                yte1 = y_testM1.copy()
                                yte1['Model'] = xte_predM1
                                if len(ixTe2) > 0:
                                    yte2 = y_testM2.copy()
                                    yte2['Model'] = xte_predM2
                                yte = yte1.copy()
                                if len(ixTe2) > 0:
                                    yte = yte.append(yte2).sort_index()
                                yte.columns = ['Reference', 'Model']
                                yTeR = yte['Reference'].values.reshape(-1, 1)
                                yTeM = yte['Model'].values.reshape(-1, 1)
                                msdTrain, msdTest, rmseTrain, rmseTrain, biasTrain, biasTest, sdTrain, sdTest, corrTrain, corrTest,\
                                    sdDeltaTrain, sdDeltaTest, fomTrain, fomTest = computeStats(
                                        yTrR, yTeR, yTrM, yTeM)
                                ntrain = len(yTrR)
                                ntest = len(yTeR)
                                CvMAE_train = np.sum(
                                    np.abs(yTrR - yTrM - biasTrain)) / np.sum(yTrR)
                                CvMAE_test = np.sum(
                                    np.abs(yTeR - yTeM - biasTest)) / np.sum(yTeR)
                                CV.loc[count, :] = [train_duration[c], cc,
                                                    rmseTrain, rmseTrain,
                                                    biasTrain, biasTest,
                                                    sdTrain, sdTest,
                                                    corrTrain, corrTest,
                                                    sdDeltaTrain, sdDeltaTest,
                                                    fomTrain, fomTest,
                                                    ntrain, ntest,
                                                    CvMAE_train, CvMAE_test]
                                count += 1
                    else:
                        X_train = ddf.loc[ix[0][0]:ix[0][1], xvars]
                        y_train = ddf.loc[ix[0][0]:ix[0][1], yvars]
                        # Train model
                        model.fit(X_train, y_train)
                        # Train prediction
                        xtr_pred = model.predict(X_train).flatten()
                        ytr1 = y_train.copy()
                        ytr1['Model'] = xtr_pred
                        ytr = ytr1.copy()
                        ytr.columns = ['Reference', 'Model']
                        yTrR = ytr['Reference'].values.reshape(-1, 1)
                        yTrM = ytr['Model'].values.reshape(-1, 1)

                        for cc, ii in enumerate(ix[1]):
                            # Test prediction
                            ddfTe = ddf[~ddf.index.isin(
                                ddf.loc[ix[0][0]:ix[0][1], :].index)]
                            X_test = ddfTe.loc[ii[0]:ii[1], xvars]
                            y_test = ddfTe.loc[ii[0]:ii[1], yvars]
                            if len(X_test) > 0:
                                xte_pred = model.predict(X_test).flatten()
                                yte = y_test.copy()
                                yte['Model'] = xte_pred
                                yte.columns = ['Reference', 'Model']
                                yTeR = yte['Reference'].values.reshape(-1, 1)
                                yTeM = yte['Model'].values.reshape(-1, 1)
                                msdTrain, msdTest, rmseTrain, rmseTrain, biasTrain, biasTest, sdTrain, sdTest, corrTrain, corrTest, \
                                    sdDeltaTrain, sdDeltaTest, fomTrain, fomTest = computeStats(
                                        yTrR, yTeR, yTrM, yTeM)
                                ntrain = len(yTrR)
                                ntest = len(yTeR)
                                CvMAE_train = np.sum(
                                    np.abs(yTrR - yTrM - biasTrain)) / np.sum(yTrR)
                                CvMAE_test = np.sum(
                                    np.abs(yTeR - yTeM - biasTest)) / np.sum(yTeR)

                                CV.loc[count, :] = [train_duration[c], cc,
                                                    rmseTrain, rmseTrain,
                                                    biasTrain, biasTest,
                                                    sdTrain, sdTest,
                                                    corrTrain, corrTest,
                                                    sdDeltaTrain, sdDeltaTest,
                                                    fomTrain, fomTest,
                                                    ntrain, ntest,
                                                    CvMAE_train, CvMAE_test]
                                count += 1
        if save and not debug:
            name = 'Time_CV'
            if prefix is None:
                wpath = self.path / self.prefix / '_save' / name / mode / \
                    str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            else:
                wpath = self.path / self.prefix / '_save' / name / mode / prefix
            if not os.path.exists(wpath):
                os.makedirs(wpath)
            print('[INFO] Saving results.')
            CV.to_pickle(path=wpath / 'Time_CV.pkl')
            if mode == '5':
                tix_train = pd.DataFrame(ixtr)
                tix_train.to_csv(path_or_buf=wpath / 'ixTrain.csv')
            else:
                tix_train = pd.DataFrame(ixtr)
                tix_test = pd.DataFrame(ixte)
                tix_test.to_csv(path_or_buf=wpath / 'ixTest.csv')
            print(f'[INFO] Results saved in {wpath}')

    def gen_legend_labs(self):
        ins_names = []
        lab_names = []
        for k in self.variables_to_plot:
            ins_names.append(self.variables[k][0])
            lab_names.append(self.variables[k][1])
        return ins_names, lab_names

    def figure_path(self):
        self.count_figs += 1
        name = self.prefix + '_' + str(self.count_figs) + self.sufix
        return self.path / name

    def plot_vars(self, variables=None, start=None, end=None, style='.', ylims=None, vline=None, bars=None,
                  break_axes=None, qq=(0, 1), marker_size=4,
                  fontsize=16, figsize=(20, 10), date_format='A', save=False):
        if variables is None:
            self.variables_to_plot = list(self.variables.keys())
        else:
            self.variables_to_plot = variables
        ins_names, lab_names = self.gen_legend_labs()
        if bars is None:
            plot_comp_all_vars(self.df,
                               self.variables_to_plot,
                               ylabs=lab_names,
                               legend_labs=ins_names,
                               figsize=figsize,
                               qq=qq,
                               date_format=date_format,
                               marker_size=marker_size,
                               start=start,
                               end=end,
                               style=style,
                               fontsize=fontsize,
                               ylims=ylims,
                               vline=vline,
                               bars=bars,
                               break_axes=break_axes,
                               file_name=self.figure_path() if save else None
                               )
        else:
            # TODO: Correct the plot to show the binary variable (parameters problem?)
            ins_names_s = []
            for i in ins_names:
                ins_names_s.append([i[0], 'Flagged Spikes'])

            print(variables, lab_names, ins_names_s)
            plot_comp_all_vars(self.df,
                               variables,
                               ylabs=lab_names,
                               legend_labs=ins_names_s,
                               figsize=figsize,
                               qq=[0, 1],
                               date_format=date_format,
                               marker_size=4,
                               start=start,
                               end=end,
                               style=style,
                               vline=vline,
                               bars=bars,
                               file_name=self.figure_path() if save else None
                               )

    @staticmethod
    def format_axs(ax_f, ylabs, xticks, reverse, fontsize, locator=(3, 1), scatter=False):
        if scatter:
            ax_f.legend(markerscale=3, prop={
                        'size': fontsize}, loc='best', frameon=True, fancybox=True)
            ax_f.set_xlabel(ylabs[0][0], fontdict={'size': fontsize})
            ax_f.set_ylabel(ylabs[1][0], fontdict={'size': fontsize})
            ax_f.yaxis.set_tick_params(labelsize=fontsize)
            ax_f.xaxis.set_tick_params(labelsize=fontsize)
            ax_f.yaxis.set_major_locator(plt.AutoLocator())
            ax_f.xaxis.set_major_locator(plt.AutoLocator())
            ax_f.yaxis.set_minor_locator(plt.AutoLocator())
            ax_f.xaxis.set_minor_locator(plt.AutoLocator())
            ax_f.spines['left'].set_linewidth(2)
            ax_f.spines['left'].set_color('gray')
            ax_f.spines['bottom'].set_linewidth(2)
            ax_f.spines['bottom'].set_color('gray')
            ax_f.spines['right'].set_linewidth(0.5)
            ax_f.spines['right'].set_color('gray')
            ax_f.spines['top'].set_linewidth(0.5)
            ax_f.spines['top'].set_color('gray')
        else:
            ax_f.set_xlabel('')
            ax_f.set_ylabel(ylabs, fontdict={'size': fontsize})
            if locator is None:
                ax_f.yaxis.set_major_locator(plt.AutoLocator())
                ax_f.yaxis.set_minor_locator(plt.AutoLocator())
            else:
                ax_f.yaxis.set_major_locator(plt.MultipleLocator(locator[0]))
                ax_f.yaxis.set_minor_locator(plt.MultipleLocator(locator[1]))
            ax_f.yaxis.set_tick_params(labelsize=fontsize)
            ax_f.xaxis.set_tick_params(labelsize=fontsize)
            locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            formatter.formats = ['%y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f']
            formatter.zero_formats = [''] + formatter.formats[:-1]
            formatter.zero_formats[2] = '%d-%b'
            formatter.offset_formats = [
                '', '', '', '%d %b %Y', '%d %b %Y', '%d %b %Y %H:%M']

            ax_f.xaxis.set_major_locator(locator)
            ax_f.xaxis.set_major_formatter(formatter)
            ax_f.xaxis.set_minor_locator(mdates.DayLocator())
            ax_f.tick_params(which='minor', length=4, color='k')
            ax_f.spines['left'].set_linewidth(2)
            ax_f.spines['left'].set_color('gray')
            ax_f.spines['bottom'].set_linewidth(2)
            ax_f.spines['bottom'].set_color('gray')
            ax_f.spines['right'].set_linewidth(0.5)
            ax_f.spines['right'].set_color('gray')
            ax_f.spines['top'].set_linewidth(0.5)
            ax_f.spines['top'].set_color('gray')

        if reverse:
            ax_f.invert_yaxis()

        if not xticks:
            ax_f.set_xticklabels('')
        return ax_f

    @staticmethod
    def format_ax0(ax, time=False, date_format='1W'):
        if time:
            if len(date_format) > 1:
                date_interval = int(date_format[0])
                date_type = date_format[1]
            else:
                date_type = date_format
                date_interval = 1

            if date_type is 'W':
                locator = mdates.WeekdayLocator(
                    byweekday=0, interval=date_interval)
                minlocator = mdates.DayLocator()
            elif date_type is 'D':
                locator = mdates.DayLocator(interval=date_interval)
                minlocator = mdates.HourLocator()
            else:
                locator = mdates.AutoDateLocator(minticks=7, maxticks=10)
                minlocator = mdates.AutoDateLocator(minticks=5, maxticks=10)
            formatter = mdates.ConciseDateFormatter(locator)
            formatter.formats = ['%y', '%b', '%d-%b',
                                 '%H:%M:%S', '%H:%M:%S', '%S.%f']
            formatter.zero_formats = [
                '', '%y', '%d-%b', '%d-%b', '%H:%M:%S', '%H:%M:%S']
            formatter.offset_formats = [
                '', '', '', '%b %Y', '%d %b %Y', '%d %b %Y']
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_locator(minlocator)
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(5), )
            ax.ticklabel_format(axis='x', style='sci',
                                scilimits=(-3, 3), useMathText=True)

        ax.xaxis.set_tick_params(rotation=0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5), )
        ax.ticklabel_format(axis='y', style='sci',
                            scilimits=(-3, 3), useMathText=True)
        ax.tick_params(which='minor', length=4, color='k')
        ax.tick_params(which='major', length=8, color='k', pad=10)

        ax.spines['left'].set_linewidth(2)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color('gray')
        return ax

    def make_reg(self, ds, variables, deg):
        ds = self.df if ds is None else ds
        dd = ds.loc[:, variables]
        dd.dropna(inplace=True)
        if deg > 1:
            model = make_pipeline(RobustScaler(quantile_range=(1.0, 99.0)),
                                  PolynomialFeatures(deg),
                                  linear_model.LinearRegression()
                                  )
        else:
            model = make_pipeline(RobustScaler(quantile_range=(1.0, 99.0)),
                                  linear_model.LinearRegression()
                                  )
        model.fit(dd.loc[:, variables[0]].values.reshape(
            len(dd), 1), dd.loc[:, variables[1]].values.reshape(len(dd), 1))
        dd['y_pred'] = model.predict(
            dd.loc[:, variables[0]].values.reshape(len(dd), 1))
        m_slope = model.named_steps['linearregression'].coef_[0]
        r_2_score = r2(dd.loc[:, variables[1]].values.reshape(
            len(dd), 1), dd.loc[:, 'y_pred'].values.reshape(len(dd), 1))
        rmse = np.sqrt(mse(dd.loc[:, variables[1]].values.reshape(
            len(dd), 1), dd.loc[:, 'y_pred'].values.reshape(len(dd), 1)))
        num_obs = len(dd)
        return dd, m_slope, r_2_score, rmse, num_obs, model

    def plot_corr(self, df, variables, figsize=(15, 10), marker_size=5, degree=1, eq=True, latex=False, save=False, ylabels=None):
        df = self.df if df is None else df
        # 1) part
        ylabel_vars = []
        legend_vars = []
        if ylabels is not None:
            ylabel_vars, legend_vars, axis_labs, axis_legs = ylabels
        else:
            for i in variables:
                ylabel_vars.append(self.variables[i][1])
                legend_vars.append(self.variables[i][0])
        # 2) part
        axis_vars = [i for i in combinations(variables, 2)]
        axis_labs = []
        axis_legs = []
        if ylabels is None:
            for i, j in axis_vars:
                axis_labs.append((self.variables[i][0], self.variables[j][0]))
                axis_legs.append((self.variables[i][1], self.variables[j][1]))

        n = len(axis_vars)
        x = int(np.ceil(np.sqrt(n)))
        y = int(np.ceil(n / x))
        # print(n,x,y)
        with plt.style.context('seaborn-whitegrid'):
            sns.set(font_scale=1)
            sns.set_style("ticks")

            fig, ax = plt.subplots(
                nrows=y + 2, ncols=x, sharey=False, figsize=figsize, squeeze=False)
            gs0 = ax[0, 0].get_gridspec()
            gs1 = ax[1, 0].get_gridspec()
            for a in ax[0, :]:
                a.remove()
            for a in ax[1, :]:
                a.remove()
            ax0 = fig.add_subplot(gs0[0, :])
            ax1 = fig.add_subplot(gs1[1, :])

            ax0 = df.loc[:, variables[0]].plot(
                ax=ax0, style='.', grid=True, rot=0, ms=marker_size, x_compat=True)  # x_compat=True
            ax0.lines[0].set_color('r')
            ax0.legend(legend_vars[0], markerscale=3, prop={
                       'size': self.fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))
            ax0 = self.format_axs(
                ax0, ylabs=ylabel_vars[0], xticks=False, reverse=False, fontsize=self.fontsize, locator=None)

            ax1 = df.loc[:, variables[1:]].plot(
                ax=ax1, style='.', grid=True, rot=0, ms=marker_size, x_compat=True)
            cmp_colors = plt.cm.get_cmap('Set2')(
                np.linspace(0, 1, len(variables)-1))
            for k, color in enumerate(cmp_colors):
                ax1.lines[k].set_color(color)
            ax1.legend(legend_vars[1], markerscale=3, prop={
                       'size': self.fontsize}, loc='center left', bbox_to_anchor=(1, 0.5))
            ax1 = self.format_axs(
                ax1, ylabs=ylabel_vars[1], xticks=True, reverse=False, fontsize=self.fontsize, locator=None)

            for count, ax_ix in enumerate(list(product(range(2, y+2), range(0, x)))):
                if count < n:
                    dd_r, m, r_2, n_r, * \
                        _ = self.make_reg(df, axis_vars[count], degree)
                    ax[ax_ix[0], ax_ix[1]] = dd_r.plot(ax=ax[ax_ix[0], ax_ix[1]], grid=True, style='.', ms=marker_size,
                                                       x=axis_vars[count][0], y=axis_vars[count][1], x_compat=True)  # x_compat=True
                    ax[ax_ix[0], ax_ix[1]] = dd_r.plot(ax=ax[ax_ix[0], ax_ix[1]], grid=True, style='.',
                                                       ms=marker_size, x=axis_vars[count][0], y='y_pred', x_compat=True)  # x_compat=True
                    ax[ax_ix[0], ax_ix[1]].lines[0].set_color('b')
                    ax[ax_ix[0], ax_ix[1]].lines[1].set_color('r')
                    ax[ax_ix[0], ax_ix[1]].lines[0].set_label('')
                    c = len(m) - 1
                    mm = ''
                    while c > 0:
                        if c == 1:
                            mm += str(round(m[c], 4)) + ' $x$ +'
                        else:
                            mm += str(round(m[c], 4)) + f' $x^{c}$ +'
                        c -= 1
                    mm += str(m[0])
                    if eq:
                        ax[ax_ix[0], ax_ix[1]].lines[1].set_label(
                            f'{mm}\n'+'$\mathrm{R^{2}}$: '+f'{r_2}\n# obs: {round(n_r)}')
                    else:
                        ax[ax_ix[0], ax_ix[1]].lines[1].set_label(
                            '$\mathrm{R^{2}}$: '+f'{r_2}\n# obs: {round(n_r)}')

                    ax[ax_ix[0], ax_ix[1]] = self.format_axs(ax[ax_ix[0], ax_ix[1]], ylabs=axis_labs[count], xticks=True,
                                                             reverse=False, fontsize=self.fontsize, locator=None, scatter=True)

                    if latex:
                        print('& ' + mm + ' {:1.3f} & {:d}'.format(r_2, n_r))
                    else:
                        print('{} \t Slope: {} \t R2: {:1.3f} \t # obs: {:d}'.format(
                            axis_vars[count], mm, r_2, n_r))

                else:
                    ax[ax_ix[0], ax_ix[1]].axis('off')
                count += 1

            plt.xticks(ha='center')
            fig.align_ylabels([ax0, ax1, ax[2, 0]])
        if save:
            fig.savefig(self.figure_path(), bbox_inches='tight',
                        pad_inches=0.1, dpi=300)
