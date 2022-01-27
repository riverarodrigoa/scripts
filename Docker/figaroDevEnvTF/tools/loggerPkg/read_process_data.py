import os
import re
import glob
# import numpy as np
import pandas as pd
from tqdm import tqdm
# from dateutil import parser
# from pandas import DataFrame
# from pandas.io.parsers import TextFileReader
from scipy.signal import savgol_filter as savgol
from scipy.signal import medfilt as median
import warnings
warnings.filterwarnings('ignore')


class Dataset(object):

    def __init__(self, root_path, files_name, id_data, ix='TS', separator=None, drop_na=True, subfolder_path=None, file_format=False,
                 last_n_files=None, offset=None, column_names=None,
                 skip_rows=0, skip_footer=0, parsedate=False, datetime=False, read_from_formated=False, format_dataset=False,
                 test_mode=False, test_level=0):
        self.root_path = root_path
        self.files_name = files_name
        self.raw_path = self.root_path + 'raw/'
        self.interim_path = self.root_path + 'interim/'
        self.files_path = self.raw_path if subfolder_path is None else self.raw_path + subfolder_path
        self.last_n_files = last_n_files
        self.offset = offset
        self.files = glob.glob1(self.files_path, self.files_name + "*")
        if self.offset is not None:
            self.files = self.files[:-self.offset]
        self.id = id_data
        self.interim_folder = self.interim_path + self.id + '/'
        self.format_flag = False
        # Flags read data
        self.ix = ix
        self.separator = "\s+" if separator is None else separator
        self.skip_rows = skip_rows
        self.skip_footer = skip_footer
        self.parsedate = parsedate
        self.datetime = datetime
        self.read_from_formated = read_from_formated
        self.format_dataset = format_dataset
        self.dn = drop_na

        self.test_mode = test_mode
        self.test_level = test_level
        self.column_names = column_names

        self.dataset = pd.DataFrame()

        if file_format:
            self.__format_files__()

        self.files_path = self.interim_folder if self.read_from_formated else self.files_path
        if self.read_from_formated:
            self.files = glob.glob1(self.interim_folder, self.files_name + "*")

        if self.last_n_files is not None:
            self.files = self.files[-self.last_n_files:]

    def __update_files__(self):
        self.files = glob.glob1(self.files_path, self.files_name + "*")

    def __get_version_file__(self):
        self.version_files = []
        regex = re.compile(r'[0-9]*[.][0-9]*')
        for file in self.files:
            with open(self.files_path + file, 'r') as f:
                first_line = f.readline()
                version = ''.join(regex.findall(first_line))
            self.version_files.append(version)

    def __check_dif_version__(self):
        self.changes = [0]
        prev = 0
        for n in range(1, len(self.version_files)):
            if self.version_files[prev] != self.version_files[n]:
                self.changes.append(n)
                prev = n
        self.changes.append(len(self.version_files))
        if self.test_mode:
            print(f'[TEST MODE] -| Changes: {self.changes}')

    def process_data(self, check_version=True):
        self.__update_files__()
        if check_version:
            self.__get_version_file__()
            self.__check_dif_version__()
        db = pd.DataFrame()
        if len(self.changes) > 2:
            if self.test_mode:
                print(f'[TEST MODE] -| Columns: {self.column_names[0]}')
            else:
                db = self.read_data(
                    by_version=[self.changes[0], self.changes[1]])
                if self.column_names is not None:
                    db.columns = self.column_names[0]

            for i in tqdm(range(0, len(self.changes)-1)):
                dataset = self.read_data(
                    by_version=[self.changes[i], self.changes[i + 1]])
                if self.test_mode:
                    print(f'[TEST MODE] -| Columns: {self.column_names[i]}')
                else:
                    if self.column_names is not None:
                        dataset.columns = self.column_names[i]
                db.append(dataset, ignore_index=True)

        else:
            db = self.read_data()
            if self.test_mode:
                print(f'[TEST MODE] -| Columns: {self.column_names}')
            else:
                if self.column_names is not None:
                    db.columns = self.column_names
        self.dataset = db.copy()

    def __format_files__(self):
        print('in format_files()')
        os.makedirs(self.interim_folder, exist_ok=True)
        regex = re.compile(r',$')

        for file in self.files:
            filename_r = self.files_path + file
            filename_w = self.interim_folder + file
            print(filename_r, filename_w)
            with open(filename_r, 'r') as f:
                names = []
                for line in f:
                    r = regex.sub('', line)
                    l = line+'\n' if r is None else r
                    names.append(l)

            with open(filename_w, 'w') as ff:
                ff.writelines(names)

    def read_data(self, by_version=None):
        if by_version is not None:
            batch_files = self.files[by_version[0]:by_version[1]]
        else:
            batch_files = self.files

        if self.test_mode:
            print(
                f'[TEST MODE] -| Batch files: {len(batch_files)}  | Datetime flag: {self.datetime} |')
            print(
                f'[TEST MODE] -| First file : {self.files_path + batch_files[0]}')
            if len(batch_files) > 2:
                if self.test_level == 1:
                    for i in range(1, len(batch_files)):
                        print(
                            f'[TEST MODE] -| File {i}: {self.files_path + batch_files[i]}')
                else:
                    print(
                        f'[TEST MODE] -| Path: {self.files_path} | # of files: {len(batch_files)}')
            return pd.DataFrame([[0, 0], [0, 0]], columns=['A', 'B'])
        else:
            if self.datetime:
                data = pd.read_csv(
                    self.files_path + batch_files[0], sep=self.separator, skiprows=self.skip_rows, skipfooter=self.skip_footer)
                if len(batch_files) > 2:
                    for i in tqdm(range(1, len(batch_files))):
                        d1 = pd.read_csv(
                            self.files_path + batch_files[i], sep=self.separator, skiprows=self.skip_rows, skipfooter=self.skip_footer)
                        data = data.append(d1, ignore_index=True)
            else:
                data = pd.read_csv(self.files_path + batch_files[0], sep=self.separator, skiprows=self.skip_rows, skipfooter=self.skip_footer,
                                   parse_dates=self.parsedate)
                if len(batch_files) > 2:
                    for i in tqdm(range(1, len(batch_files))):
                        # print(self.files[i])
                        d1 = pd.read_csv(self.files_path + batch_files[i], sep=self.separator, skiprows=self.skip_rows, skipfooter=self.skip_footer,
                                         parse_dates=self.parsedate)
                        data = data.append(d1, ignore_index=True)
            dataset = data
            # print(dataset.head())
            dataset[self.ix] = dataset[self.ix].astype('datetime64[ns]')
            dataset.set_index(self.ix, inplace=True)
            return dataset

    def save(self, pathtosave, name, format_file='csv'):
        try:
            if format_file is 'csv':
                self.dataset.to_csv(path_or_buf=pathtosave + name + '.csv')
            elif format_file is 'pickle':
                self.dataset.to_pickle(path=pathtosave + name + '.pkl')
            elif format_file is 'parquet':
                self.dataset.to_parquet(
                    fname=pathtosave + name+'.parquet', index=True)
            else:
                pass

        except NameError:
            print("[ERROR]: Format not implemented or incorrect. (Nothing saved).")

    def subset_obs(self, start_time=None, end_time=None, select_vars=None):
        start = self.dataset.index[0] if start_time is None else start_time
        end = self.dataset.index[-1] if end_time is None else end_time
        sel_vars = [
            i for i in self.dataset.columns] if select_vars is None else select_vars

        return self.dataset.loc[start:end, sel_vars]


if __name__ == '__main__':
    pass
