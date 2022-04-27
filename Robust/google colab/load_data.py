import pandas as pd
import numpy as np
from datetime import datetime,timedelta,date


class BikeNYCDataset():
    def __init__(self):
        self.service_region_num = 50
        self.read_data()
    
    def read_data(self):
        rawdata_df = pd.read_csv('data/demand-manhattan-202106.csv')
        rawdata_df['datetime'] = pd.to_datetime(rawdata_df['datetime'])
        rawdata_df = rawdata_df.sort_values(by=['datetime','service_region']).reset_index(drop=True)
        self.rawdata_np = rawdata_df['demand'].to_numpy().reshape((-1,self.service_region_num))
        rawdatatest_df = pd.read_csv('data/demand-manhattan-202107.csv')
        rawdatatest_df['datetime'] = pd.to_datetime(rawdatatest_df['datetime'])
        rawdatatest_df = rawdatatest_df.sort_values(by=['datetime','service_region']).reset_index(drop=True)
        self.rawdatatest_np = rawdatatest_df['demand'].to_numpy().reshape((-1,self.service_region_num))
    
    def scenarios(self,samplenum=None):
        if samplenum is None:
            samplenum = self.rawdata_np.shape[0]
        prob = np.ones([samplenum])
        prob /= samplenum
        scenario_np = self.rawdata_np[:samplenum,:]
        return scenario_np, prob
    
    def scenarios_test(self):
        samplenum = self.rawdatatest_np.shape[0]
        prob = np.ones([samplenum])
        prob /= samplenum
        scenario_np = self.rawdatatest_np[:samplenum,:]
        return scenario_np, prob
