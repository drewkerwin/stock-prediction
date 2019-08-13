from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

class InterestRateLoader:

	def parse(self, x):
		return pd.to_datetime(x, infer_datetime_format=True)
		
	def load_all_data(self, all=True, years=[]):
		
		df = pd.DataFrame()
		if(all==True):
			for i in range(1995, 2018):
				filename = "data/int-rate-{}.csv".format(str(i))
				data_temp = read_csv(filename, date_parser=self.parse, sep='\t', index_col=0, na_values=['nan'])
				df = df.append(data_temp)
				
			df.to_csv("data/int-combined.csv")
			df.columns = ['1_Mo','3_Mo','6_Mo','1_Yr','2_Yr','3_Yr','5_Yr','7_Yr','10_Yr','20_Yr','30_Yr']
			df.drop('1_Mo', axis=1, inplace=True)
			df.drop('30_Yr', axis=1, inplace=True)
		return df
		
def unit_test():
	loader = DataLoader()
	loader.load_all_data(all=True)
	
if __name__ =='__main__':
	unit_test()