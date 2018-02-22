import os,pickle,pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cols = ['Datetime','Humidity','Pressure','Temperature','Wind Speed']
se = pandas.read_csv('Atlanta.csv',names=cols,skiprows=1,index_col=False)
ne = pandas.read_csv('Boston.csv',names=cols,skiprows=1,index_col=False)
sw = pandas.read_csv('Phoenix.csv',names=cols,skiprows=1).append(pandas.read_csv('SanDiego.csv',names=cols,skiprows=1),ignore_index=True)
nw = pandas.read_csv('Seattle.csv',names=cols,skiprows=1).append(pandas.read_csv('Portland.csv',names=cols,skiprows=1),ignore_index=True)
mw = pandas.read_csv('KansasCity.csv',names=cols,skiprows=1).append(pandas.read_csv('SaintLouis.csv',names=cols,skiprows=1),ignore_index=True)

for col in cols[1:]:
	df = pandas.DataFrame()
	df['nw'] = nw[col].sample(2000)
	df['ne'] = ne[col].sample(2000)
	df['mw'] = mw[col].sample(2000)
	df['se'] = se[col].sample(2000)
	df['sw'] = sw[col].sample(2000)

	plt.xlabel('Region')
	plt.ylabel(col)
	df.boxplot(showfliers=False)
	plt.savefig(col+'.png')
	plt.gcf().clear()

