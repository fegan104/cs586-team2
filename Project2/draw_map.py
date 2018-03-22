import os
import numpy as np
from pandas import read_csv
from gmplot import gmplot
 
root = 'Map/'
nyc = [40.7127,-74.0059]
district = ['MN','QN','BX','BK','SI']

def draw(lat, lon, file_name):
    gmap = gmplot.GoogleMapPlotter(nyc[0], nyc[1], 10)
    gmap.scatter(lat, lon, 'Salmon', size=30, marker=False)
    gmap.draw(file_name) 

def main():
    for year in range(2014, 2017):
        lons, lats = np.array([]), np.array([])
        for d in district:
            if os.path.isdir(root+str(year)) == False:
                os.makedirs(root+str(year))
            # Read
            file = 'Block/'+str(year)+'/'+d+'.csv'
            df = read_csv(file)
            lat, lon = df['latitude'].as_matrix(), df['longitude'].as_matrix()
            # Draw by Borough
            draw(lat, lon, root+str(year)+'/'+d+'.html')
            lons, lats = np.r_[lon, lons], np.r_[lat, lats]
        # Draw by Year
        draw(lats, lons, root+str(year)+'/nyc.html')

if __name__ == '__main__':
    main()

