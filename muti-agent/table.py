
import matplotlib.pyplot as plt

# GDP data for the United States from 2000 to 2020
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
gdp = [10.5, 11.7, 12.2, 13.2, 14.3, 15.4, 16.7, 17.4, 18.0, 18.7, 19.4, 20.1, 21.0, 21.9, 22.6, 23.3, 24.0, 24.8, 25.5, 26.2, 27.0]

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(years, gdp, marker='o')
plt.title('United States GDP (2000-2020)')
plt.xlabel('Year')
plt.ylabel('GDP (in trillion USD)')
plt.grid(True)
plt.xticks(years)
plt.show()