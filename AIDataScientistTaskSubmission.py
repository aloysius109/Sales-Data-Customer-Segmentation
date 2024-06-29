#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:53:59 2024

@author: kathrynhopkins
"""
# =============================================================================
# This code demonstrates cleaning, analysis, and visualisation methods for Sales, Product, and Customer data using Python.
# =============================================================================
#%% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import StrMethodFormatter
from datetime import datetime
import calendar
import seaborn as sns
from tabulate import tabulate
#%% Read in data and analyse and clean
data = pd.read_excel('encrypted_sales_data.xlsx')
# Summarise the data
data.head()
data.shape
data.info()
data.describe(include = 'all')
data.columns
data['Invoice ID'].value_counts(dropna = False)
len(data['Invoice ID'])
len(data['Invoice ID'].unique())
data['Invoice Status'].value_counts(dropna = False)
a = data['Invoice Number'].value_counts(dropna = False)
len(a)
data['Invoice Number']=data['Invoice Number'].str.replace("*", "")
data['Invoice Number']=data['Invoice Number'].str.replace("'", "")
data['Product Brand'].value_counts(dropna = False)
#Calculate total sales
data['Total Sales'] = data['Items.Purchase Price']*data['Invoice Items.Quantity']
# Remove inconsistent Invoice Number rows
data = data[:12288]
GrandTotalSales = data['Total Sales'].sum()
#Format the output
"${:,.2f}".format(GrandTotalSales)
print('The Total Sales is : ' "${:,.2f}".format(GrandTotalSales))
#%% Using K notation
# =============================================================================
# data['Total Sales'] = data['Total Sales'].apply(lambda x: "${:.1f}k".format((x/1000)))
# data['Total Sales']
# =============================================================================
#%% Format currency column
data['Total SalesFORMATTED'] = data['Total Sales'].apply(lambda x: "${:,.2f}".format((x)))
data['Total SalesFORMATTED']
#%% Calculate Average Sales
# Year Variable
data['Year'] = pd.DatetimeIndex(data['Invoices.Invoice Date']).year
# Or data['Year'] = data['Invoices.Invoice Date'].dt.year
data['Month'] = pd.DatetimeIndex(data['Invoices.Invoice Date']).month
data['Day'] = data['Invoices.Invoice Date'].dt.day
data['Weekday'] = data['Invoices.Invoice Date'].dt.day_name()
data['MonthName'] = data['Invoices.Invoice Date'].dt.month_name()
data['Quarter'] = data['Invoices.Invoice Date'].dt.quarter
# Create string Year-Month variable
data['Year-Month'] = data['Year'].apply(str) + '-' + data['Month'].apply(str)
# Create string Year-Quarter variable
data['Year-Quarter'] = data['Year'].apply(str) + '-' + data['Quarter'].apply(str)
#%% Total Sales each month
SalesYrMonth = data.groupby(data['Year-Month'])['Total Sales'].agg(np.sum)
SalesYrMonth = pd.DataFrame(SalesYrMonth)
SalesYrMonth.reset_index(inplace = True)
SalesYrMonth['Year-Month'] = pd.Categorical(SalesYrMonth['Year-Month'],
                                             ['2019-11',
                                              '2019-12', 
                                              '2020-1',
                                              '2020-2',
                                              '2020-3',
                                              '2020-4',
                                              '2020-5',
                                              '2020-6',
                                              '2020-7',
                                              '2020-8',
                                              '2020-9',
                                              '2020-10',
                                              '2020-11',
                                              '2020-12',
                                              '2021-1',
                                              '2021-2',
                                              '2021-3',
                                              '2021-4',
                                              '2021-5',
                                              '2021-6',
                                              '2021-7',
                                              '2021-8',
                                              '2021-9',
                                              '2021-10',
                                              '2021-11',
                                              '2021-12',
                                              '2022-1',
                                              '2022-2',
                                              '2022-3',
                                              '2022-4',
                                              '2022-5',
                                              '2022-6',
                                              '2022-7',
                                              '2022-8',
                                              '2022-9',
                                              '2022-10',
                                              '2022-11',
                                              '2022-12',
                                              '2023-1',
                                              '2023-2',
                                              '2023-3',
                                              '2023-4',
                                              '2023-5',
                                              '2023-6',
                                              '2023-7',
                                              '2023-8',
                                              '2023-9',
                                              '2023-10',
                                              '2023-11',
                                              '2023-12',
                                              '2024-1',
                                              '2024-2',
                                              '2024-3',
                                              '2024-4',
                                              '2024-5',
                                              '2024-6'])
SalesYrMonth.sort_values('Year-Month', ascending = True, inplace = True)
SalesYrMonth.reset_index(inplace = True)
SalesYrMonth.drop(['index'], axis = 1, inplace = True)
SalesYrMonth['Formatted'] = SalesYrMonth['Total Sales'].apply(lambda x: "${:,.0f}".format((x)))
#%% Create data for plotting
labels = SalesYrMonth['Year-Month']
pos = np.arange(len(labels))
values = SalesYrMonth['Total Sales'].values
points = values
#points = pd.Series(values, index = pos.astype(str))
formatted = pd.Series(SalesYrMonth['Formatted'], index = pos)
#%% And plot
fig, ax = plt.subplots(figsize = (50,10))
plt.plot(values, '-o', linewidth = 3, markersize = 5, color = '#002664')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.xticks(pos, labels, size = 12, rotation = 45)
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
plt.yticks(size = 12)
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
# Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Annotate high and low points
ymax = max(points)
[i for i, e in enumerate(points) if e == ymax] 
for idx, point in enumerate(points):
    if point == ymax:
        print(idx, point)
for idx, point in enumerate(points):
    print(idx, point)
ymin = min(points)
[i for i, e in enumerate(points) if e == ymin]
plt.plot(12, ymax, 'go', ms = 20, mfc = 'none', mec = 'g', mew = 2)
plt.text(12+2, ymax + 20, labels[12] +': '+ SalesYrMonth['Formatted'][12])
plt.plot(5, ymin, 'ro', ms = 20, mfc = 'none', mec = 'r', mew = 2)
plt.text(5+2, ymin-10, labels[5] + ' ' + SalesYrMonth['Formatted'][5])
plt.plot(23, points[23], 'go', ms = 20, mfc = 'none', mec = 'g', mew = 2)
plt.text(23+2, points[23], labels[23] + ': ' + SalesYrMonth['Formatted'][23])
plt.plot(55, points[55], 'ro', ms = 20, mfc = 'none', mec = 'r', mew = 2)
plt.text(55+2, points[55], labels[55] + ': ' + SalesYrMonth['Formatted'][55])
plt.text(5+2, ymin-10, labels[5] + ' ' + SalesYrMonth['Formatted'][5])
# Add labels and titles
plt.title('Total Sales by Month',color = '#000000',fontsize=18, pad = 20)
plt.xlabel('Year and Month',color = '#000000', fontsize = 20, labelpad = 30)
plt.ylabel('Sales', color = '#000000',fontsize = 20, labelpad = 30)
plt.show()
plt.savefig('TotalSalesByMonth.png')
#%% Average Sales Each Month
AveSalesYrMonth = data.groupby(data['Year-Month'])['Total Sales'].agg(np.mean)
AveSalesYrMonth = pd.DataFrame(AveSalesYrMonth)
AveSalesYrMonth.reset_index(inplace = True)
AveSalesYrMonth['Year-Month'] = pd.Categorical(AveSalesYrMonth['Year-Month'],
                                              ['2019-11',
                                              '2019-12', 
                                              '2020-1',
                                              '2020-2',
                                              '2020-3',
                                              '2020-4',
                                              '2020-5',
                                              '2020-6',
                                              '2020-7',
                                              '2020-8',
                                              '2020-9',
                                              '2020-10',
                                              '2020-11',
                                              '2020-12',
                                              '2021-1',
                                              '2021-2',
                                              '2021-3',
                                              '2021-4',
                                              '2021-5',
                                              '2021-6',
                                              '2021-7',
                                              '2021-8',
                                              '2021-9',
                                              '2021-10',
                                              '2021-11',
                                              '2021-12',
                                              '2022-1',
                                              '2022-2',
                                              '2022-3',
                                              '2022-4',
                                              '2022-5',
                                              '2022-6',
                                              '2022-7',
                                              '2022-8',
                                              '2022-9',
                                              '2022-10',
                                              '2022-11',
                                              '2022-12',
                                              '2023-1',
                                              '2023-2',
                                              '2023-3',
                                              '2023-4',
                                              '2023-5',
                                              '2023-6',
                                              '2023-7',
                                              '2023-8',
                                              '2023-9',
                                              '2023-10',
                                              '2023-11',
                                              '2023-12',
                                              '2024-1',
                                              '2024-2',
                                              '2024-3',
                                              '2024-4',
                                              '2024-5',
                                              '2024-6'])
AveSalesYrMonth.sort_values('Year-Month', ascending = True, inplace = True)
AveSalesYrMonth.reset_index(inplace = True)
AveSalesYrMonth.drop(['index'], axis = 1, inplace = True)
AveSalesYrMonth['Formatted'] = AveSalesYrMonth['Total Sales'].apply(lambda x: "${:,.2f}".format((x)))
AveSalesYrMonth.rename(columns = {'Total Sales':'Average Sales'}, inplace = True)
#%% Create data for plotting
labels = AveSalesYrMonth['Year-Month']
pos = np.arange(len(labels))
values = AveSalesYrMonth['Average Sales']
GrandAverage = np.average(AveSalesYrMonth['Average Sales'])
GrandAverageFormatted = '${:,.2f}'.format(GrandAverage)
#%% And plot
fig, ax = plt.subplots(figsize = (20,15))
plt.plot(values, '-o', linewidth = 3, markersize = 5, color = 'lightslategray')
plt.hlines(GrandAverage, min(pos), max(pos), linestyle = '--',color = 'black')
plt.text(max(pos), GrandAverage+10,'Grand Average: ' +GrandAverageFormatted)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.xticks(pos, labels, size = 12, rotation = 45)
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
plt.yticks(size = 12)
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
# Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
ymax = max(values)
ymin = min(values)
[i for i, e in enumerate(values) if e == ymax]
[i for i, e in enumerate(values) if e == ymin]
for idx, value in enumerate(values):
    print(idx, value)
plt.plot(44, ymax, 'go', ms = 20, mfc = 'none', mec = 'g', mew = 2)
plt.text(44, ymax + 10, labels[44] +': '+ AveSalesYrMonth['Formatted'][44])
plt.plot(55, ymin, 'ro', ms = 20, mfc = 'none', mec = 'r', mew = 2)
plt.text(55+2, ymin-10, labels[55] + ': ' + AveSalesYrMonth['Formatted'][55])
# Add title
plt.title('Average Sales per Day',color = '#000000',fontsize=16, pad = 00)
plt.xlabel('Year and Month',color = '#000000', fontsize = 12, labelpad = 10)
plt.ylabel('Sales', color = '#000000',fontsize = 14, labelpad = 10)
plt.show()
plt.savefig('AveSalesPerDay.png')
#%% Total Sales Each Year 2020 to 2023 only (full years)
TotSalesYr = data.groupby(data['Year'])['Total Sales'].agg(np.sum)
TotSalesYr = pd.DataFrame(TotSalesYr)
TotSalesYr.reset_index(inplace = True)
# Select only 2020 to 2023 inclusive
TotSalesYr = TotSalesYr[TotSalesYr['Year'].isin([2020, 2021, 2022, 2023])]
TotSalesYr.sort_values('Year', ascending = True, inplace = True)
TotSalesYr.reset_index(inplace = True)
TotSalesYr.drop(['index'], axis = 1, inplace = True)
TotSalesYr['Formatted'] = TotSalesYr['Total Sales'].apply(lambda x: "${:,.2f}".format((x)))
#%% Create data for plotting
labels = TotSalesYr['Year']
pos = np.arange(len(labels))
values = TotSalesYr['Total Sales']
formatted = TotSalesYr['Formatted']
ymax = max(values)
ymin = min(values)
#%% And plot
fig, ax = plt.subplots(figsize = (30,10))
plt.plot(values, '-o', markersize = 5,linewidth = 2, color = 'darkslategrey')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.xticks(pos, labels, size = 12, rotation = 0)
plt.yticks(size = 12)
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
# Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# And annotate
[i for i, e in enumerate(values) if e == ymax]
[i for i, e in enumerate(values) if e == ymin]
plt.plot(pos[1], ymax, 'go', ms = 20, mfc = 'none', mec = 'g', mew = 2)
plt.annotate(formatted[1], (pos[1],values[1]), textcoords = 'offset points', xytext = (60,0), ha = 'center', size = 12, color = 'black', weight = 'bold')
plt.plot(pos[3], ymin, 'ro', ms = 20, mfc = 'none', mec = 'r', mew = 2)
plt.annotate(formatted[3], (pos[3],values[3]), textcoords = 'offset points', xytext = (60,0), ha = 'center', size = 12, color = 'black', weight = 'bold')
# Add title and labels
plt.title('Total Sales 2020 to 2023',color = '#000000',fontsize=16, pad = 20)
plt.xlabel('Year',color = '#000000', fontsize = 12, labelpad = 10)
plt.ylabel('Sales', color = '#000000',fontsize = 12, labelpad = 10)
plt.show()
plt.savefig('TotalSalesEachYear.png')
#%% Average Sales Each Month By Year
AveSalesYr = data.groupby(['Year', 'Month'])['Total Sales'].agg(np.sum)
AveSalesYr = pd.DataFrame(AveSalesYr)
AveSalesYr.reset_index(inplace = True)
AveSalesYr = AveSalesYr.groupby('Year')['Total Sales'].agg(np.mean)
AveSalesYr = pd.DataFrame(AveSalesYr)
AveSalesYr.reset_index(inplace = True)
AveSalesYr.sort_values('Year', ascending = True, inplace = True)
AveSalesYr.rename(columns = {'Total Sales':'AverageSalesPerMonth'}, inplace = True)
AveSalesYr['Formatted'] = AveSalesYr['AverageSalesPerMonth'].apply(lambda x: "${:,.0f}".format((x)))
#%% Create data for plotting
labels = AveSalesYr['Year']
pos = np.arange(len(labels))
values = AveSalesYr['AverageSalesPerMonth']
formatted = AveSalesYr['Formatted']
ymax = max(values)
ymin = min(values)
#%% And plot
fig, ax = plt.subplots(figsize = (30,10))
plt.plot(values, '-o', linewidth = 3, color = 'lightslategrey')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.xticks(pos, labels, size = 12, rotation = 90)
plt.yticks(size = 12)
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
# Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# And annotate
[i for i, e in enumerate(values) if e == ymax]
[i for i, e in enumerate(values) if e == ymin]
plt.plot(pos[2], ymax, 'go', ms = 20, mfc = 'none', mec = 'g', mew = 2)
plt.annotate(formatted[2], (pos[2],values[2]), textcoords = 'offset points', xytext = (60,0), ha = 'center', size = 12, color = 'black', weight = 'bold')
plt.plot(pos[5], ymin, 'ro', ms = 20, mfc = 'none', mec = 'r', mew = 2)
plt.annotate(formatted[5], (pos[5],values[5]), textcoords = 'offset points', xytext = (60,0), ha = 'center', size = 12, color = 'black', weight = 'bold')
#Add title and labels
plt.title('Average Sales each Month',color = '#000000',fontsize=16, pad = 20)
plt.xlabel('Year',color = '#000000', fontsize = 12, labelpad = 30)
plt.ylabel('Sales per Month', color = '#000000',fontsize = 12, labelpad = 30)
plt.show()
plt.savefig('AveSalesPerMonth.png')
#%% Total Sales each Quarter
SalesYrQtr = data.groupby(data['Year-Quarter'])['Total Sales'].agg(np.sum)
SalesYrQtr = pd.DataFrame(SalesYrQtr)
SalesYrQtr.reset_index(inplace = True)
SalesYrQtr['Year-Quarter'] = pd.Categorical(SalesYrQtr['Year-Quarter'],
                                             ['2019-4',
                                              '2020-1',
                                              '2020-2',
                                              '2020-3',
                                              '2020-4',
                                              '2021-1',
                                              '2021-2',
                                              '2021-3',
                                              '2021-4',
                                              '2022-1',
                                              '2022-2',
                                              '2022-3',
                                              '2022-4',
                                              '2023-1',
                                              '2023-2',
                                              '2023-3',
                                              '2023-4',
                                              '2024-1',
                                              '2024-2'])
SalesYrQtr.sort_values('Year-Quarter', ascending = True, inplace = True)
SalesYrQtr.reset_index(inplace = True)
SalesYrQtr.drop(['index'], axis = 1, inplace = True)
SalesYrQtr['Formatted'] = SalesYrQtr['Total Sales'].apply(lambda x: "${:,.2f}".format((x)))
#%% Create data for plotting
labels = SalesYrQtr['Year-Quarter']
pos = np.arange(len(labels))
values = SalesYrQtr['Total Sales']
formatted = SalesYrQtr['Formatted']
ymax = max(values)
ymin = min(values)
#%% And plot
fig, ax = plt.subplots(figsize = (20,10))
plt.plot(values, '-o', markersize = 5, linewidth = 2, color = '#002664')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.xticks(pos, labels, size = 12, rotation = 90)
plt.yticks(size = 12)
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
# Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# And annotate
# Get ymax and ymin indices
[i for i, e in enumerate(values) if e == ymax]
[i for i, e in enumerate(values) if e == ymin]
[i for i, e in enumerate(values)]
plt.plot(pos[8], ymax, 'go', ms = 20, mfc = 'none', mec = 'g', mew = 2)
plt.annotate(formatted[8], (pos[8],values[8]), textcoords = 'offset points', xytext = (60,0), ha = 'center', size = 12, color = 'black', weight = 'bold')
plt.plot(pos[0], ymin, 'ro', ms = 20, mfc = 'none', mec = 'r', mew = 2)
plt.annotate('(part quarter): ' + formatted[0], (pos[0],values[0]), textcoords = 'offset points', xytext = (110,0), ha = 'center', size = 12, color = 'black', weight = 'regular')
plt.plot(pos[18], values[18], 'ro', ms = 20, mfc = 'none', mec = 'r', mew = 2)
plt.annotate('(to 19 June): ' + formatted[18], (pos[18],values[18]), textcoords = 'offset points', xytext = (50,-25), ha = 'center', size = 12, color = 'black', weight = 'regular')
plt.title('Total Sales by Quarter',color = '#000000',fontsize=16, pad = 10)
plt.xlabel('Year and Quarter',color = '#000000', fontsize = 12, labelpad = 10)
plt.ylabel('Sales', color = '#000000',fontsize = 12, labelpad = 10)
plt.show()
plt.savefig('TotalSalesbyQuarter.png')
#%% Look at seasonal trends, first by Quarter (mean)
SalesYrQtr = data.groupby(data['Year-Quarter'])['Total Sales'].agg(np.sum)
SalesYrQtr = pd.DataFrame(SalesYrQtr)
SalesYrQtr.reset_index(inplace = True)
#Split to get the quarter
SalesYrQtr[['Year', 'Quarter']] = SalesYrQtr['Year-Quarter'].str.split('-', expand = True)
SalesQuarter = SalesYrQtr.groupby(SalesYrQtr['Quarter'])['Total Sales'].agg(np.mean)
SalesQuarter = pd.DataFrame(SalesQuarter)
SalesQuarter.reset_index(inplace = True)
SalesQuarter['Formatted'] = SalesQuarter['Total Sales'].apply(lambda x: "${:,.0f}".format((x)))
# Relabel Total Sales to Average Sales
SalesQuarter.rename(columns = {'Total Sales':'Average Sales'}, inplace = True)
#%% Create data for plotting
labels = SalesQuarter['Quarter']
pos = np.arange(len(labels))
values = SalesQuarter['Average Sales']
formattedvalues = SalesQuarter['Formatted']
#%% and plot
width = 0.8
fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(labels, values, width, color='darkslategrey', label= 'Sales')
plt.xticks(size = 12)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
ax.annotate(formattedvalues[0], (labels[0],values[0]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 20, color = 'white', weight = 'bold')
ax.annotate(formattedvalues[1], (labels[1],values[1]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 20, color = 'red', weight = 'bold')
ax.annotate(formattedvalues[2], (labels[2],values[2]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 20, color = 'white', weight = 'bold')
ax.annotate(formattedvalues[3], (labels[3],values[3]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 20, color = 'white', weight = 'bold')
# # Add title and labels
plt.xlabel('Quarter',color = '#000000', fontsize = 20, labelpad = 10)
plt.ylabel('Sales', color = '#000000',fontsize = 20)
ax.set_title("Seasonal Trends: Average Sales by Quarter, Q4 2019 to Q2 2024", color = '#000000', fontsize = 16, pad = 10)
# Save the plot
plt.show()
plt.savefig('SeasonalbyQuarter.png')
#%% Look at seasonal trends, by month
SalesYrMonth = data.groupby(data['Year-Month'])['Total Sales'].agg(np.sum)
SalesYrMonth = pd.DataFrame(SalesYrMonth)
SalesYrMonth.reset_index(inplace = True)
#Split to get the quarter
SalesYrMonth[['Year', 'Month']] = SalesYrMonth['Year-Month'].str.split('-', expand = True)
SalesMonth = SalesYrMonth.groupby(SalesYrMonth['Month'])['Total Sales'].agg(np.mean)
SalesMonth = pd.DataFrame(SalesMonth)
SalesMonth.reset_index(inplace = True)
SalesMonth['Month']=SalesMonth['Month'].apply(int)
SalesMonth.sort_values(by = 'Month', ascending = True, inplace = True)
SalesMonth.reset_index(inplace = True)
# Name each month
SalesMonth['month_names'] = SalesMonth['Month'].apply(lambda x: calendar.month_name[x])
SalesMonth['Formatted'] = SalesMonth['Total Sales'].apply(lambda x: "${:,.0f}".format((x)))
# Relabel Total Sales to Average Sales
SalesMonth.rename(columns = {'Total Sales':'Average Sales'}, inplace = True)
#%% Create data for plotting
labels = SalesMonth['month_names']
pos = np.arange(len(labels))
values = SalesMonth['Average Sales']
formattedvalues = SalesMonth['Formatted']
#%% and plot
width = 0.8
fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(labels, values, width, color='darkslategrey', label= 'Sales')
plt.xticks(size = 12, rotation = 45)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
ax.annotate(formattedvalues[0], (labels[0],values[0]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'red', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[1], (labels[1],values[1]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[2], (labels[2],values[2]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[3], (labels[3],values[3]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[4], (labels[4],values[4]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'orange', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[5], (labels[5],values[5]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[6], (labels[6],values[6]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[7], (labels[7],values[7]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[8], (labels[8],values[8]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[9], (labels[9],values[9]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[10], (labels[10],values[10]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
ax.annotate(formattedvalues[11], (labels[11],values[11]), textcoords = 'offset points', xytext = (0,-80), ha = 'center', size = 14, color = 'white', weight = 'bold', rotation = 90)
# # Add title and labels
plt.xlabel('Month',color = '#000000', fontsize = 12, labelpad = 10)
plt.ylabel('Sales', color = '#000000',fontsize = 12)
ax.set_title("Seasonal Trends: Average Sales by Month, Nov 2019 to June 2024", color = '#000000', fontsize = 16, pad = 10)
#ax.legend(loc="upper center", fontsize = 14)
# Save the plot
plt.show()
plt.savefig('SeasonalByMonth.png')
#%% Look at seasonal trends, by day of week
SalesWeekday = data.groupby(data['Weekday'])['Total Sales'].agg(np.mean)
SalesWeekday = pd.DataFrame(SalesWeekday)
SalesWeekday.reset_index(inplace = True)
# Put the weekdays in order
SalesWeekday['Weekday'] = pd.Categorical(SalesWeekday['Weekday'],
                                             ['Monday',
                                              'Tuesday', 
                                              'Wednesday',
                                              'Thursday',
                                              'Friday',
                                              'Saturday',
                                              'Sunday'])
SalesWeekday.sort_values('Weekday', ascending = True, inplace = True)
SalesWeekday['Formatted'] = SalesWeekday['Total Sales'].apply(lambda x: "${:,.0f}".format((x)))
# Relabel Total Sales to Average Sales
SalesWeekday.rename(columns = {'Total Sales':'Average Sales'}, inplace = True)
SalesWeekday.reset_index(inplace = True)
#%% Create data for plotting
labels = SalesWeekday['Weekday']
pos = np.arange(len(labels))
values = SalesWeekday['Average Sales']
formattedvalues = SalesWeekday['Formatted']
#%% and plot
width = 0.8
fig, ax = plt.subplots(figsize=(15, 10))
ax.bar(labels, values, width, color='darkblue', label= 'Sales')
plt.xticks(size = 16, rotation = 0)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
ax.annotate(formattedvalues[0], (labels[0],values[0]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 16, color = 'white', weight = 'bold')
ax.annotate(formattedvalues[1], (labels[1],values[1]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 16, color = 'red', weight = 'bold')
ax.annotate(formattedvalues[2], (labels[2],values[2]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 16, color = 'orange', weight = 'bold')
ax.annotate(formattedvalues[3], (labels[3],values[3]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 16, color = 'white', weight = 'bold')
ax.annotate(formattedvalues[4], (labels[4],values[4]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 16, color = 'white', weight = 'bold')
ax.annotate(formattedvalues[5], (labels[5],values[5]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 16, color = 'white', weight = 'bold')
ax.annotate(formattedvalues[6], (labels[6],values[6]), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 16, color = 'white', weight = 'bold')
# # Add title and labels
plt.xlabel('Day',color = '#000000', fontsize = 12, labelpad = 10)
plt.ylabel('Sales', color = '#000000',fontsize = 12)
ax.set_title("Average Sales by Day of Week, Nov 2019 to June 2024", color = '#000000', fontsize = 16, pad = 10)
#ax.legend(loc="upper center", fontsize = 14)
# Display the plot
plt.show()
plt.savefig('DayofWeekSales.png')
#%% Top 100 Customers by Total Sales Value
TopCustomers = data.groupby('Customer Name')['Total Sales'].agg(np.sum)
TopCustomers = pd.DataFrame(TopCustomers)
TopCustomers.reset_index(inplace = True)
TopCustomers.sort_values('Total Sales', ascending = False, inplace = True)
TotalSales = TopCustomers['Total Sales'].sum()
TopCustomers= TopCustomers.head(100)
TopCustomers.reset_index(inplace = True)
TopCustomersSales = TopCustomers['Total Sales'].sum()
TopCustomers['Formatted'] = TopCustomers['Total Sales'].apply(lambda x: "${:,.0f}".format((x)))
TopCustomers.drop(['index', 'Total Sales'], axis = 1, inplace = True)
TopCustomers.rename(columns = {'Formatted':'Total Sales'}, inplace = True)
TopCustomers
a = tabulate(TopCustomers, headers = ['Rank', 'Customer Number', 'Total Sales'], tablefmt = 'pretty')
TopCustomers.to_csv('TopCustomers.csv')
TopCustomersSales/TotalSales
TotalCustomers = data['Customer Name'].unique()
len(TotalCustomers)
#%% Plot table
ax01 = plt.subplot(111, frame_on=False) # no visible frame
ax01.xaxis.set_visible(False)  # hide the x axis
ax01.yaxis.set_visible(False)  # hide the y axis
table03 = pd.plotting.table(ax01, TopCustomers[:50], loc = 'center', cellLoc = 'center', colWidths=list([.2, .2]))
plt.savefig('TopCustomersTable03.png')
ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
table02 = pd.plotting.table(ax, TopCustomers[50:], loc = 'center', cellLoc = 'center', colWidths=list([.2, .2]))
plt.savefig('TopCustomersTable02.png')
table02.scale(1,2)
#%% Customer purchase frequency and average order value
# Customer purchase frequency
# For the Top Customers, find the invoice dates
TopCustomers = TopCustomers.merge(data, left_on = 'Customer Name', right_on = 'Customer Name', how = 'inner')
# Now find the start date of purchasing (min invoice date)
PurchaseFrequency = TopCustomers.groupby('Customer Name').agg({'Invoices.Invoice Date':[np.min, np.max, np.size]})
# Add new column subtracting the start from the end date
PurchaseFrequency.columns
PurchaseFrequency.columns = PurchaseFrequency.columns.droplevel(0)
PurchaseFrequency.columns
PurchaseFrequency['Period'] = PurchaseFrequency['amax']-PurchaseFrequency['amin']
PurchaseFrequency['Frequency'] = PurchaseFrequency['Period']/PurchaseFrequency['size']
PurchaseFrequency['Frequency'].dtypes
PurchaseFrequency.sort_values(['Frequency'], ascending = True, inplace = True)
PurchaseFrequency['Frequency'] = (PurchaseFrequency['Frequency']).astype(str)
a = PurchaseFrequency['Frequency'].str.split(' ', expand = True)
PurchaseFrequency = PurchaseFrequency.merge(a, left_index = True, right_index = True, how = 'inner')
#%%Plot a histogram of purchase frequency of the Top 100 Customers
PurchaseFrequency.rename(columns = {0:'Purchase Rate'}, inplace = True)
PurchaseFrequency.reset_index(inplace = True)
PurchaseFrequency['Purchase Rate'].describe()
PurchaseFrequency['Purchase Rate'].info()
#%% Plot purchase history on histogram
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(PurchaseFrequency['Purchase Rate'], facecolor='gray', edgecolor='white')
# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)
plt.yticks(size = 12)
plt.xticks(size = 12, rotation = 0)
ax.set_ylim([0,25])
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Give ourselves some more room at the bottom of the plot
plt.subplots_adjust(bottom=0.15)
plt.xlabel('Purchase Frequency (days)',color = '#000000', fontsize = 12, labelpad = 10)
plt.ylabel('No. of Customers', color = '#000000',fontsize = 12)
ax.set_title("Purchase Frequency Histogram", color = '#000000', fontsize = 16, pad = 10)
plt.show()
plt.savefig('PurchaseFreqHist.png')
#%% And a boxplot
#%% Create data for boxplot
medianPurchaseRate = PurchaseFrequency['Purchase Rate'].median()
meanPurchaseRate = PurchaseFrequency['Purchase Rate'].mean()
boxdata = PurchaseFrequency['Purchase Rate'].astype(int)
boxdata = pd.DataFrame(boxdata)
#%%
fig, ax = plt.subplots(figsize=(10, 15))
boxprops = dict(linestyle='-', linewidth=3, color='black')
flierprops = dict(marker='o', markerfacecolor='black', markersize=8,
                  markeredgecolor='none')
medianprops = dict(linestyle='-', linewidth=2.5, color='black')
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='black')
meanlineprops = dict(linestyle='--', linewidth=2.5, color='black')
whiskerprops = dict(linestyle='--', linewidth = 1, color = 'black')

#Plot 
boxdata.boxplot(patch_artist = True, notch = True, showbox = False, medianprops = medianprops, flierprops = flierprops, showcaps = True, showmeans = True, meanline = True, whiskerprops = whiskerprops)
plt.grid(False)
plt.yticks(size = 12)
plt.ylabel('Days')
plt.xlabel('Days between Purchases')
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
#Annotate
ax.annotate('Median: 25.5 days', (1,25), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 12, color = 'black', weight = 'regular')
ax.annotate('Mean: 32.7 days', (1,33), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 12, color = 'green', weight = 'regular')
ax.annotate('Infrequent Purchase Customers', (1,110), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 12, color = 'black', weight = 'regular')
ax.set_ylim([0,130])
plt.ylabel('No. of Days between Purchases', color = '#000000',fontsize = 12)
ax.set_title("Customer Purchase Frequency Boxplot", color = '#000000', fontsize = 16, pad = 10)
plt.show()
plt.savefig('PurchaseFreqBoxPlot.png')
#%%Average order value
AverageOrder = TopCustomers.groupby('Customer Name')['Total Sales_y'].agg([np.mean, np.max, np.min, np.size])
AverageOrder = pd.DataFrame(AverageOrder)
AverageOrder.reset_index(inplace = True)
AverageOrder.sort_values('mean', ascending = False, inplace = True)
AverageOrder.reset_index(inplace = True)
# Create formatted columns
AverageOrder['FormattedMean'] = AverageOrder['mean'].apply(lambda x: "${:,.0f}".format((x)))
AverageOrder['FormattedMin'] = AverageOrder['amin'].apply(lambda x: "${:,.0f}".format((x)))
AverageOrder['FormattedMax'] = AverageOrder['amax'].apply(lambda x: "${:,.0f}".format((x)))
# Truncate customer name
b = AverageOrder['Customer Name'].str.split('_', expand = True)
# Rejoin to dataset
AverageOrder= AverageOrder.merge(b, left_index = True, right_index = True, how = 'inner')
AverageOrder.rename(columns = {1:'Customer Number'}, inplace = True)
#Plot charts
#%% Create plotting data
customers = AverageOrder['Customer Number'][:50]
pos = np.arange(len(customers))
average=AverageOrder['mean'][:50]
AverageOrderValue = AverageOrder['mean'][:50].mean()
#%% Create line plot
fig, ax = plt.subplots(figsize=(20, 10))
plt.plot(average, '-o', markersize = 5, linewidth = 2, color = '#002664')
plt.xticks(pos, customers, size = 12, rotation = 90)
plt.yticks(size = 14)
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add title and labels
for x,y in zip(pos,average):
    label = "${:,.0f}".format(y)
    if x % 2 ==0:
        plt.annotate(label, 
                 (x,y), 
                 textcoords="offset points", 
                 xytext=(10,10),
                 ha='center') 
plt.title('Top 50 Customers: Average Order Value',color = '#000000',fontsize=16, pad = 20)
plt.xlabel('Customer Number',color = '#000000', fontsize = 14, labelpad = 30)
plt.ylabel('Order Value', color = '#000000',fontsize = 14, labelpad = 30)
plt.show()
#plt.yticks([])
plt.savefig('AveOrderValue.png')
#%% Now create bubble plot
customers = AverageOrder['Customer Number'][1:50]
pos = np.arange(len(customers))
average=AverageOrder['mean'][1:50]
size=AverageOrder['size'][1:50].astype(str)
#%%
fig, ax = plt.subplots(figsize=(40, 10))
plt.scatter(customers, average, s = AverageOrder['size'][1:50]*10, alpha = 0.5)
plt.xticks(pos, customers, size = 14, rotation = 90)
plt.yticks(size = 14)
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add title and labels
# Add title and labels
for x,y,z in zip(pos,average, size):
        plt.annotate(z, 
                 (x,y), 
                 textcoords="offset points", 
                 xytext=(0,0),
                 ha='center') 
plt.title('Top 50 Customers: Average Order Value and Volume, Nov 2019 to Jun 2024',color = '#000000',fontsize=16, pad = 20)
plt.xlabel('Customer Number',color = '#000000', fontsize = 14, labelpad = 30)
plt.ylabel('Order Value', color = '#000000',fontsize = 14, labelpad = 30)
# Annotate
# plt.annotate(AverageOrder['FormattedMean'][0], (pos[0],average[0]), textcoords = 'offset points',  xytext=(0,-10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.savefig('CustomerBubble.png')
#%% Customer Lifetime Value
TotalGrossProfit = TopCustomers.groupby('Customer Name')['Total_GP'].agg([np.sum, np.mean])
TotalGrossProfit.rename(columns = {'sum':'Total Gross Profit', 'mean':'Average Gross Profit'}, inplace = True)
TotalGrossProfit.reset_index(inplace = True)
# Add the average order values from before
TotalGrossProfit = TotalGrossProfit.merge(AverageOrder, left_on = 'Customer Name', right_on = 'Customer Name', how = 'inner')
#%% Calcualte Expected Lifespan
PurchaseFrequency['Period'] = (PurchaseFrequency['Period']).astype(str)
ExpectedLifespan = PurchaseFrequency['Period'].str.split(expand = True)
ExpectedLifespan.reset_index(inplace = True)
ExpectedLifespan.rename(columns = {0:'Lifespan'}, inplace = True)
ExpectedLifespan['Lifespan'] = ExpectedLifespan['Lifespan'].map(int)
ExpectedLifespan['Lifespan'].describe()
ExpectedLifespan = ExpectedLifespan[['Customer Name', 'Lifespan']]
TotalGrossProfit = TotalGrossProfit.merge(ExpectedLifespan, left_on='Customer Name', right_on = 'Customer Name', how = 'inner')
TotalGrossProfit['Lifespan']=TotalGrossProfit['Lifespan'].astype(int)
AverageLifespan = TotalGrossProfit['Lifespan'].mean()
TotalGrossProfit['Profit per Day'] = TotalGrossProfit['Total Gross Profit'] / TotalGrossProfit['Lifespan']
TotalGrossProfit['CLV'] = TotalGrossProfit['Profit per Day'] * AverageLifespan
TotalGrossProfit['FormattedCLV'] = TotalGrossProfit['CLV'].apply(lambda x: "CLV: ${:,.0f}".format((x)))
TotalGrossProfit.sort_values(by = 'CLV', ascending = False, inplace = True)
TotalGrossProfit.reset_index(inplace = True)
#%% Create data for line plot
customers = TotalGrossProfit['Customer Number'][:50]
pos = np.arange(len(customers))
clv = TotalGrossProfit['CLV'][:50]
formattedclv = TotalGrossProfit['FormattedCLV'][:50]
#%%
fig, ax = plt.subplots(figsize=(20, 10))
plt.plot(pos, clv)
plt.xticks(pos, customers, size = 12, rotation = 90)
plt.yticks(size = 12)
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)

# Add title and labels
for x,y,z in zip(pos,clv, formattedclv):
    if x % 6 ==0:
        plt.annotate(z, 
                 (x,y), 
                 textcoords="offset points", 
                 xytext=(10,-20),
                 ha='center') 
plt.title('Top 50 Customers: Customer Lifetime Value (CLV)',color = '#000000',fontsize=16, pad = 20)
plt.xlabel('Customer Number',color = '#000000', fontsize = 12, labelpad = 30)
plt.ylabel('Order Value', color = '#000000',fontsize = 12, labelpad = 30)
plt.savefig('CLV.png')
#%% Gross Profit Margins
Products = data.groupby('Product Name').agg({'Total Sales':np.sum,
                                             'Invoice Items.Quantity':[np.sum],
                                             'Total_GP':np.sum})
Products.columns
Products.sort_values(by = ('Total_GP', 'sum'), ascending = False, inplace=True)
Products.columns = Products.columns.droplevel(0)

Products.reset_index(inplace = True)
Products.columns =['Product Name', 'Total Sales', 'Quantity', 'Total_GP']
Products['FormattedTotalGP'] = Products['Total_GP'].apply(lambda x: "${:,.2f}".format((x)))
Products['FormattedQtySold'] = Products['Quantity'].apply(lambda x: "{:,.0f}".format((x)))
Products[['Product', 'Product Number']] = Products['Product Name'].str.split('_', expand = True)
#%% Plot the Top 20 most profitable products
#%% Create data for plotting
labels = Products['Product Number'][:20]
pos = np.arange(len(labels))
values = Products['Total_GP'][:20]
formattedvalues = Products['FormattedTotalGP'][:20]
formattedqty = Products['FormattedQtySold'][:20]
#%% and plot
width = 0.9
fig, ax = plt.subplots(figsize=(30, 10))
ax.bar(labels, values, width, color='paleturquoise', label= 'Gross Profit', align = 'center')
plt.xticks(pos, labels, size = 12, rotation = 45)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
ax.annotate(formattedvalues[0], (labels[0],values[0]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[1], (labels[1],values[1]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[2], (labels[2],values[2]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[3], (labels[3],values[3]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[4], (labels[4],values[4]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black',weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[5], (labels[5],values[5]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black',weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[6], (labels[6],values[6]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black',weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[7], (labels[7],values[7]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[8], (labels[8],values[8]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[9], (labels[9],values[9]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[10], (labels[10],values[10]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[11], (labels[11],values[11]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black',weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[12], (labels[12],values[12]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black',weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[13], (labels[13],values[13]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black',weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[14], (labels[14],values[14]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[15], (labels[15],values[15]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[16], (labels[16],values[16]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black', weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[17], (labels[17],values[17]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black',weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[18], (labels[18],values[18]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black',weight = 'regular', rotation = 90)
ax.annotate(formattedvalues[19], (labels[19],values[19]), textcoords = 'offset points', xytext = (0,-100), ha = 'center', size = 14, color = 'black',weight = 'regular', rotation = 90)
# # Add title and labels
plt.xlabel('Product Number',color = '#000000', fontsize = 14, labelpad = 10)
plt.ylabel('Gross Profit', color = '#000000',fontsize = 16)
ax.set_title("Top 20 Products: Gross Profit & Quantity Sold", color = '#000000', fontsize = 18, pad = 20)
#ax.legend(loc="upper center", fontsize = 14)
# Add the qty sold
plt.annotate(formattedqty[0], (pos[0],values[0]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[1], (pos[1],values[1]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[2], (pos[2],values[2]), textcoords = 'offset points',  xytext=(5,30), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1), rotation = 0)
plt.annotate(formattedqty[3], (pos[3],values[3]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[4], (pos[4],values[4]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[5], (pos[5],values[5]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[6], (pos[6],values[6]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[7], (pos[7],values[7]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[8], (pos[8],values[8]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[9], (pos[9],values[9]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[10], (pos[10],values[10]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[11], (pos[11],values[11]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[12], (pos[12],values[12]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[13], (pos[13],values[13]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[14], (pos[14],values[14]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[15], (pos[15],values[15]), textcoords = 'offset points',  xytext=(5,30), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1), rotation = 0)
plt.annotate(formattedqty[16], (pos[16],values[16]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[17], (pos[17],values[17]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[18], (pos[18],values[18]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedqty[19], (pos[19],values[19]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =12, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
# Display the plot
plt.show()
plt.savefig('Top20Products.png')
#%% Coursera Customer Segmentation Project
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#matplotlib.style.available
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
#%% Sales Data Customer Segmentation
df = TotalGrossProfit
df.head()
df.columns.tolist()
df = df.rename(columns={'mean':'Mean Order Value', 'size': 'Purchase Volume'})
features = ['Total Gross Profit', 'Purchase Volume', 'Mean Order Value', 'CLV']


fig, axes = plt.subplots(2,2, figsize = (15,10))

for feature, ax in zip(features, axes.ravel()):
    ax.hist(df[feature], bins = 50, color = 'lightslategrey')
    ax.set_title(feature)
plt.tight_layout()
#%% Plot pair plots
pplot = sns.pairplot(df, vars = features)
pplot.fig.set_size_inches(10,10)
pplot.set_ylabels(size = 12)


