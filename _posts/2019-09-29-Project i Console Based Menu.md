---
layout:     post
title:      Origin-Project I Console Based Menu
subtitle:   The use of csv, collections in python
date:       2019-09-29
author:     Joe Zhang
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Python
---

Provide the system user with a console based menu as follows:

* Load the data set from exams.csv file (file is comma delimited).

* Print a list of student names, final scores, and letter grades sorted by names.

* Print score summary (Student Count, Min, Max, mean, mode,and standard deviation).

* Identify values that are larger than the mean and two times standard deviation.

* Plot a pie chartshowing the final letter grades distribution.

* Create box plotsparameters (not drawing them but just computing the numbers[min, max, medial, Q1, Q3]for a box plot). 

* Exit the system.
  
>  解决方案：进入项目文件夹下的 .git文件中（显示隐藏文件夹或rm .git/index.lock）删除index.lock文件即可。

```python
import matplotlib.pyplot as plt
import os
import csv
from scipy import stats
import numpy as np
from collections import Counter

def openfile(filename = 'exams.csv'):
    # open the file and get data
    with open(os.path.join('./',filename),'r',encoding="utf-8") as f:
        reader = csv.reader(f)
        fieldnames = next(reader)   # get the first row as the header, which also compose the fieldnames of the studic below
        csv_reader = csv.DictReader(f,fieldnames=fieldnames)   # self._fieldnames = fieldnames   # list of keys for the dict
        stulist = []
        for row in csv_reader:
            studic={}
            for k,v in row.items():
                studic[k] = v
            studic['FinalScore'] = round(float(studic['Exams']) * 0.4\
                  + float(studic['Quizzes']) * 0.3\
                  + float(studic['Projects']) * 0.3, 1)
            if (studic['FinalScore'] >= 90): 
                studic['FinalGrade'] = 'A'
            elif (studic['FinalScore'] >= 80):
                studic['FinalGrade'] = 'B'
            elif (studic['FinalScore'] >= 20):
                studic['FinalGrade'] = 'C'
            elif (studic['FinalScore'] >= 10):
                studic['FinalGrade'] = 'D'
            else:
                studic['FinalGrade'] = 'F'
            stulist.append(studic)
        print('Student Data has been loaded!\n')
    return stulist
#stulist = openfile()
#print (stulist)

def getlist(data, ls = None):
    # get the list of data by passing a list of field names
    ls = ls if ls is not None else ['FinalScore']
    colname = []
    for i in range(0, len(ls)):
        colname.append([])
    for stu in data:
        for i, col in enumerate(ls):
            if (col in ['Quizzes', 'Projects', 'Exams']):
                colname[i].append(round(float(stu[col]), 1))
            else:
                colname[i].append(stu[col])
    return colname
#getlist(stulist, ['id', 'Name'])

def getmode(ls):
    # get the mode of a given list
    count = Counter(list(ls)).most_common()
    i = 1
    result = []
    while (count[i-1][1] == count[i][1] and i<= len(count)):
        i += 1
    for i in Counter(ls).most_common(i):
        result.append(i[0])
    return result

def getprintlist(data):
    # Print a list of student names, final scores, and letter grades sorted by names
    data = getlist(data, ['Name', 'FinalScore', 'FinalGrade'])
    result = [*zip(*data)]
    result.sort()
    print('Below is the students\' grades: ')
    for row in result:
        print(list(row))
    print()
#getprintlist(stulist)

def scoresummary(data, ls = None):
    # Print score summary (Student Count, Min, Max, mean, mode,and standard deviation)
    ls = ls if ls is not None else ['FinalScore']
    data = getlist(data, ls)
    for i, col in enumerate(data):
        print('Summary of {0}:'.format(ls[i]))
        print(' - Count: {}'.format(len(col)))
        print(' - Min:   {0:.1f}'.format(min(col)))
        print(' - Max:   {0:.1f}'.format(max(col)))
        print(' - Mean:  {0:.1f}'.format(np.mean(col)))
        #print(' - Mode:  {0:.1f}'.format(stats.mode(col)[0][0]))
        print(' - Mode:  {0}'.format(*getmode(col)))
        print(' - SD:    {0:.1f}\n'.format(np.std(col, ddof=1)))

#scoresummary(stulist, ['Quizzes', 'Projects'])  

def largevalue2(data, ls = None):
    # Identify values that are larger than the mean and two times standard deviation
    ls = ls if ls is not None else ['FinalScore']
    data = getlist(data, ls)
    for i, col in enumerate(data):
        print('Large value for {0}:'.format(ls[i]))
        print([x for x in col if (x > np.mean(col) + 2 * np.std(col, ddof=1))], '\n')
#largevalue(stulist, ['Quizzes', 'Projects'])

def largevalue(data, ls = None):
    # Identify values that are larger than the mean and two times standard deviation
    ls = ls if ls is not None else ['FinalScore']
    ls = ['id','Name'] + ls
    data = getlist(data, ls)
    for i, col in enumerate(data[2::]):
        print('Large value for {0}:'.format(ls[i+2]))
        for i, x in enumerate(col):
            if (x > np.mean(col) + 2 * np.std(col, ddof=1)):
                print (data[0][i], data[1][i], x)
    print()            
#largevalue(stulist, ['Quizzes', 'Projects', 'FinalScore','Exams'])

def piechart(data):
    # Plot a pie chart showing the final letter grades distribution
    data = getlist(data, ['FinalGrade'])[0]
    cnt = []
    labels = ['A', 'B', 'C', 'D', 'F']  
    cnt.append(data.count('A'))
    cnt.append(data.count('B'))
    cnt.append(data.count('C'))
    cnt.append(data.count('D'))
    cnt.append(data.count('F'))
    plt.pie(cnt, labels = labels)
    plt.show()
#piechart(stulist)   

def calboxpara(data, ls = None):
    # Create box plots parameters (not drawing them but just computing the numbers[min, max, medial, Q1, Q3] for a box plot)
    ls = ls if ls is not None else ['FinalScore']
    data = getlist(data, ls)
    for i, col in enumerate(data): 
        maxv = max(col)
        minv = min(col)    
        Q1 = np.percentile(col, 25)
        Q3 = np.percentile(col, 75)
        medianv = np.median(col)
        print('Boxplot parameters for {0}:'.format(ls[i]))
        print(' - Min:    {0:.1f}'.format(minv))
        print(' - Q1:     {0:.1f}'.format(Q1))
        print(' - Median: {0:.1f}'.format(medianv))
        print(' - Q3:     {0:.1f}'.format(Q3))
        print(' - Max:    {0:.1f}\n'.format(maxv))
#calboxpara(stulist, ['Quizzes', 'Projects'])

if __name__ == '__main__':
    flag = True
    stulist = None
    while flag:
        print('Please choose a function:')
        print(' 1: Load the data')
        print(' 2: Print student info')
        print(' 3: Print score summary')
        print(' 4: Identify large values')
        print(' 5: Pie chart of final grades')
        print(' 6: Box plots parameters')
        print(' 0: Exit the system\n')
        choice = input()
        if choice == '0': flag = False
        elif choice == '1':
            stulist = openfile()
        elif choice == '2':
            if stulist == None: print('Data Must be loaded first!\n')
            else: 
                getprintlist(stulist)
        elif choice == '3':
            if stulist == None: print('Data Must be loaded first!\n')
            else: 
                print('Please enter the fieldname you want to see.')
                print('Filednames are seperated with a space.')
                print('Eample: Quizzes Projects')
                print('Available field names are: Quizzes, Projects, Exams, FinalScore')
                flag1 = True
                while flag1:
                    fieldname = [*map(lambda x: x.strip(), input().split())]
                    try:
                        print()
                        scoresummary(stulist, fieldname)
                        flag1 = False
                    except:
                        print('Entered fields not found. Please try again.')
        elif choice == '4':
            if stulist == None: print('Data Must be loaded first!\n')
            else: 
                print('Please enter the fieldname you want to see.')
                print('Filednames are seperated with a space.')
                print('Eample: Quizzes Projects')
                print('Available field names are: Quizzes, Projects, Exams, FinalScore')
                flag1 = True
                while flag1:
                    fieldname = [*map(lambda x: x.strip(), input().split())]
                    try:
                        print()
                        largevalue(stulist, fieldname)
                        flag1 = False
                    except:
                        print('Entered fields not found. Please try again.')
        elif choice == '5':
            if stulist == None: print('Data Must be loaded first!\n')
            else: 
                piechart(stulist) 
        elif choice == '6':
            if stulist == None: print('Data Must be loaded first!\n')
            else: 
                print('Please enter the fieldname you want to see.')
                print('Filednames are seperated with a space.')
                print('Eample: Quizzes Projects')
                print('Available field names are: Quizzes, Projects, Exams, FinalScore')
                flag1 = True
                while flag1:
                    fieldname = [*map(lambda x: x.strip(), input().split())]
                    try:
                        print()
                        calboxpara(stulist, fieldname)
                        flag1 = False
                    except:
                        print('Entered fields not found. Please try again.')

```