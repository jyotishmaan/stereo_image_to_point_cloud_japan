from __future__ import division 
import pandas as pd

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        print('\n')
        print(x,y)
        return x,y
    else:
        print('\n')
        print('Same')
        return False

df = pd.read_csv (r'book.csv')
#print(df)
LinesRight = [] 
for index, row in df.iterrows():
     # access data using column names
     if index == 700:
         break
     x1 = row['R1X']
     y1 = row['R1Y']

     x2 = row['R2X']

     y2 = row['R2Y']

     L1 = line([x1,y1],[x2,y2])
     LinesRight.append(L1)


df = pd.DataFrame(columns =['x', 'y']) 
df.to_csv(r'port_estimation.csv')

for x in range(10):
    first = LinesRight[x]
    print('@@@@@@')
    for l in LinesRight[x+1::10]:
        print('###############')
        second = l
        x,y = intersection(first,second)
        coordinates = []
        coordinates = coordinates + [x]
        coordinates = coordinates + [y]

        dataframe = pd.DataFrame([coordinates], columns =['x', 'y']) 
        print(dataframe)
        dataframe.to_csv(r'port_estimation.csv',mode='a', header=False)

        first = second

#a = line([44,29],[58,15])
#b = line([36,20],[60,10])
#print(a)
#print(b)
#print(intersection(a,b))
    
    


