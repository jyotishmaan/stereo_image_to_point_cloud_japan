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
        print('Same')
        return False

df = pd.read_csv (r'testing.csv')
#print(df)
LinesRight = [] 
for index, row in df.iterrows():
     # access data using column names
     x1 = row['R1x']
     y1 = row['R1y']

     x2 = row['R2x']

     y2 = row['R2y']

     L1 = line([x1,y1],[x2,y2])
     LinesRight.append(L1)

print(LinesRight)
first = LinesRight[0]
for l in LinesRight[1:]:
    second = l
    point = intersection(first,second)
    first = second


    
    


