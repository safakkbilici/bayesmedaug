import math

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def discrete_angle(x):
    if x in range(0,30):
        x = 30
    elif x in range(30,60):
        x = 60
    elif x in range(60,90):
        x = 90
    elif x in range(90,120):
        x = 120
    elif x in range(120,150):
        x = 150
    elif x in range(150,180):
        x = 180

    return x

def discrete_angle_normalized(x):
    return int(math.ceil(x*100 / 10.0)) * 10

def discrete_shift(x):
    if x >=0:
        x = 100
    else:
        x = -100
    return x
