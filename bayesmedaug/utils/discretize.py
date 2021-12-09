import math

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def discrete_angle(x):
    return int(x * 1000)

def discrete_angle_normalized(x):
    return int(math.ceil(x*100 / 10.0)) * 10

def discrete_shift(x):
    if x >=0:
        x = 100
    else:
        x = -100
    return x

def discrete_rcrop(x):
    return int(x * 1000)
