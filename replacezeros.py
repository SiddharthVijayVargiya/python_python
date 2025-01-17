x= [0,1,0,2,3,4,12]
non_zero = 0
for i in range(len(x)):
    if x[i] != 0:
        x[non_zero] = x[i]
        non_zero+=1
for i in range(non_zero,len(x)):
    x[i] =0
print(x)
        