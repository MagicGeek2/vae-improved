def func():
    # return 1,2,3
    re=(1,2)
    print(type(re))
    re=re+tuple([3])
    return re

a,b,c = func()
print(a,b,c)