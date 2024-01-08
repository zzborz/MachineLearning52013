import datetime
year, month, day = map(int, input().split(" "))
YuanDan=datetime.datetime(year,1,1)
Now=datetime.datetime(year,month,day)
print((Now-YuanDan).days+1)
#完成