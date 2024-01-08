total=0
up=2
dowm=1
for i in range(20):
    total+=up/dowm
    up, dowm=up+dowm, up
print(total)