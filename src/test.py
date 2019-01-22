p = ['accendif', 'accend', 'ag', 'acces', 'agh', 'accendifu', 'accessor', 'aghi', 'accendifuo', 'accendi', 'access']
s_p = sorted(p)
l = ['']

for item1 in p:
    for item2 in p:
        if item1 in item2 and len(item2) > len(l[0]):
            l[0] = item2
        elif item2 in item1 and len(item1) > len(l[0]):
            l[0] = item1

print(l)

list = []
for i, val in enumerate(s_p):
    if i < len(s_p)-1:
        if not s_p[i] in s_p[i+1]:
            list.append(s_p[i])
    else:
        list.append(s_p[i])

print(list)