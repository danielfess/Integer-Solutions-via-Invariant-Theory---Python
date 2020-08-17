a = [1,2,3,4]
a = [1,2,3]
print(a)

b = a[1]
a.remove(2)
print(b)
print(a[1])

count = 0
c = [count]
print(c)
count += 1
print(c)
print(count)

d = tuple(c)
print(len(d))
e = [d]
print(e)

print(tuple())

test_dict = dict()
test_dict['apple'] = 'fruit'
test_dict['banana'] = 'fruit'
test_dict['carrot'] = 'vegetable'
print(test_dict)
print(test_dict.items())
print(type(test_dict.items()))
for entry in test_dict.items():
	print(entry)