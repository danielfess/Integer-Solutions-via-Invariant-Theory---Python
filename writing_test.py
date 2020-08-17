import os
from os import path

filename = 'test_write.txt'
indices_done = []

if path.exists(filename):
	with open(filename, 'r') as f:
		if os.stat(filename).st_size != 0:
			line = f.readline()
			while line:
				if line.startswith('('):
					indices_done.append(line[:9])
				line = f.readline()

print(indices_done)
print(len(indices_done))

with open(filename, 'a') as f:
	if os.stat(filename).st_size == 0:
		f.write('#(i,j,k), D_{i,j}^k\n')
	for i in range(5):
		for j in range(5):
			for k in range(5):
			    if str((i+1, j+1, k+1)) not in indices_done:
				    f.write('{}\n'.format((i+1,j+1,k+1)))