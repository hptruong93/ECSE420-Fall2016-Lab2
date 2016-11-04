import subprocess

HOSTS = [

'lab1-10',
'lab1-11',
'lab1-12',
'lab1-14',
'lab1-15',
'lab1-16',

'lab2-21',
'lab2-22',
'lab2-24',
'lab2-25',
'lab2-26',
'lab2-27',

]

PATH = '/home/2013/ptruon4/ECSE420-Fall2016-Lab2'
ITERATIONS = 100

if __name__ == "__main__":
	cores = [1,2,4,8,16,32]

	for i, core in enumerate(cores):
		# cmd = "ssh {0}.cs.mcgill.ca 'cd {1} ; mpiexec -np {2} python grid_512_512.py {3} -t' &> run_{4}.txt".format(HOSTS[i], PATH, core, ITERATIONS, '{}_{}'.format(core, ITERATIONS))
		cmd = 'mpiexec -np {1} python grid_512_512.py {2} -t &> run_{3}.txt ; '.format(PATH, core, ITERATIONS, '{}_{}'.format(core, ITERATIONS))
		# subprocess.check_call(cmd, shell = True)

		print cmd