'''
Automatic code formatting
'''
autopep8 --in-place --aggressive filename.py


'''
Read a file and access its lines
'''

f=open('dataset_2_4.txt','r')
stuff=f.readlines();

text=stuff[0];
k=int(stuff[1]);
l=len(text);

'''
Dictionnary manipulation
'''
d={}
for i in range(l-k+1):
   kmer = text[i:i+k]
   if kmer in d:
      d[kmer]+=1
   else:
      d[kmer]=1
'''
Returns an array of pair: key word and associated values
'''
print(d.items())

'''
Advanced loop: will look for kmer of the (kmer,freq) doublet in d.items() with an if condition
'''
max_frequency = max(d.values())
answer =[kmer for kmer, freq in d.items() if freq==max_frequency]


'''
Some print string concatenation option
'''
print(' '.join(answer))


'''
Reverse a string
'''
'hello world'[::-1]

'''
Range
'''
for i in range(1,4):
.....

gives 1, 2 ,3 not 4



'''
Array and string manipulation
'''

if list: del list[-1] removes the last element
if str: str =str[:-i] removes the last i elements  [:-1] allows to remove the \n character
can also use  '   spacious   '.rstrip() which returns '   spacious'

str = ''.join(str.split()) removes all and any whitespace in str


''' 
Read word by word
'''
truc=patt.rsplit(' ');

if patt = '12  13   14' returns an array ['12', '13', '14']



''' 
Iterer sur les elements qui ne sont pas dans la liste
'''

for i in mah:
  if i not in newmah:
    newmah.append(i)

''' 
Replace string by a list of the string elements if need to change the value of one such element
in python, strings are immutable
'''

'''
Named tuples
'''
from collections import namedtuple
Coo=namedtuple('Coordonnées','x y')
create namedtuple: Pt=Coo(1,2)
then can call Pt.x ->returns 1   Pt.y -> returns 2

'''
remove white space
'''
text=text.split()
if text= 0   113   123, output= ['0','113','123']


'''
Sort according to 2nd element of tuple
'''
from operator import itemgetter
data = [('abc', 121),('abc', 231),('abc', 148), ('abc',221)]
sorted(data,key=itemgetter(1))
[('abc', 121), ('abc', 148), ('abc', 221), ('abc', 231)]

'''
Copying a list properly
'''
if I do: 
a=[0,12] and b=a
then a.append(1) will give: a=[0,12,1] and b =[0,12,1]
so write b=list(a)
then b isnt modified

'''
Execute a script in interactive mode
'''
execfile('test.py')

'''
Import functions defined in other scripts
'''
from tes2 import *


'''
Automatize ssh/scp
=> Use paramiko
'''

import os
import paramiko

ssh = paramiko.SSHClient() 
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()
sftp.put(localpath, remotepath) # to upload the file   /!\ both  path must end with 
stfp.get(remotepath,localpath) #to download the file
sftp.close()
ssh.close()

#Lines below: read all files which end with .py and download them in target dir
stdin, stdout, stderr =ssh.exec_command("cd /sps/edelweis/kdata/code/dev/KDataPy/ && ls *.py ")
data = stdout.read().splitlines()
sftp = ssh.open_sftp()
for elem in data:
   sftp.get('/sps/edelweis/kdata/code/dev/KDataPy/'+elem,'/home/irfulx204/mnt/tmain/Desktop/KData/Upload_util/'+elem)  #put( repertoire origine, repertoire final)
sftp.close()
ssh.close()

'''
List of files in python
'''
from os import listdir
from os.path import isfile, join
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

# other option

import glob
print glob.glob("/home/adam/*.txt")  #with this can  use wildcard


'''
Write a list to a file python
'''

f=open('./listname.txt', 'w')
for item in listf:
  f.write("%s\n" % item)
  
  
  
'''
Function arguments
'''
def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
#voltage is positional, state is key word argument
#keyword arguments MUST FOLLOW positional arguments

def cheeseshop(kind,truc='a', *args, **kwargs):
 can be called as: cheeseshop("Limburger",'a' "It's very runny, sir.",
           			"It's really very, VERY runny, sir.",
           			shopkeeper='Michael Palin',
           			client="John Cleese",
           			sketch="Cheese Shop Sketch")

# then, args will contain "It's very runny, sir.", and  "It's really very, VERY runny, sir."
# kwargs is a dictionnary which constains shopkeeper='Michael Palin', client="John Cleese", sketch="Cheese Shop Sketch")

'''
Prompt pwd w/o echoing
'''
import getpass
pwd = getpass.getpass('Password please:   ')


'''
Date and time
'''
Look for datetime in python doc

'''
Element wise concatenation
'''

a1 = ['a','b']
a2 = ['E','F']
map(''.join, zip(a1, a2))
# output : ['aE', 'bF']

'''
Number of occurences:
'''

>>> from collections import Counter
>>> z = ['blue', 'red', 'blue', 'yellow', 'blue', 'red']
>>> Counter(z)
Counter({'blue': 3, 'red': 2, 'yellow': 1})

'''
First element that matches a given one
'''
next(i for i in xrange(100000) if i == 1000)

'''
convert comma separated string into reducing string list
'''

location_in  = 'London, Greater London, England, United Kingdom'
locations    = location_in.split(', ')
location_out = [', '.join(locations[n:]) for n in range(len(locations))]

a='3,10'
a.split(',') => ['3','10']


'''
Get the time
'''

>>> from time import gmtime, strftime
>>> strftime("%Y-%m-%d %H:%M:%S", gmtime())
'2009-01-05 22:14:39'

# (remove gmtime to get local time)


'''
Convert list of string to list of int
'''
results=['1','2']
results = map(int, results)

'''
recursivity
'''
# If need high recursion depth:
import sys 
sys.setrecursionlimit(1500)
# (default is 999)

'''
Global variables
'''

define var outside func:
var=''
def func():
   do stuff
   global var
   do some operations on var
   end
   
'''
List of list to list
'''
[item for sublist in l for item in sublist] 

'''
Index of element in list
'''
 list_lett=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
 return list_lett.index(letter) 
 
'''
Remove all occurrences
'''

 [value for value in the_list if value != val]
 
'''
Python performance
''' 
# check out this:
# http://www.huyng.com/posts/python-performance-analysis/
# (use line by line profiler to see what takes time)

'''
Flatten a list
'''
[item for sublist in l for item in sublist]

'''
Float range
'''

Use numpy 
>>> from numpy import arange
>>> arange(0.5, 5, 1.5)
array([0.5, 2.0, 3.5])


'''
Create a directory
'''
    
if not os.path.exists('./Plots/'+bolo_name): os.makedirs('./Plots/'+bolo_name)

'''
Convert a string to a dictionnary
'''

import ast
ast.literal_eval("{'muffin' : 'lolz', 'foo' : 'kitty'}")
#output {'muffin': 'lolz', 'foo': 'kitty'}


'''
Exit script
'''
sys.exit()
# (kills everything as errors propagate)

'''
Overwrite print line
'''
for x in range(100000):
    print '{0}\r'.format(str(x)+'er'+str(x)),
#print (uncomment if want to print the last entry)


'''
Argparse a list
'''
def truc(mach):
   
    for elem in mach:
        print elem+elem

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("bolo_name", help="Bolometer name", nargs='+', type=str)
    args = parser.parse_args()

    truc(args.bolo_name)

 
 
'''
Enumerate
'''
 
a = ['a', 'b', 'c', 'd', 'e']
for index, item in enumerate(a): print index, item
# output
  # 0 a
  # 1 b
  # 2 c
  # 3 d
  # 4 e

 '''
 Column wise adding files
 '''
from itertools import izip
import csv
with open('A','rb') as f1, open('B','rb') as f2, open('out.csv','wb') as w:
    writer = csv.writer(w)
    for r1,r2 in izip(csv.reader(f1),csv.reader(f2)):
        writer.writerow(r1+r2)


'''
Ordered dictionnary
'''
from collections import OrderedDict
OrderedDict((word, True) for word in words)

'''
Unix to real time
'''
import datetime
print(
    datetime.datetime.fromtimestamp(
        int("1284101485")
    ).strftime('%Y-%m-%d %H:%M:%S')
)

'''
String formatting  .3 => 3 significative figures    g => best presentation
'''
"{:.3g}".format(scipy.stats.ks_2samp(arr_S1Pb, arr_S1Pb_ref)[1]),


'''
Progress bar
'''
sys.stdout.write("\r" + str(i)+" / "+str(-1+nsimu))
sys.stdout.flush()

'''
save dict to pickle
'''

import pickle
a = {'hello': 'world'}

with open('filename.pickle', 'wb') as handle:
  pickle.dump(a, handle)

with open('filename.pickle', 'rb') as handle:
  b = pickle.load(handle)

"""
Get the output of a subprocess call
"""
for f in arr_extracted_files:
    subprocess.call(shlex.split("wc -l %s" % f))
    proc = subprocess.Popen(shlex.split("wc -l %s" % f), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    ll.append(out)
plt.hist(ll, bins=100)
plt.show()

"""
Break out of nested loops
"""
class Found(Exception): pass
try:
    for i in range(100):
        for j in range(1000):
            for k in range(10000):
               if i + j + k == 777:
                  raise Found
except Found:
    print i, j, k 

      
"""
subprocess
"""

subprocess.Popen() may be faster for some operations (like copy) than subprocess.call()


"""
upper case
"""

string.title() to upper case the first letter of each word in string

"""
Sort files by creation date
"""

list_HDF5.sort(key=lambda x: os.path.getmtime(x))

"""
Get file size in bytes
"""
import os
os.stat(file).st_size
