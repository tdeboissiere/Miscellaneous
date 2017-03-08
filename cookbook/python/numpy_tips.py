'''
maximum of an array 
'''

a.argmax(axis=0)
#Will give index of max over the axis 0
#if axis=None (default) => maximum over flattened array

'''
Linearly space points
'''       
np.linspace(borne_inf, borne_sup, Npoints)


'''
Universal functions
'''

# They can be used on arrays or int, float, ...
# Go to http://docs.scipy.org/doc/numpy/reference/ufuncs.html
# for a list.
# It is faster to use these when operating on arrays. (Implemented vecotrisation)
# e.g. :   

def funcnew(x,par):
    return np.exp(x)+par

l=np.linspace(0,1,10)
g=funcnew(l,1)


'''
Convert array to array of float
'''
np.array([(3, 0, 1)]).astype('float')


'''
Import array from .csv file 
'''
y=np.genfromtxt('heatA_heatonlypulses_4V_FID822.txt', delimiter=',')


'''
Size of an array
'''

a.shape

'''
2D integration with arguments
'''
from scipy import integrate
def f(y,x, c, d):
    return x*y +c + d
    
def lowfun(x):
    return 0

def upfun(x):
    return x
    
c= 1
d=0

print integrate.dblquad(f, 0, 1, lambda x : 0, lambda x : x , args=(c,d))
# here it does not work if I used only one argument => use 2 arguments and set d to zero...
# WARNING: f(y,x) => will do the integration first on y, between 0 and x, then on x, between 0 and 1


'''
Vectorization of function
'''
import numpy as np
from scipy import special

def computerate_sfg(Er, sigma_nuc, Mchi, A) :
    return rate

vrate=np.vectorize(computerate_sfg, excluded=['sigma_nuc', 'Mchi', 'A'])
print vrate([10,11,12], sigma_nuc=1E-5, Mchi=100, A=72)
# => this will return computerate_sfg evaluated at Er=10, 11, 12, with sigma_nuc, Mchi, A fixed

'''
Convert structured array to normal array
'''
import numpy as np

data = [ (1, 2), (3, 4.1), (13, 77) ]
dtype = [('x', float), ('y', float)]

print('\n ndarray')
nd = np.array(data)
print nd

print ('\n structured array')

struct_1dtype = np.array(data, dtype=dtype)
print struct_1dtype

print('\n structured to ndarray')
struct_1dtype_float = struct_1dtype.view(np.ndarray).reshape(len(struct_1dtype), -1)
print struct_1dtype_float

print('\n structured to float: alternative ways')
struct_1dtype_float_alt = struct_1dtype.view((np.float, len(struct_1dtype.dtype.names)))
print struct_1dtype_float_alt

# with heterogeneous dtype.
struct_diffdtype = np.array([(1.0, 'string1', 2.0), (3.0, 'string2', 4.1)],
dtype=[('x', float),('str_var', 'a7'),('y',float)])
print('\n structured array with different dtypes')
print struct_diffdtype
struct_diffdtype_nd = struct_diffdtype[['str_var', 'x', 'y']].view(np.ndarray).reshape(len(struct_diffdtype), -1)


print('\n structured array with different dtypes to reshaped ndarray')
print struct_diffdtype_nd


print('\n structured array with different dtypes to reshaped float array ommiting string columns')
struct_diffdtype_float = struct_diffdtype[['x', 'y']].view(float).reshape(len(struct_diffdtype),-1)
print struct_diffdtype_float

'''
Text file formatting to array
'''
data_types = {"names": ("RUN", "SN", "EC1_ERA", "EC2_ERA"), "formats": ("i", "i", "f", "f")}
arr_ERA = np.loadtxt("/home/irfulx204/mnt/tmain/Desktop/Paco_ldb/FID837_heatonly_ERA.txt", delimiter=",",  dtype=data_types)


'''
Masks to filter array
'''
arr_S1Pb_heat = coeff_EC1*arr_S1Pb[:,0] + coeff_EC2*arr_S1Pb[:,1]
arr_S2Pb_heat = coeff_EC1*arr_S2Pb[:,0] + coeff_EC2*arr_S2Pb[:,1]

mask_S1Pb     = np.where((1<arr_S1Pb_heat) & (arr_S1Pb_heat<15))
arr_S1Pb      = arr_S1Pb[mask_S1Pb]        

mask_S2Pb     = np.where((1<arr_S2Pb_heat) & (arr_S2Pb_heat<15))
arr_S2Pb      = arr_S2Pb[mask_S2Pb]


'''
Reshape arrays
'''
#Here Xerr is a 6,6 matrix
#It is reshaped to a 1, 6, 6 matrix
Xerr = np.reshape(Xerr, (1,6,6))
#Then tiling is used so that it is a 3,6,6 matrix
#where each of the 3 rows contains the previous 6,6 matrix
Xerr = np.tile(Xerr, (3,1,1))

'''
Use a header
'''
#comments ="" => no # at the beginning of header
out_dir = script_utils.create_directory("./Eval_data/" + bolo_name + "/" + analysis_type + "/")
np.savetxt(out_dir + bolo_name + "_" + analysis_type + "_fond.csv", arr, delimiter = ",", fmt = "%.5f", header = "EC1,EC2,EIA,EIB,EIC,EID,ENR,RUN,SN", comments = "")   

'''
array slicing
'''

arr[0:10]  # fast
arr[[0, 1, 2, 3, 4, ...]]  # slow
