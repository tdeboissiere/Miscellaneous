'''
Plot a 2D matrix array
'''

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt

H = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])  # added some commas and array creation code

fig, ax = plt.subplots()

image = H #np.random.uniform(size=(10, 10))
ax.imshow(image,   interpolation='nearest') #cmap=plt.cm.gray,

ax.set_title('dropped spines')

plt.imshow(H,   interpolation='nearest')
plt.colorbar(orientation='vertical')
plt.show()


'''
Subplots
'''

fig, (ax0, ax1)  = plt.subplots(nrows=2)
xmax=Pulse.TraceLength()

ax0.plot(np.arange(xmax),PulseA.Trace(),color="black")
ax0.set_title('Channel A spike')

ax1.plot(np.arange(xmax),PulseB.Trace(),color="red")
ax1.set_title('Channel B spike')

plt.subplots_adjust(hspace=0.3)

'''
Clearing figures
'''
# cla() clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
# clf() clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
# close() closes a window, which will be the current window, if not specified otherwise.
# Which functions suits you best depends thus on your use-case.
# The close() function furthermore allows one to specify which window should be closed. The argument can either be a number or name given to a window when it was created using figure(number_or_name) or it can be a figure instance fig obtained, i.e., usingfig = figure(). If now argument is given to close(), the currently active window will be closed. Furthermore, there is the syntax close('all'), which closes all figures.

'''
Marker size in scatter plot
'''
# => specify s=...
scatter(x, y, s=500, color='green', marker='h')

'''
Move the legend
'''
ax1_1.legend(bbox_to_anchor=[0.33, 0.9])

'''
Properly normalise to 1
'''
weights = np.ones_like(myarray)/len(myarray)
plt.hist(myarray, weights=weights)

'''
Color scale
'''

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c('red'), c('red'), 0.33, c('red'), c('green'), 0.76, c('green')])


'''
Nice plotting options
'''     

plt.figure()
plt.scatter(arr_heat_WIMP, arr_ion_fid_WIMP, color="0.6", s=1 )
plt.scatter(arr_heat, arr_ion_fid, s=20, c=arr['NN'], cmap=rvb, vmin = -1, vmax = 1)
cbar = plt.colorbar()
cbar.set_label('BDT output', labelpad = 15, fontsize= 18)
plt.xlabel('Heat (keV)', fontsize = 20)
plt.ylabel('Fiducial Ion (keV)', fontsize = 20)
plt.ylim([0,5])
plt.xlim([1,5])
plt.title(list_title[index] + "   " r'$M_{\chi}=$' + " " + str(mass) + " GeV", y=1.031, fontsize = 24)
plt.grid(True)
plt.savefig(fig_path + list_fig_name[index] + ".png")
plt.close("all")


'''
labels, legend title, nice layout
'''  

plt.xlabel("Heat (keVee)")
plt.legend(loc="lower right", title = "KS p-value", prop={"size":10})
plt.tight_layout()


'''
Use latex
'''  

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

'''
Group boxplot
''' 

bp1 = plt.boxplot(d_train[8],0, "", positions = [1], widths = 0.6, patch_artist=True)
bp2 = plt.boxplot(d_test[8],0, "", positions = [2], widths = 0.6, patch_artist=True)
# # second boxplot pair
plt.boxplot([d_train[7], d_test[7]],0, "", positions = [4, 5], widths = 0.6)

'''
Nice scatter plot with seaborn
''' 

sns.set_style("white")
sns.kdeplot(arr_WIMP, cmap = sns.dark_palette(sns.xkcd_rgb["ocean"], as_cmap = True))
plt.plot([0.5,3], [0,0], "k--")
sns.regplot(arr_heat[:,0], arr_heat[:,1], color =sns.xkcd_rgb["orange red"], fit_reg=False, scatter_kws={"s": 5, "alpha":0.5})
# plt.scatter(arr_heat[:,0], arr_heat[:,1], color = sns.xkcd_rgb["pale red"], s=2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.grid()
plt.xlabel("Combined Heat (keVee)", fontsize = 20)
plt.ylabel("Fiducial ionisation (keVee)", fontsize = 20)
plt.xlim([0.5, 3])
plt.ylim([-0.5, 2])
plt.tick_params(axis='both', which='major', labelsize=15) 
plt.tight_layout()
plt.savefig("FID837_heat_sideband.png")