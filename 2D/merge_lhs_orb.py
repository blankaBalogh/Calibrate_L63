import numpy as np

# Loading datasets
fdir = 'dataset/'
extra_tag = ''

xt_lhs = np.load(fdir+'x_data-a1'+extra_tag+'.npz')['arr_0']
xt_orb = np.load(fdir+'x_data-a2'+extra_tag+'.npz')['arr_0']

yt_lhs = np.load(fdir+'y_data-a1'+extra_tag+'.npz')['arr_0']
yt_orb = np.load(fdir+'y_data-a2'+extra_tag+'.npz')['arr_0']


# Reshaping datasets
xt_lhs, yt_lhs = xt_lhs[0], yt_lhs[0]
lhs_data = np.concatenate((xt_lhs, yt_lhs), axis=1)

xt_orb, yt_orb = xt_orb.reshape(-1,6), yt_orb.reshape(-1,6)
orb_data = np.concatenate((xt_orb, yt_orb), axis=1)

learning_data = np.concatenate((lhs_data, orb_data), axis=0)


# Shuffling rows
np.random.seed(42)
np.random.shuffle(learning_data)

# Saving learning sample 
xt_mix, yt_mix = learning_data[:,:6], learning_data[:,6:]

np.savez_compressed(fdir+'x_data-amix-2.npz', xt_mix)
np.savez_compressed(fdir+'y_data-amix-2.npz', yt_mix)

print(' > Datasets successfully saved : ')
print('   '+fdir+'x_data-amix-2.npz')
print('   '+fdir+'y_data-amix-2.npz.')

exit()
