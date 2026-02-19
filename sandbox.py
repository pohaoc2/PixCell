# Read .h5 file
# %%
import h5py

# Open the file
f = h5py.File('../patches/metadata/controlnet_consep_small.hdf5', 'r')

# Print the keys
print(f.keys())

# Print the first 5 entries of the first key
print(f['controlnet_256'][:5])
print(len(f['controlnet_256']))

# Close the file
# %%
