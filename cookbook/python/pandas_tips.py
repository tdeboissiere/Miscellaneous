'''
Read csv file without header
'''
df = pd.read_csv(eval_file, delimiter=" ", header=None)
df.columns = ["img_path", "dset_idx"]
