'''
Read csv file without header
'''
df = pd.read_csv(eval_file, delimiter=" ", header=None)
df.columns = ["img_path", "dset_idx"]


'''
Simple date indexing
'''

# assuming a time column ['2016-01-07', '2016-01-08' ..]
dft = pd.to_datetime(df["time"])
idx = dft < "2016-01-10" # will give correct selection

'''
Adding columns
'''

#faster to concatenate two dataframes than adding a column as in df["C"] = blabla


'''
Concatenating dataframe
'''

# Make sure indices are compatible, otherwise unexpected results
