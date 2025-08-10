import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import glob
import pandas as pd

npz_files = glob.glob("debug_gym_results/20250807_231522/*.npz")

for npz_file in npz_files:
    data = np.load(npz_file)
    df = pd.read_csv(npz_file.replace(".npz", ".csv"))
    print(data.keys())
    # plt.scatter(data['max_sequence_similarity_list'], data['lls'].mean(axis=1))
    plt.scatter(df['max_sequence_similarity'], df['spearman'])
    plt.xlabel("Max sequence similarity")
    plt.ylabel("Spearman correlation")
    plt.title(npz_file)
    savepath = npz_file.replace(".npz", "_spearman.png")
    plt.savefig(savepath)
    plt.close()
    plt.clf()
    bp=1