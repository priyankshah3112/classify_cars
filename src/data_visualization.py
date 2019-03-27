from matplotlib import pyplot as plt
import numpy as np
import os
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)


def histograms_plot(df,label='unknown_data'):

    df.hist()
    plt.show()
    plt.savefig('graphs/feature_histograms'+str(label)+'.png')
    return

def correlation_matrix_plot(df,label='unknown_data'):

    names=df.columns.values.tolist()
    correlations = df.corr()
    correlations.to_csv('reports/features_correlation_after_cleaning.csv')
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 7, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names,rotation=90)
    ax.set_yticklabels(names)
    plt.show()
    plt.savefig('graphs/feature_correlation_plot'+str(label)+'.png')

