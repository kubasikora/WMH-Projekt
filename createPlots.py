import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import itertools

path_prefix = "./gridsearch-results"

class FrameDescriptor:
    def __init__(self, name, additionalParams=[]):
        self.name = name
        self.params = ['C', 'epsilon', *additionalParams]
        self.outputPath = f"./gridsearch-results/plots/{name}"
        self.scores = ['mean_test_mean_squared_error', 'mean_test_mean_absolute_error', 'mean_test_r2']
        self.dataFrame = self.prune_dataframe(pd.read_csv(f"{path_prefix}/{self.name}.csv"))

    def generate_score_plot(self, input, output):
        input1 = input[0]
        input2 = input[1]

        fig = plt.figure()

        for value in self.dataFrame[f"param_{input1}"].unique():
            rows = self.dataFrame[self.dataFrame[f"param_{input1}"] == value]
            plt.plot(rows[f"param_{input2}"], rows[output], label=f"{input2}={value}")
    
        plt.title(f"{input1} & {input2} vs {output}")
        plt.xscale('log')
        plt.xlabel(f"{input2}")
        plt.grid()
        plt.legend()
        plt.savefig(f"{self.outputPath}/{input1}_{input2}_{output}.pdf")
        plt.savefig(f"{self.outputPath}/{input1}_{input2}_{output}.png")
        plt.close()

    def generate_3d_score_plot(self, input, output):
        input1 = input[0]
        input2 = input[1]

        shape = (len(self.dataFrame[f"param_{input1}"].unique()), len(self.dataFrame[f"param_{input2}"].unique()))

        print(input, output)
        print(pd.unique(self.dataFrame[[f"param_{input1}", f"param_{input2}"]]))
        x = self.dataFrame[f"param_{input1}"].to_numpy().reshape(shape)
        y = self.dataFrame[f"param_{input2}"].to_numpy().reshape(shape)
        z = self.dataFrame[output].to_numpy().reshape(shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(np.log10(x), np.log10(y), z, rstride=1, cstride=1, alpha=0.9)

        ax.set_xlabel(f"log10 ({input1})")
        ax.set_ylabel(f"log10 ({input2})")
        ax.set_zlabel(output)
    
        plt.title(f"{input1} & {input2} vs {output}")
        plt.savefig(f"{self.outputPath}/3d_{input1}_{input2}_{output}.pdf")
        plt.savefig(f"{self.outputPath}/3d_{input1}_{input2}_{output}.png")
        plt.close()

    def generate_all_plots(self):
        for (ins, outs) in itertools.product(itertools.combinations(self.params, 2), self.scores):
            self.generate_score_plot(ins, outs)
            self.generate_score_plot((ins[1], ins[0]), outs)
            self.generate_3d_score_plot(ins, outs)
        
    def prune_dataframe(self, df):
        df['mean_test_mean_absolute_error'] = (-1)*df['mean_test_neg_mean_absolute_error']
        df['mean_train_mean_absolute_error'] = (-1)*df['mean_train_neg_mean_absolute_error']
        df['mean_test_mean_squared_error'] = (-1)*df['mean_test_neg_mean_squared_error']
        df['mean_train_mean_squared_error'] = (-1)*df['mean_train_neg_mean_squared_error']
        df = df.drop(['params',
                     'split0_test_neg_mean_absolute_error',
                     'split1_test_neg_mean_absolute_error',
                     'split2_test_neg_mean_absolute_error',
                     'split3_test_neg_mean_absolute_error',
                     'split4_test_neg_mean_absolute_error',
                     'split0_train_neg_mean_absolute_error',
                     'split1_train_neg_mean_absolute_error',
                     'split2_train_neg_mean_absolute_error',
                     'split3_train_neg_mean_absolute_error',
                     'split4_train_neg_mean_absolute_error',
                     'split0_test_neg_mean_squared_error',
                     'split1_test_neg_mean_squared_error',
                     'split2_test_neg_mean_squared_error',
                     'split3_test_neg_mean_squared_error',
                     'split4_test_neg_mean_squared_error',
                     'split0_train_neg_mean_squared_error',
                     'split1_train_neg_mean_squared_error',
                     'split2_train_neg_mean_squared_error',
                     'split3_train_neg_mean_squared_error',
                     'split4_train_neg_mean_squared_error',
                     'split0_test_r2', 
                     'split1_test_r2', 
                     'split2_test_r2', 
                     'split3_test_r2',
                     'split4_test_r2',
                     'split0_train_r2', 
                     'split1_train_r2', 
                     'split2_train_r2',
                     'split3_train_r2', 
                     'split4_train_r2'], 
                axis=1)
        return df


dfs = [
        # FrameDescriptor("linear"), 
        FrameDescriptor("poly", ["degree", "coef0"]), 
        # FrameDescriptor("rbf"), 
        # FrameDescriptor("sigmoid", ["coef0"])
    ]

for df in dfs:
    print(df.name)
    df.generate_all_plots()
