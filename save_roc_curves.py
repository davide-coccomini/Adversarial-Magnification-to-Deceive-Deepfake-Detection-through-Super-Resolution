import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
CSV_PATH = "outputs/test/all_metrics_tests_scale2/rates.csv"
METHOD_COLORS = {"Deepfakes": "c", "Face2Face": "r", "FaceShifter": "g", "FaceSwap": "#F27F0C", "NeuralTextures": "#190087" }

OUTPUT_DIR = "outputs/test/all_metrics_tests_scale2/roc_curve"
MODELS = ["Resnet50", "Swin", "XceptionNet"]

df = pd.read_csv(CSV_PATH, sep=",")
df = df.sort_values(by=['Model'])

dataframes = []
for model in MODELS:
    dataframes.append(df[df["Model"]==model])


for index, df in enumerate(dataframes):
    plt.figure(index)
    #plt.figure(figsize=(9,8))
    plt.plot([0, 1], [0, 1], 'k--')
    for row in df.iterrows():
        row = row[1]
        model_name = row["Model"]
        method = row["Method"]
        sr = row["SR"]
        fpr = row["FPR"].split(" ")
        tpr = row["TPR"].split(" ")
        fpr = [float(value) for value in fpr]
        tpr = [float(value) for value in tpr]
        if bool(sr):
            linestyle = "dashed"
            text = " SR"
        else:
            linestyle = "solid"
            text = ""

        model_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, METHOD_COLORS[method], linestyle=linestyle, label=method + text)
        

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    #plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, -0.05))

    plt.savefig(OUTPUT_DIR + model_name + ".png")
    plt.clf()
