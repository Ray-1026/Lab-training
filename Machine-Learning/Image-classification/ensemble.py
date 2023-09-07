import numpy as np
import pandas as pd
from tqdm import tqdm

# df1 = pd.read_csv("./submission/resnet18_submission.csv")
# df2 = pd.read_csv("./submission/vgg19_bn_submission.csv")
# df3 = pd.read_csv("./submission/efficientnet_v2_s_submission.csv")
# df4 = pd.read_csv("./submission/squeezenet1_0_submission.csv")

df1 = pd.read_csv("./submission/sample1_submission.csv")
df2 = pd.read_csv("./submission/sample2_submission.csv")
df3 = pd.read_csv("./submission/sample3_submission.csv")

dfs = [df1, df2, df3]
scores = [0.78082, 0.78733, 0.79008]
results = np.ones(3000, dtype=int) * -1

for i in tqdm(range(3000)):
    dic = {}
    for j in range(len(dfs)):
        p = dfs[j].iloc[i]["Category"]
        if p == -1:
            continue
        if p not in dic.keys():
            dic[p] = scores[j]
        else:
            dic[p] += scores[j]

    if not dic:
        continue
    results[i] = sorted(dic, key=lambda x: dic[x])[-1]


def pad4(i):
    return "0" * (4 - len(str(i))) + str(i)


df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(0, len(results))]
df["Category"] = results
df.to_csv("submission.csv", sep=",", index=False)
