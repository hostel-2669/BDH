import pandas as pd
import matplotlib.pyplot as plt
from safe import RobustSafetySystem


system = RobustSafetySystem()


df = pd.read_csv("data.csv")


predictions = []

for text in df["text"]:
    result = system.evaluate(text)
    predictions.append(result["risk"])

df["prediction"] = predictions


harmful_labels = ["SENSITIVE", "REQUIRES_REVIEW", "DANGEROOUS", "CRITICAL"]

df["pred_harmful"] = df["prediction"].isin(harmful_labels)
df["true_harmful"] = df["label"] == "harmful"


TP = len(df[(df["pred_harmful"] == True) & (df["true_harmful"] == True)])
TN = len(df[(df["pred_harmful"] == False) & (df["true_harmful"] == False)])
FP = len(df[(df["pred_harmful"] == True) & (df["true_harmful"] == False)])
FN = len(df[(df["pred_harmful"] == False) & (df["true_harmful"] == True)])


total = len(df)

Accuracy = (TP + TN) / total
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
FNR = FN / (TP + FN) if (TP + FN) > 0 else 0

print("Confusion Matrix Values:")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nMetrics:")
print("Accuracy:", round(Accuracy,3))
print("Precision:", round(Precision,3))
print("Recall:", round(Recall,3))
print("FPR:", round(FPR,3))
print("FNR:", round(FNR,3))



matrix = [[TP, FN],
          [FP, TN]]

plt.figure()
plt.imshow(matrix, cmap="Blues")
plt.xticks([0,1], ["Pred Harmful", "Pred Safe"])
plt.yticks([0,1], ["Actual Harmful", "Actual Safe"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, matrix[i][j], ha="center", va="center")

plt.title("Confusion Matrix")
plt.colorbar()
plt.show()



plt.figure()
plt.bar(["False Positive", "False Negative"], [FP, FN])
plt.title("False Positive vs False Negative")
plt.ylabel("Count")
plt.show()



plt.figure()
plt.bar(["FPR", "FNR"], [FPR, FNR])
plt.title("False Positive Rate vs False Negative Rate")
plt.ylabel("Rate")
plt.ylim(0,1)
plt.show()



plt.figure()
plt.bar(["Precision", "Recall"], [Precision, Recall])
plt.title("Precision vs Recall")
plt.ylabel("Score")
plt.ylim(0,1)
plt.show()
