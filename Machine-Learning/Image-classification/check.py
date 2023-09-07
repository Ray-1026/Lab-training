import csv

submission = "submission.csv"
test_label = "test_label.csv"

with open(submission) as file:
    csv_reader = csv.reader(file)
    pred = list(csv_reader)
    pred.pop(0)

with open(test_label, "r", newline="") as file:
    csv_reader = csv.reader(file)
    label = list(csv_reader)
    label.pop(0)


correct = 0
num = 0
total = len(pred)
for i in range(total):
    num += 1
    if pred[i][1] == label[i][1]:
        correct += 1
print(f"acc: {correct / num:.5f}")
