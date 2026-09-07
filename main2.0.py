import pandas as pd

data = {
    "Name": ["Rahul", "Aman", None, "Priya"],
    "Marks": [80, None, 75, 90],
    "Attendance": [85, 90, None, None]
}

df = pd.DataFrame(data)

print(df.isnull())
print(df.isnull().sum())

