import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame
data = {
    'Year': [2021, 2022, 2023],
    'Users': [1500, 3000, 5000]
}

df = pd.DataFrame(data)

# Plot using pandas + matplotlib
plt.plot(df['Year'], df['Users'], marker='o')
plt.title("User Growth Over Time")
plt.xlabel("Year")
plt.ylabel("Users")
plt.grid(True)
plt.show()