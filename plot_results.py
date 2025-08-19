import pandas as pd
import matplotlib.pyplot as plt

# Load your log CSV
df = pd.read_csv("server_updates.csv")

# Check what columns exist
print("Columns in CSV:", df.columns)

# Adjust column names if needed
round_col = "round" if "round" in df.columns else df.columns[0]
client_col = "client_id" if "client_id" in df.columns else df.columns[1]
acc_col = "accuracy" if "accuracy" in df.columns else df.columns[2]

plt.figure(figsize=(10, 6))

# Plot accuracy per client
for client, group in df.groupby(client_col):
    plt.plot(group[round_col], group[acc_col], marker='o', label=f"{client}")

# Compute & plot global average accuracy per round
global_acc = df.groupby(round_col)[acc_col].mean().reset_index()
plt.plot(global_acc[round_col], global_acc[acc_col], color="black",
         linewidth=2.5, linestyle="--", label="🌍 Global Avg")

# Styling
plt.title("Federated Learning - Accuracy per Client and Global Average")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend(title="Client", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save plot as PNG
plt.savefig("federated_accuracy_trend.png", dpi=300)
print("✅ Plot saved as 'federated_accuracy_trend.png'")

# Show plot
plt.show()
