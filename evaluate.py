import csv
import matplotlib.pyplot as plt

errors = []
manual_vals = []
model_vals = []

with open("static/severity_results.csv") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["manual_percent"]:
            manual = float(row["manual_percent"])
            model = float(row["model_percent"])

            error = abs(manual - model)

            errors.append(error)
            manual_vals.append(manual)
            model_vals.append(model)

if len(errors) == 0:
    print("No data available. Fill manual_percent.")
else:
    avg_error = sum(errors) / len(errors)

    print("\n--- Severity Validation ---")
    print("Samples:", len(errors))
    print("Average Error:", round(avg_error, 2), "%")

    # 📊 GRAPH (ADD HERE)
    plt.scatter(manual_vals, model_vals)

    plt.xlabel("Manual Severity (%)")
    plt.ylabel("Model Severity (%)")
    plt.title("Severity Validation")

    # ideal prediction line
    plt.plot([0,100], [0,100], linestyle='--')

    plt.grid(True)

    plt.show()