import pandas as pd
import matplotlib.pyplot as plt
from socket import gethostname
import re

DEVICE_NAME = gethostname()
FIG_SIZE = (16, 9)


def ham_str(i):
    return f"perc_hamilton_found_{i}"


col_basic = ["date", "description", ham_str(100)]
col_training = ["train_epochs", "learn_rate", "train_example_details", "final_loss_exp", "final_loss_var", "final_loss_trend"]
col_standard = ["date", "description", "train_epochs"] + [ham_str(i) for i in range(10, 110, 10)] + ["train_example_details"]


def recent(df, to_display=30):
    df["day"] = df["date"].str.split().apply(lambda x: x[0])
    df = df.sort_values(by=["day", ham_str(100)], ascending=[False, False])
    df.drop(["day"], axis=1)
    print(df[col_standard].iloc[:to_display])
    return df


def best(df, to_display=30):
    df = df.sort_values(by=ham_str(100), ascending=False)
    print(df[col_standard].iloc[:to_display])
    return df


def model_depth(df):
    def compute_depth(x):
        lines = x.model_raw.splitlines()
        depth = sum([1 if re.match(r"^[^A-Za-z].*[0-9]\s*$", l.strip()) is not None else 0 for l in lines])
        return depth
    return df.apply(compute_depth, axis=1)


def model_params(df):
    def compute_params(x):
        matches = re.findall(r"Total params:\s*[0-9,]+", x.model_raw)
        params = sum([int(m.split()[2].replace(",", "").strip()) for m in matches])
        return params
    return df.apply(compute_params, axis=1)


def _load_db():
    df = pd.read_csv("EVALUATIONS/Evaluation_results.csv")
    df = df.sort_values(by=ham_str(100), ascending=False)
    df["parameters"] = model_params(df)
    df["depth"] = model_depth(df)
    return df


df = _load_db()
eval_sizes = [int(x) for c in df.columns if c.startswith("perc_hamilton") for x in c.split("_") if x.isdigit()]
eval_sizes.sort()


def count_hamilton_cycles_in_data():
    from hamgnn.DatasetBuilder import ErdosRenyiInMemoryDataset
    dataset = ErdosRenyiInMemoryDataset(["DATA"])
    hamiltonian_per_size = {}
    examples_per_size = {}
    for d, hamilton_cycle in dataset:
        if d.num_nodes not in hamiltonian_per_size:
            examples_per_size[d.num_nodes] = 1
        else:
            examples_per_size[d.num_nodes] += 1

        if hamilton_cycle is None or len(hamilton_cycle) == 0:
            continue
        if d.num_nodes not in hamiltonian_per_size:
            hamiltonian_per_size[d.num_nodes] = 1
        else:
            hamiltonian_per_size[d.num_nodes] += 1
    return {k: hamiltonian_per_size[k] / examples_per_size[k] for k in hamiltonian_per_size}


def accuracy_curves(df, max_displayed=5):
    if len(df) > max_displayed:
        df = df.iloc[:max_displayed]
    data = df[[ham_str(i) for i in eval_sizes]].to_numpy()
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.get_cmap("hsv", df.shape[0] + 1)
    for row_index in range(data.shape[0]):
        description = df.description.iloc[row_index] if "description" in df.columns else f"model {row_index}"
        ax.plot(eval_sizes, [100*x for x in data[row_index,:]], color=cmap(row_index), marker=".", label=description)

    hamiltonian_per_size = count_hamilton_cycles_in_data()
    ax.plot(eval_sizes, [100*hamiltonian_per_size[s] if s in hamiltonian_per_size else 0 for s in eval_sizes],
            color="black", linestyle="dashed", label="Percentage of Hamiltonian graphs")

    ax.set_xlabel("Graph size")
    ax.set_xticks(eval_sizes)
    ax.set_ylabel("% of solutions")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    return fig


print(df[col_standard].iloc[:30])
