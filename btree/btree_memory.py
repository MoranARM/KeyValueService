VERBOSE = False


class Node:
    def __init__(self, index=-1):
        self.keys = []
        self.values = []
        self.children_indices = []  # holds indices of child nodes
        self.parent_index = -1  # Index of the parent node
        self.index = index  # Index of the node

    def leaf(self):
        return len(self.children_indices) == 0


class BTree:

    def __init__(self, t):
        self.pages = dict()
        self.next_index = 0
        root = Node(self.next_index)
        self.root_index = root.index
        self.t = t
        self.pages[self.root_index] = root
        self.next_index += 1

    def search(self, key, node=None):
        node = self.pages[self.root_index] if node == None else node
        if VERBOSE:
            print(f"searching for {key} in {node.keys} with index {node.index}")

        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return (node, i)
        elif node.leaf():
            return None
        else:
            if VERBOSE:
                print(f"searching for {key} in {node.keys}")
                print(f"node.children_indices: {node.children_indices}")
                print(f"node.children_indices[i]: {node.children_indices[i]}")
                print(
                    f"self.pages[node.children_indices[i]]: {self.pages[node.children_indices[i]]}"
                )
            return self.search(key, self.pages[node.children_indices[i]])

    def split_child(self, x, i):
        t = self.t

        # Write x to disk and read it in again to ensure that we are working with the latest version of x
        self.pages[x.index] = x
        x = self.pages[x.index]

        # y is a full child of x
        if VERBOSE:
            print(f"splitting child {i} of {x.keys}")
            print(f"x.children_indices: {x.children_indices}")
            print(f"x.children_indices[i]: {x.children_indices[i]}")
            print(
                f"self.pages[x.children_indices[i]]: {self.pages[x.children_indices[i]]}"
            )
        y = self.pages[x.children_indices[i]]

        # Write y to disk and read it in again to ensure that we are working with the latest version of y'
        self.pages[y.index] = y
        y = self.pages[y.index]
        assert y.parent_index == x.index

        # create a new node and add it to x's list of children
        z = Node(self.next_index)
        z.parent_index = x.index
        self.pages[z.index] = z
        self.next_index += 1
        if VERBOSE:
            print(f"making z node with self.next_index: {z.index}")
        x.children_indices.insert(i + 1, z.index)

        # insert the median of the full child y into x
        x.keys.insert(i, y.keys[t - 1])
        x.values.insert(i, y.values[t - 1])

        # split apart y's keys into y & z
        z.keys = y.keys[t : (2 * t) - 1]
        z.values = y.values[t : (2 * t) - 1]
        y.keys = y.keys[0 : t - 1]
        y.values = y.values[0 : t - 1]

        self.pages[x.index] = x
        self.pages[y.index] = y
        self.pages[z.index] = z

        # if y is not a leaf, we reassign y's children to y & z
        if not y.leaf():
            z.children_indices = y.children_indices[t : 2 * t]
            y.children_indices = y.children_indices[0:t]

            # No need to update the parent index of the children of y since they are still children of y
            # update the parent index of the children of y
            for child_index in y.children_indices:
                child = self.pages[child_index]
                child.parent_index = y.index
                self.pages[child.index] = child

            # update the parent index of the children of z
            for child_index in z.children_indices:
                child = self.pages[child_index]
                child.parent_index = z.index
                self.pages[child.index] = child

    def insert(self, k, v):
        t = self.t
        root = self.pages[self.root_index]

        # check if the key is already in the tree
        search_result = self.search(k, root)
        # if the key is already in the tree, update the value
        if search_result is not None:
            node, i = search_result
            old_value = node.values[i]
            node.values[i] = v
            # Update the node in pages
            self.pages[node.index] = node
            return old_value

        # if root is full, create a new node - tree's height grows by 1
        if len(root.keys) == (2 * t) - 1:
            new_root = Node(self.next_index)
            self.next_index += 1
            if VERBOSE:
                print(f"making new root node with self.next_index: {new_root.index}")
            new_root.children_indices.insert(0, root.index)
            self.pages[new_root.index] = new_root
            self.root_index = new_root.index

            # Update the parent index of the children of the old root
            root.parent_index = new_root.index
            self.pages[root.index] = root
            self.split_child(new_root, 0)
            new_root = self.pages[new_root.index]
            self.insert_non_full(new_root, k, v)
            self.pages[new_root.index] = new_root
        else:
            self.insert_non_full(root, k, v)
            self.pages[root.index] = root

    def insert_non_full(self, x, k, v):
        self.pages[x.index] = x
        t = self.t
        i = len(x.keys) - 1

        # find the correct spot in the leaf to insert the key
        if x.leaf():
            x.keys.append(None)
            x.values.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                x.values[i + 1] = x.values[i]
                i -= 1
            if x.keys[i] == k:
                print("ERROR SHOULD NEVER REACH HERE")
                x.values[i] = v
                # Remove the appended None values
                print(f"popping key {x.keys[-1]} from {x.keys}")
                x.keys.pop()
                print(f"popping value {x.values[-1]} from {x.values}")
                x.values.pop()
            else:
                x.keys[i + 1] = k
                x.values[i + 1] = v

            self.pages[x.index] = x
        # if not a leaf, find the correct subtree to insert the key
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            # if child node is full, split it
            if len(self.pages[x.children_indices[i]].keys) == (2 * t) - 1:
                self.split_child(x, i)
                x = self.pages[x.index]
                if k > x.keys[i]:
                    i += 1
            x = self.pages[x.index]
            self.insert_non_full(self.pages[x.children_indices[i]], k, v)

    def print_tree(self, x, level=0):
        print(f"Level {level}", end=": ")

        for i, k in zip(range(len(x.keys)), x.keys):
            print(f"{i}:{k}", end=" ")

        print()
        level += 1

        if len(x.children_indices) > 0:
            for i, c in zip(range(len(x.children_indices)), x.children_indices):
                print(f"Child {i} of {x.keys}", end=": ")
                self.print_tree(self.pages[c], level)


def insert_and_search_example():
    B = BTree(3)
    print(f"Order {B.t}")

    for i in range(20):
        B.insert(i, i * 2)
    for i in range(20):
        B.insert(i, i * 3)

    B.print_tree(B.pages[B.root_index])
    print()

    keys_to_search_for = [1, 2, 3, 5, 8, 13, 21]
    for key in keys_to_search_for:
        if B.search(key) is not None:
            print(f"{key} is in the tree")
        else:
            print(f"{key} is NOT in the tree")

    print(f"Next Index: {B.next_index}")
    for k, v in B.pages.items():
        print(
            f"Index: {k}, Children: {v.children_indices}, Keys: {v.keys}, Values: {v.values}"
        )

    print(f"B.root.index: {B.root_index}")


def _average_over_trials(
    col_sub_name: str, trial_reported_times: dict, num_trials: int, reported_times: dict
):
    """
    Helper function to average the times over the trials
    """
    reported_times[f"Average {col_sub_name} Time"].append(
        sum(trial_reported_times[f"Average {col_sub_name} Time"]) / num_trials
    )
    reported_times[f"Max {col_sub_name} Time"].append(
        sum(trial_reported_times[f"Max {col_sub_name} Time"]) / num_trials
    )
    reported_times[f"Min {col_sub_name} Time"].append(
        sum(trial_reported_times[f"Min {col_sub_name} Time"]) / num_trials
    )
    reported_times[f"Total {col_sub_name} Time"].append(
        sum(trial_reported_times[f"Total {col_sub_name} Time"]) / num_trials
    )
    # For the first 10 insert times, we need to average each of the 10 times across the trials
    first_10_times = []
    for i in range(10):
        first_10_times.append(
            sum(
                [
                    trial_reported_times[f"First 10 {col_sub_name} Times"][j][i]
                    for j in range(num_trials)
                ]
            )
            / num_trials
        )
    reported_times[f"First 10 {col_sub_name} Times"].append(first_10_times)


def run_benchmark(
    suffix="mem", test_sizes=[10, 100, 1000, 10000, 100000, 1000000], num_trials=3
):
    # Additional imports required for benchmarking
    import pandas as pd  # Used for throwing the data into a csv
    import time  # Used for timing the inserts

    # Benchmark the BTree for different orders and object counts
    reported_times = {
        "Order": [],
        "Objects": [],
        "Average Insert Time": [],
        "Max Insert Time": [],
        "Min Insert Time": [],
        "Total Insert Time": [],
        "First 10 Insert Times": [],
        "Average Search Time": [],
        "Max Search Time": [],
        "Min Search Time": [],
        "Total Search Time": [],
        "First 10 Search Times": [],
        "Average Update Time": [],
        "Max Update Time": [],
        "Min Update Time": [],
        "Total Update Time": [],
        "First 10 Update Times": [],
    }
    for o in test_sizes:
        for t in range(2, 16):
            print(f"Order: {t}, Objects: {o}")

            trial_reported_times = {
                "Average Insert Time": [],
                "Max Insert Time": [],
                "Min Insert Time": [],
                "Total Insert Time": [],
                "First 10 Insert Times": [],
                "Average Search Time": [],
                "Max Search Time": [],
                "Min Search Time": [],
                "Total Search Time": [],
                "First 10 Search Times": [],
                "Average Update Time": [],
                "Max Update Time": [],
                "Min Update Time": [],
                "Total Update Time": [],
                "First 10 Update Times": [],
            }
            reported_times["Order"].append(t)
            reported_times["Objects"].append(o)

            for trial in range(num_trials):
                print(f"Trial: {trial+1} of {num_trials}")

                btree = BTree(t)
                insert_times = []

                print("Inserting...")

                for i in range(o):
                    # time each insert
                    start = time.time()
                    btree.insert(i, i * 2)
                    end = time.time()
                    insert_times.append(end - start)

                trial_reported_times["Average Insert Time"].append(
                    sum(insert_times) / len(insert_times)
                )
                trial_reported_times["Max Insert Time"].append(max(insert_times))
                trial_reported_times["Min Insert Time"].append(min(insert_times))
                trial_reported_times["Total Insert Time"].append(sum(insert_times))
                trial_reported_times["First 10 Insert Times"].append(insert_times[:10])

                search_times = []

                print("Searching...")

                # Time searches
                for i in range(o):
                    try:
                        start = time.time()
                        btree.search(i)
                        end = time.time()
                        search_times.append(end - start)
                    except RecursionError:
                        search_times.append(
                            1000
                        )  # Add a large placeholder value to indicate an error

                trial_reported_times["Average Search Time"].append(
                    sum(search_times) / len(search_times)
                )
                trial_reported_times["Max Search Time"].append(max(search_times))
                trial_reported_times["Min Search Time"].append(min(search_times))
                trial_reported_times["Total Search Time"].append(sum(search_times))
                trial_reported_times["First 10 Search Times"].append(search_times[:10])

                update_times = []

                print("Updating...")

                # Time updates
                for i in range(o):
                    try:
                        start = time.time()
                        btree.insert(i, i * 3)
                        end = time.time()
                        update_times.append(end - start)
                    except RecursionError:
                        update_times.append(
                            1000
                        )  # Add a large placeholder value to indicate an error

                trial_reported_times["Average Update Time"].append(
                    sum(update_times) / len(update_times)
                )
                trial_reported_times["Max Update Time"].append(max(update_times))
                trial_reported_times["Min Update Time"].append(min(update_times))
                trial_reported_times["Total Update Time"].append(sum(update_times))
                trial_reported_times["First 10 Update Times"].append(update_times[:10])

            # Average the times over the trials
            _average_over_trials("Insert", trial_reported_times, num_trials, reported_times)
            _average_over_trials("Search", trial_reported_times, num_trials, reported_times)
            _average_over_trials("Update", trial_reported_times, num_trials, reported_times)

    df = pd.DataFrame(reported_times)
    df.to_csv(f"btree_report_{suffix}.csv")

    gen_plots(df, suffix, test_sizes=test_sizes)


def _gen_plots(
    df,
    col_sub_name="Insert",
    prefix="In-Memory",
    suffix="mem",
    test_sizes=[10, 100, 1000, 10000, 100000, 1000000],
):
    """
    Helper function to generate plots for the benchmark results
    """
    import matplotlib.pyplot as plt

    for o in test_sizes:
        df_o = df[df["Objects"] == o]
        plt.plot(
            df_o["Order"], df_o[f"Average {col_sub_name} Time"], label=f"Objects: {o}"
        )
        # Save the plot
        plt.xlabel("Order")
        plt.ylabel(f"Average {col_sub_name} Time")
        plt.title(f"{prefix} Average {col_sub_name} Time vs. Order for {o} Objects")
        plt.legend()
        plt.savefig(f"btree_{col_sub_name}_{o}_{suffix}.png")
        plt.clf()

    # Create a plot for the total insert time
    for o in test_sizes:
        df_o = df[df["Objects"] == o]
        plt.plot(
            df_o["Order"], df_o[f"Total {col_sub_name} Time"], label=f"Objects: {o}"
        )
        # Save the plot
        plt.xlabel("Order")
        plt.ylabel(f"Total {col_sub_name} Time")
        plt.title(f"{prefix} Total {col_sub_name} Time vs. Order for {o} Objects")
        plt.legend()
        plt.savefig(f"btree_total_{col_sub_name}_{o}_{suffix}.png")
        plt.clf()

    # Create a plot for the max insert time
    for o in test_sizes:
        df_o = df[df["Objects"] == o]
        plt.plot(df_o["Order"], df_o[f"Max {col_sub_name} Time"], label=f"Objects: {o}")
        # Save the plot
        plt.xlabel("Order")
        plt.ylabel(f"Max {col_sub_name} Time")
        plt.title(f"{prefix} Max {col_sub_name} Time vs. Order for {o} Objects")
        plt.legend()
        plt.savefig(f"btree_max_{col_sub_name}_{o}_{suffix}.png")
        plt.clf()

    # Create a plot for the min insert time
    for o in test_sizes:
        df_o = df[df["Objects"] == o]
        plt.plot(df_o["Order"], df_o[f"Min {col_sub_name} Time"], label=f"Objects: {o}")
        # Save the plot
        plt.xlabel("Order")
        plt.ylabel(f"Min {col_sub_name} Time")
        plt.title(f"{prefix} Min {col_sub_name} Time vs. Order for {o} Objects")
        plt.legend()
        plt.savefig(f"btree_min_{col_sub_name}_{o}_{suffix}.png")
        plt.clf()

    # Create a plot for the first 10 insert times
    for o in test_sizes:
        df_o = df[df["Objects"] == o]
        for i in range(10):
            plt.plot(
                df_o["Order"],
                df_o[f"First 10 {col_sub_name} Times"].apply(lambda x: x[i]),
                label=f"{col_sub_name} #{i+1}",
            )
        # Save the plot
        plt.xlabel("Order")
        plt.ylabel(f"First 10 {col_sub_name} Times (s)")
        plt.title(f"{prefix} First 10 {col_sub_name} Times vs. Order for {o} Objects")
        plt.legend()
        plt.savefig(f"btree_first_10_{col_sub_name}_{o}_{suffix}.png")
        plt.clf()


def gen_plots(df, suffix, test_sizes=[10, 100, 1000, 10000, 100000, 1000000]):
    # Create plots for the benchmark results
    _gen_plots(df, "Insert", "In-Memory", suffix, test_sizes)
    _gen_plots(df, "Search", "In-Memory", suffix, test_sizes)
    _gen_plots(df, "Update", "In-Memory", suffix, test_sizes)


def plot_benchmark_results(filepath="", suffix="mem"):
    if filepath == "":
        filepath = f"btree_report_{suffix}.csv"

    # Additional imports required for benchmarking
    import pandas as pd  # Used for reading the data from a csv

    df = pd.read_csv(
        filepath,
        dtype={
            "First 10 Insert Times": "string",
            "First 10 Search Times": "string",
            "First 10 Update Times": "string",
        },
    )
    # Convert the string representation of the list to a list
    df["First 10 Insert Times"] = df["First 10 Insert Times"].apply(
        lambda x: [float(i.strip()) for i in x[1:-1].split(",")]
    )
    df["First 10 Search Times"] = df["First 10 Search Times"].apply(
        lambda x: [float(i.strip()) for i in x[1:-1].split(",")]
    )
    df["First 10 Update Times"] = df["First 10 Update Times"].apply(
        lambda x: [float(i.strip()) for i in x[1:-1].split(",")]
    )
    gen_plots(df, suffix)


def main():
    insert_and_search_example()

    run_benchmark()


if __name__ == "__main__":
    main()

"""
Order 3
Level 0: 0:8 
Child 0 of [8]: Level 1: 0:2 1:5 
Child 0 of [2, 5]: Level 2: 0:0 1:1 
Child 1 of [2, 5]: Level 2: 0:3 1:4 
Child 2 of [2, 5]: Level 2: 0:6 1:7 
Child 1 of [8]: Level 1: 0:11 1:14 
Child 0 of [11, 14]: Level 2: 0:9 1:10 
Child 1 of [11, 14]: Level 2: 0:12 1:13 
Child 2 of [11, 14]: Level 2: 0:15 1:16 2:17 3:18 4:19 

1 is in the tree
2 is in the tree
3 is in the tree
5 is in the tree
8 is in the tree
13 is in the tree
21 is NOT in the tree
Next Index: 9
Index: 0, Children: [], Keys: [0, 1], Values: [0, 3]
Index: 1, Children: [0, 2, 3], Keys: [2, 5], Values: [6, 15]
Index: 2, Children: [], Keys: [3, 4], Values: [9, 12]
Index: 3, Children: [], Keys: [6, 7], Values: [18, 21]
Index: 4, Children: [], Keys: [9, 10], Values: [27, 30]
Index: 5, Children: [], Keys: [12, 13], Values: [36, 39]
Index: 6, Children: [], Keys: [15, 16, 17, 18, 19], Values: [45, 48, 51, 54, 57]
Index: 7, Children: [1, 8], Keys: [8], Values: [24]
Index: 8, Children: [4, 5, 6], Keys: [11, 14], Values: [33, 42]
B.root.index: 7
"""
