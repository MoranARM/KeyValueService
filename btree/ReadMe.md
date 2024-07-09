# B Tree Implementation

This repository contains an implementation of the B Tree data structure. 

## Overview

The B Tree is a self-balancing search tree that maintains sorted data and allows efficient operations such as insertion, deletion, and search. It is commonly used in databases and file systems.

## Features

- Search: The B Tree allows for fast searching of elements based on their keys.
- Insertion: The B Tree supports efficient insertion of new elements while maintaining the sorted order of the data.
- Update: Handled within insertion, if key is found value will be replaced

## Usage

To use the B Tree implementation, follow these steps:

1. Clone the repository: `git clone https://github.com/MoranARM/btree.git`
2. install python 3 (Validated on 3.9 and 3.12)
3. Run the program: `python btree_memory` for the in-memory implementation and `python btree.py` for the on-disk implementation
    a. Note that the DEBUG constant at the top of the file is not implemented as a parameter due to the vast quantity of assert and print statements done, which were used for debugging along with the python debugger when developing the code. It is only recommended if trying to modify the existing code.
4. Running the unit tests can be done with `python -m unittest test_btree_memory.py` for the in-memory implementation and `python3 -m unittest test_btree.py` for the on-disk implementation

## Example

Here are some examples of how to use the B Tree:

```python
def insert_and_search_example():
    btree = BTree(3)

    for i in range(20):
        if DEBUG:
            print(f"Inserting key {i} with value {i*2}")
        btree.insert(i, i * 2)

    btree.print_tree(btree.root)
    print()

    # 21 is expected to be missing in this example
    keys_to_search_for = [1, 2, 3, 5, 8, 13, 21] 
    for key in keys_to_search_for:
        if btree.search(key) is not None:
            print(f"{key} is in the tree")
        else:
            print(f"{key} is NOT in the tree")

    print(f"Next Index: {btree.next_index}")
    for i in range(btree.next_index):
        node = btree.read_node_from_disk(i)
        print(f"Node {i} has keys: {node.keys} and children: {node.children}")

    btree.close()
```

## Assumptions

* Every node should be written to disk after a moditifcation is made, before the next read of that node is done. This is especially true when passing nodes between functions.
* The on memory implementation uses indicies and a `pages` dictionary in order to best simulate still needing to access all nodes from a central location. This allows for only the indices to be stored in the nodes instead of the nodes themselves.
