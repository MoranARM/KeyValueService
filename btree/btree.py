import os
import struct
import sys
import ctypes

PAGE_SIZE = 4096
KEY_VALUE_SIZE = 64  # Size of a key or value in bytes
# Ensure that each set of key-value pairs in a node fit in a single page
MAX_KEYS = (PAGE_SIZE - 16) // KEY_VALUE_SIZE // 2  # Maximum number of keys in a node
MAX_ORDER = MAX_KEYS // 2  # Maximum number of children in a node
MIN_ORDER = 2  # Minimum number of children in a node
DEBUG = False  # Prints debug statements and runs asserts if True


class Node:
    supported_types = [float, int, str]

    def __init__(self, index=-1, key_dtype=str, value_dtype=str):
        self.keys = []
        self.values = []  # Store values in the nodes
        self.children_indices = []  # Store the indices of the children nodes
        self.parent_index = -1  # Index of the parent node on disk
        self.index = index  # Index of the node on disk
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype

    def leaf(self) -> bool:
        """
        Check if the node is a leaf node.
        """
        return len(self.children_indices) == 0

    def _dtype_to_enum(self, dtype):
        """
        Get the index of the data type in the supported types list.
        """
        return Node.supported_types.index(dtype)

    def _enum_to_dtype(self, enum):
        """
        Get the data type from the index in the supported types list.
        """
        return Node.supported_types[enum]

    def write_to_disk(self, index: int, file: ctypes.c_void_p) -> None:
        """
        Write the node to disk at the given index.
        """
        self.index = index
        if DEBUG:
            print(f"Current size of the file: {os.path.getsize('btree.dat')}")
            print(f"PAGE_SIZE: {PAGE_SIZE}, index: {index}")
            print(f"len(self.keys): {len(self.keys)}")
            print(f"self.keys: {self.keys}")
            print(f"self.parent_index: {self.parent_index}")
            print(f"self.index: {self.index}")
        file.seek(PAGE_SIZE * index)
        data = self.to_bytes()
        padded_data = data + b"\x00" * (
            PAGE_SIZE - len(data)
        )  # Pad the data to the page size
        if DEBUG:
            print(f"len(padded_data): {len(padded_data)}")
        file.write(padded_data)
        if DEBUG:
            print(
                f"Updated size of the file after writing: {os.path.getsize('btree.dat')}"
            )
        if DEBUG:
            # Force the OS to write the pages to disk
            os.fsync(file.fileno())

    def read_from_disk(self, index: int, file: ctypes.c_void_p) -> None:
        """
        Read the node from disk at the given index.
        """
        if DEBUG:
            # Force the OS to write the pages to disk
            os.fsync(file.fileno())
        self.index = index
        file.seek(PAGE_SIZE * index)
        data = file.read(PAGE_SIZE)
        self.from_bytes(data)

    def to_bytes(self):
        """
        Convert the node to bytes.
        """
        byteorder = "<" if sys.byteorder == "little" else ">"
        format_str = byteorder + "iiiiii"
        if DEBUG:
            print(f"to_bytes self.index: {self.index}")
            print(f"to_bytes self.parent_index: {self.parent_index}")
            print(f"to_bytes len(self.keys): {len(self.keys)}")
            print(f"to_bytes len(self.values): {len(self.values)}")
            print(f"to_bytes len(self.children_indices): {len(self.children_indices)}")

        packed_ints = struct.pack(
            format_str,
            self.index,
            self.parent_index,
            len(self.keys),
            len(self.children_indices),
            self._dtype_to_enum(self.key_dtype),
            self._dtype_to_enum(self.value_dtype),
        )

        packed_keys = self.to_bytes_keys()
        packed_values = self.to_bytes_values()
        packed_children = self.to_bytes_children()

        data = packed_ints + packed_keys + packed_values + packed_children
        if DEBUG:
            print(f"to_bytes len(data): {len(data)}")
        return data

    def from_bytes(self, data: bytes):
        """
        Load the node from bytes.
        """
        byteorder = "<" if sys.byteorder == "little" else ">"
        format_str = byteorder + "iiiiii"
        self.index, self.parent_index, keys_vals_len, child_len, key_enum, val_enum = (
            struct.unpack(format_str, data[:24])
        )

        self.key_dtype = self._enum_to_dtype(key_enum)
        self.value_dtype = self._enum_to_dtype(val_enum)

        if DEBUG:
            print(
                f"from_bytes Loading from bytes: index: {self.index}, parent_index: {self.parent_index}, keys_vals_len: {keys_vals_len}, child_len: {child_len}"
            )

        keys_start = 24
        keys_end = keys_start + keys_vals_len * KEY_VALUE_SIZE
        values_start = keys_end
        values_end = values_start + keys_vals_len * KEY_VALUE_SIZE
        child_start = values_end
        child_end = child_start + child_len * 4  # 4 bytes per child index

        self.from_bytes_keys(data[keys_start:keys_end], keys_vals_len)
        self.from_bytes_values(data[values_start:values_end], keys_vals_len)
        self.from_bytes_children(data[child_start:child_end], child_len)

    def _to_bytes_list(self, lst: list) -> bytes:
        """
        Convert a list to bytes.
        """
        byteorder = "<" if sys.byteorder == "little" else ">"
        packed_lst = b""
        for item in lst:
            try:
                packed_lst += struct.pack(byteorder + "64s", item.encode())
            except AttributeError:
                # parse the key as a string
                packed_lst += struct.pack(byteorder + "64s", str(item).encode())
        return packed_lst

    def to_bytes_children(self) -> bytes:
        """
        Convert the children to bytes.
        """
        byteorder = "<" if sys.byteorder == "little" else ">"
        packed_lst = b""
        for item in self.children_indices:
            packed_lst += struct.pack(byteorder + "i", item)
        return packed_lst

    def to_bytes_keys(self) -> bytes:
        """
        Convert the keys to bytes.
        """
        return self._to_bytes_list(self.keys)

    def to_bytes_values(self) -> bytes:
        """
        Convert the values to bytes.
        """
        return self._to_bytes_list(self.values)

    def _from_bytes_list(self, data: bytes, lst_len: int, dtype: type) -> list:
        """
        Load a list from bytes.
        """
        byteorder = "<" if sys.byteorder == "little" else ">"
        format_str = byteorder + "64s"
        # Unpack each string of size KEY_VALUE_SIZE bytes and decode it, removing any null bytes
        string_list = [
            dtype(
                struct.unpack(
                    format_str, data[i * KEY_VALUE_SIZE : (i + 1) * KEY_VALUE_SIZE]
                )[0]
                .decode()
                .strip("\x00")
            )
            for i in range(lst_len)
        ]
        return string_list

    def from_bytes_keys(self, data, keys_len):
        """
        Load the keys from bytes.
        """
        self.keys = self._from_bytes_list(data, keys_len, self.key_dtype)

    def from_bytes_values(self, data, values_len):
        """
        Load the values from bytes.
        """
        self.values = self._from_bytes_list(data, values_len, self.value_dtype)

    def from_bytes_children(self, data, children_len):
        """
        Load the children from bytes.
        """
        byteorder = "<" if sys.byteorder == "little" else ">"
        format_str = byteorder + "i"
        self.children_indices = []
        for i in range(children_len):
            if DEBUG:
                print(f"Loading in value for child {i} out of expected {children_len}")
            self.children_indices.append(
                struct.unpack(format_str, data[i * 4 : (i + 1) * 4])[0]
            )
        if DEBUG:
            print(f"Loaded children from bytes: {self.children_indices}")


class BTree:
    def __init__(self, t: int, reopen=False, filename="btree.dat"):
        """
        Intiailizes the BTree with an order t bounded by {MIN_ORDER} and {MAX_ORDER}
        """
        self.t = max(MIN_ORDER, min(t, MAX_ORDER))  # Minimum degree of the BTree
        if DEBUG:
            print(f"Creating BTree with t={t}")
            print(f"New BTree with root node: self.next_index: {self.next_index}")
            self.read_count = 0
            self.write_count = 0
        self.next_index = 0  # Next available page index for nodes
        root = Node(self.next_index)
        self.root_index = root.index
        if not os.path.exists(filename):
            open(
                filename, "wb", buffering=0
            ).close()  # Create an empty file if it doesn't exist
        self.file = open(filename, "r+b", buffering=0)
        self.filename = filename
        if DEBUG:
            self.pages = dict()  # Store the pages in memory
        # Either load in the existing B Tree or write the root of the new one
        if reopen:
            self.open()
        else:
            self.write_node_to_disk(self.root_index, root)
            self.next_index += 1
            self.key_dtype = None
            self.value_dtype = None
        if DEBUG:
            self.pages[self.root_index] = root

    def search(self, key, node=None) -> tuple:
        """
        Search for a key in the BTree
        Note that this search function assumes a balanced tree with unique keys
        """
        if type(key) != self.key_dtype:
            if DEBUG:
                print(f"Attempting to cast key to {self.key_dtype}")
            try:
                key = self.key_dtype(key)
                if DEBUG:
                    print(f"Successfully cast key to {self.key_dtype}, new key: {key}")
            except ValueError:
                raise ValueError(f"Query key {key} is not of type {self.key_dtype}")
        node = self.read_node_from_disk(self.root_index) if node == None else node
        # node = self.root if node == None else node
        if DEBUG:
            node_page = self.pages[node.index] if node == None else node
            assert node.index == node_page.index
            assert node.keys == node_page.keys
            assert node.values == node_page.values
            assert node.children_indices == node_page.children_indices
            assert node.parent_index == node_page.parent_index
        if DEBUG and key == 9:
            print(
                f"Searching for key: {key} in node: {node.index} with root: {self.root_index} with keys: {node.keys} and children: {node.children_indices} and parent: {node.parent_index}"
            )

        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            if DEBUG:
                print(f"Found key {key} at index {i} with value {node.values[i]}")
            return (node, i)
        elif node.leaf():
            return None
        else:
            if DEBUG:
                disk_node = self.read_node_from_disk(node.children_indices[i])
                page_node = self.pages[node.children_indices[i]]
                assert disk_node.index == page_node.index
                assert disk_node.keys == page_node.keys
                assert disk_node.values == page_node.values
                assert disk_node.children_indices == page_node.children_indices
                assert disk_node.parent_index == page_node.parent_index
            return self.search(key, self.read_node_from_disk(node.children_indices[i]))

    def split_child(self, x: Node, index: int):
        t = self.t

        if DEBUG:
            print(f"split_child 236 x.index: {x.index}")
            for ind in range(len(x.children_indices)):
                print(
                    f"split_child 238 x.children_indices[i]: {x.children_indices[ind]}"
                )
                child = self.read_node_from_disk(x.children_indices[ind])
                print(f"split_child 240 child.index: {child.index}")
                print(f"split_child 241 child.parent_index: {child.parent_index}")
                assert child.parent_index == x.index

            print(f"split_child x.index: {x.index}")
            print(f"split_child x.keys: {x.keys}")
            print(f"split_child x.values: {x.values}")
            print(f"split_child x.children_indices: {x.children_indices}")
            print(f"split_child index: {index}")
            print(f"split_child x.parent_index: {x.parent_index}")

            assert x.children_indices[index] == index

        # y is a node with full keys and a child of x
        y = self.read_node_from_disk(x.children_indices[index])
        if DEBUG:
            y_page = self.pages[x.children_indices[index]]
            assert y.index == y_page.index
            assert y.keys == y_page.keys
            assert y.values == y_page.values
            assert y.children_indices == y_page.children_indices
            assert y.parent_index == y_page.parent_index

        if DEBUG:
            print(f"split_child y.index: {y.index}")
            print(f"split_child y.keys: {y.keys}")
            print(f"split_child y.values: {y.values}")
            print(f"split_child y.children_indices: {y.children_indices}")
            print(f"split_child y.parent_index: {y.parent_index}")

        # create a new node and add it to node x's children
        if DEBUG:
            print(f"split_child making z node with self.next_index: {self.next_index}")
        z = Node(
            self.next_index, key_dtype=self.key_dtype, value_dtype=self.value_dtype
        )
        z.parent_index = x.index
        self.write_node_to_disk(z.index, z)

        if DEBUG:
            self.pages[z.index] = z
            assert z.index == self.next_index
            assert z.parent_index == x.index

        self.next_index += 1
        if DEBUG:
            print(f"split_child z.index: {z.index}")
            print(f"split_child z.parent_index: {z.parent_index}")
            print(
                f"split_child len(x.children_indices) before insert: {len(x.children_indices)}"
            )
            print(f"split_child i: {index}")
            print(f"split_child x index: {x.index}")
        x.children_indices.insert(index + 1, z.index)
        if DEBUG:
            print(
                f"split_child len(x.children_indices) after insert: {len(x.children_indices)}"
            )
            print(f"split_child x.children_indices: {x.children_indices}")

        # insert the median of the full child node y's keys and values into node x (parent)
        x.keys.insert(index, y.keys[t - 1])
        x.values.insert(index, y.values[t - 1])

        # split apart node y's keys into nodes y and z (y is the left half, z is the right half)
        z.keys = y.keys[t : (2 * t) - 1]
        z.values = y.values[t : (2 * t) - 1]
        y.keys = y.keys[0 : t - 1]
        y.values = y.values[0 : t - 1]

        # Write the nodes to disk
        self.write_node_to_disk(x.index, x)
        self.write_node_to_disk(y.index, y)
        self.write_node_to_disk(z.index, z)

        if DEBUG:
            self.pages[x.index] = x
            self.pages[y.index] = y
            self.pages[z.index] = z

            # CONFIRMATION OF VALUES

            disk_x = self.read_node_from_disk(x.index)
            page_x = self.pages[x.index]
            assert disk_x.index == page_x.index
            assert disk_x.keys == page_x.keys
            assert disk_x.values == page_x.values
            print(f"disk_x.children_indices: {disk_x.children_indices}")
            print(f"page_x.children_indices: {page_x.children_indices}")
            assert disk_x.children_indices == page_x.children_indices
            assert disk_x.parent_index == page_x.parent_index

            disk_y = self.read_node_from_disk(y.index)
            page_y = self.pages[y.index]
            assert disk_y.index == page_y.index
            assert disk_y.keys == page_y.keys
            assert disk_y.values == page_y.values
            assert disk_y.children_indices == page_y.children_indices
            assert disk_y.parent_index == page_y.parent_index

            disk_z = self.read_node_from_disk(z.index)
            page_z = self.pages[z.index]
            assert disk_z.index == page_z.index
            assert disk_z.keys == page_z.keys
            assert disk_z.values == page_z.values
            assert disk_z.children_indices == page_z.children_indices
            assert disk_z.parent_index == page_z.parent_index

            # END OF CONFIRMATION OF VALUES

        # if y is not a leaf node, reassign node y's children to nodes y and z
        if not y.leaf():
            z.children_indices = y.children_indices[t : 2 * t]
            y.children_indices = y.children_indices[0:t]
            # No need to update the parent index of the children of y since they are still children of y
            # Update the parent index of the children of y
            for child_index in y.children_indices:
                child = self.read_node_from_disk(child_index)
                child.parent_index = y.index
                self.write_node_to_disk(child.index, child)
                if DEBUG:
                    self.pages[child.index] = child

            # Update the parent index of the children of z
            for child_index in z.children_indices:
                if DEBUG:
                    print(
                        f"split_child child_index: {child_index} for z index: {z.index}"
                    )
                    print(f"split_child z.children_indices: {z.children_indices}")
                    print(f"split_child z.parent_index: {z.parent_index}")
                child = self.read_node_from_disk(child_index)
                if DEBUG:
                    child_page = self.pages[child_index]
                    assert child.index == child_page.index
                    assert child.keys == child_page.keys
                    assert child.values == child_page.values
                    assert child.children_indices == child_page.children_indices

                    print(f"split_child former parent_index: {child.parent_index}")
                child.parent_index = z.index
                if DEBUG:
                    assert child.parent_index == z.index
                    assert child.index == child_index
                    print(f"split_child new parent_index: {child.parent_index}")
                self.write_node_to_disk(child.index, child)
                if DEBUG:
                    self.pages[child.index] = child

    def insert(self, k, v):
        t = self.t
        if DEBUG and k == 9:
            print(f"Inserting key {k} with value {v}")
            self.print_tree(self.read_node_from_disk(self.root_index))
            print()
        # Read in the latest root from disk
        root = self.read_node_from_disk(self.root_index)
        if DEBUG:
            root_index = root.index

        if DEBUG:
            root_page = self.pages[self.root_index]
            assert root.index == root_page.index
            print("Asserting root and root_page are the same")
            print(f"insert root.index: {root.index}")
            print(f"insert root.parent_index: {root.parent_index}")
            print(f"insert root.keys: {root.keys}")
            print(f"insert root.values: {root.values}")
            print(f"insert root.children_indices: {root.children_indices}")
            print(f"insert root_page.index: {root_page.index}")
            print(f"insert root_page.parent_index: {root_page.parent_index}")
            print(f"insert root_page.keys: {root_page.keys}")
            print(f"insert root_page.values: {root_page.values}")
            print(f"insert root_page.children_indices: {root_page.children_indices}")
            print(f"insert self.root_index: {self.root_index}")
            print(f"insert self.root.parent_index: {self.root.parent_index}")
            print(f"insert self.root.keys: {self.root.keys}")
            print(f"insert self.root.values: {self.root.values}")
            print(f"insert self.root.children_indices: {self.root.children_indices}")
            assert root.keys == root_page.keys
            assert root.values == root_page.values
            assert root.children_indices == root_page.children_indices

        # Check if there is a defined key data type
        if self.key_dtype is None:
            key_dtype = type(k)
            if key_dtype not in Node.supported_types:
                raise ValueError(
                    f"Key type {key_dtype} is not supported. Supported types: {Node.supported_types}"
                )
            self.key_dtype = key_dtype
            root.key_dtype = key_dtype
        else:
            if self.key_dtype != type(k):
                raise ValueError(
                    f"Key {k} is not of the expected type {self.key_dtype}"
                )
        # Similarly define the value data type
        if self.value_dtype is None:
            value_dtype = type(v)
            if value_dtype not in Node.supported_types:
                raise ValueError(
                    f"Value type {value_dtype} is not supported. Supported types: {Node.supported_types}"
                )
            self.value_dtype = value_dtype
            root.value_dtype = value_dtype
        else:
            if self.value_dtype != type(v):
                raise ValueError(
                    f"Value {v} is not of the expected type {self.value_dtype}"
                )

        # check if the key is already in the tree
        search_result = self.search(k)
        if DEBUG and k == 9:
            print(f"search_result: {search_result}")
        # if the key is already in the tree, update the value
        if search_result is not None:
            node, index = search_result
            node.values[index] = v
            # Update the node to disk
            self.write_node_to_disk(node.index, node)
            if DEBUG:
                self.pages[node.index] = node
            return

        # if the root node is full, create a new node and increment the tree's height
        if len(root.keys) == (2 * t) - 1:
            if DEBUG:
                print(f"making new root node with self.next_index: {self.next_index}")
            # create new node as the root
            new_root = Node(
                self.next_index, key_dtype=self.key_dtype, value_dtype=self.value_dtype
            )
            if DEBUG:
                print(f"new_root.index: {new_root.index}")
                print(f"Current root index: {root.index}")
                print(f"Current disk root index: {self.root_index}")
                assert new_root.index == self.next_index
                assert root.index != new_root.index
            self.next_index += 1
            if DEBUG:
                assert new_root.index == self.next_index - 1
            # add the old root as a child of the new root
            # new_root.children_indices.insert(0, root.index)
            # new_root.children_indices.insert(0, root_index)
            new_root.children_indices.insert(0, self.root_index)
            if DEBUG:
                print(f"insert root_index: {root_index}")
                print(f"insert self.root_index: {self.root_index}")
                print(f"insert root.index: {root.index}")
                assert root_index == self.root_index
                assert root.index == root_index
                print(f"Writing to new root index: {new_root.index}")
            self.write_node_to_disk(new_root.index, new_root)
            if DEBUG:
                self.pages[new_root.index] = new_root

            # self.root = new_root  # update the tree's root to the new root
            self.root_index = new_root.index

            if DEBUG:
                print(f"Updated new_root.index: {new_root.index}")
                print(f"Updated current root index: {root.index}")
                print(f"Updated current disk root index: {self.root.index}")

            # Set the old root's parent to the new root
            root.parent_index = new_root.index
            self.write_node_to_disk(root.index, root)
            if DEBUG:
                self.pages[root.index] = root

                print(f"insert root.index: {root.index}")
                print(f"insert root.parent_index: {root.parent_index}")
                print(f"insert root.keys: {root.keys}")
                print(f"insert root.values: {root.values}")
                print(f"insert root.children_indices: {root.children_indices}")
                print(f"insert new_root.index: {new_root.index}")
                print(f"insert new_root.parent_index: {new_root.parent_index}")
                print(f"insert new_root.keys: {new_root.keys}")
                print(f"insert new_root.values: {new_root.values}")
                print(f"insert new_root.children_indices: {new_root.children_indices}")
            self.split_child(
                new_root, 0
            )  # split old root node and add to new node's children
            self.insert_non_full(new_root, k, v)
        else:
            self.insert_non_full(root, k, v)

        if DEBUG:
            print(f"XXXXX inserted {k} : {v} into root node {self.root.index} XXXXXX")
            self.print_tree(self.root)
            print(
                f"XXXXX End of printing tree after inserting {k} : {v} into root node {self.root.index} XXXXXX"
            )

    def insert_non_full(self, x: Node, k, v):
        self.write_node_to_disk(x.index, x)
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
                x.values[i] = v
                # Remove the appended None values
                if DEBUG:
                    print(f"popping key {x.keys[-1]} from {x.keys}")
                    print(f"popping value {x.values[-1]} from {x.values}")
                x.keys.pop()
                x.values.pop()
            else:
                x.keys[i + 1] = k
                x.values[i + 1] = v
            if DEBUG:
                print(f"insert_non_full inserted {k} : {v} into leaf node {x.index}")
            # Write the node to disk
            self.write_node_to_disk(x.index, x)
            if DEBUG:
                self.pages[x.index] = x
        # if not a leaf, find the correct subtree to insert the key
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            # if child node is full, split it
            if DEBUG:
                child = self.read_node_from_disk(x.children_indices[i])
                print(f"insert_non_full 362 child.index: {child.index}")
                print(f"insert_non_full 363 child.parent_index: {child.parent_index}")
                print(f"insert_non_full 364 child.keys: {child.keys}")
                print(f"insert_non_full 365 child.values: {child.values}")
                print(
                    f"insert_non_full 366 child.children_indices: {child.children_indices}"
                )
                assert child.parent_index == x.index

                print(f"insert_non_full x.index: {x.index}")
                print(f"insert_non_full x.keys: {x.keys}")
                print(f"insert_non_full x.values: {x.values}")
                print(f"insert_non_full x.children_indices: {x.children_indices}")
                for child_index in x.children_indices:
                    child_disk = self.read_node_from_disk(child_index)
                    child_page = self.pages[child_index]
                    print(f"insert_non_full child_disk.index: {child_disk.index}")
                    print(
                        f"insert_non_full child_disk.parent_index: {child_disk.parent_index}"
                    )
                    print(f"insert_non_full child_disk.keys: {child_disk.keys}")
                    print(f"insert_non_full child_disk.values: {child_disk.values}")
                    print(
                        f"insert_non_full child_disk.children_indices: {child_disk.children_indices}"
                    )
                    assert child_disk.parent_index == x.index
                    print(f"insert_non_full child_page.index: {child_page.index}")
                    print(
                        f"insert_non_full child_page.parent_index: {child_page.parent_index}"
                    )
                    print(f"insert_non_full child_page.keys: {child_page.keys}")
                    print(f"insert_non_full child_page.values: {child_page.values}")
                    print(
                        f"insert_non_full child_page.children_indices: {child_page.children_indices}"
                    )
                    assert child_page.parent_index == x.index
                    assert child_disk.index == child_page.index
                    assert child_disk.keys == child_page.keys
                    assert child_disk.values == child_page.values
                    assert child_disk.children_indices == child_page.children_indices

                print(
                    f"insert_non_full disk x keys: {self.read_node_from_disk(x.children_indices[i]).keys}"
                )
                print(
                    f"insert_non_full page x keys: {self.pages[x.children_indices[i]].keys}"
                )

                assert len(self.read_node_from_disk(x.children_indices[i]).keys) == len(
                    self.pages[x.children_indices[i]].keys
                )
                assert (
                    self.read_node_from_disk(x.children_indices[i]).keys
                    == self.pages[x.children_indices[i]].keys
                )
                assert (
                    self.read_node_from_disk(x.children_indices[i]).values
                    == self.pages[x.children_indices[i]].values
                )
                assert (
                    self.read_node_from_disk(x.children_indices[i]).children_indices
                    == self.pages[x.children_indices[i]].children_indices
                )

            if len(self.read_node_from_disk(x.children_indices[i]).keys) == (2 * t) - 1:
                self.split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self.insert_non_full(self.read_node_from_disk(x.children_indices[i]), k, v)

    def get_node_levels(self, node: Node, level=0) -> dict:
        """
        Get the levels of the nodes in the BTree.
        """
        if node.leaf():
            return {node.index: level}
        levels = {node.index: level}
        for child in node.children_indices:
            child_node = self.read_node_from_disk(child)
            levels.update(self.get_node_levels(child_node, level + 1))
        return levels

    def print_tree(self, x: Node):
        """
        Print the BTree.
        """
        node_levels = self.get_node_levels(x)
        for i in range(self.next_index):
            node = self.read_node_from_disk(i)
            level = node_levels[i]
            print(
                f"Node {i} has children: {node.children_indices} keys: {node.keys} and values: {node.values} and parent index: {node.parent_index} at level {level}"
            )

    def write_node_to_disk(self, index: int, node: Node):
        """
        Write the node to disk at the given index.
        """
        node.write_to_disk(index, self.file)
        if DEBUG:
            self.write_count += 1

    def read_node_from_disk(self, index: int):
        """
        Read the node from disk at the given index.
        """
        node = Node(index=index)  # Create a new node object
        node.read_from_disk(index, self.file)
        if DEBUG:
            self.read_count += 1
        return node

    def close(self):
        """
        Used for closing the file/database
        """
        # Ensure that the root node has no parent
        # Will be unable to load the file later if this is not the case
        assert self.read_node_from_disk(self.root_index).parent_index == -1
        os.fsync(self.file.fileno())  # Force the OS to write the pages to disk
        self.file.close()

    def open(self):
        """
        Used to read in and open the file/database
        """
        self.file = open("btree.dat", "r+b", buffering=0)
        # Read in the node at index 0
        potential_root = self.read_node_from_disk(0)
        # Recursively check if the parent index is not -1
        while potential_root.parent_index != -1:
            potential_root = self.read_node_from_disk(potential_root.parent_index)
        # Set the root node
        self.root = potential_root

        # Set the proper next index
        self.next_index = os.path.getsize(self.filename) // PAGE_SIZE + 1

        # Find the key dtype and value dtype
        if len(self.root.keys) > 0:
            self.key_dtype = type(self.root.keys[0])
            self.value_dtype = type(self.root.values[0])
        else:
            # If the root node is empty, set the key and value dtypes to None
            if self.root.leaf():
                self.key_dtype = None
                self.value_dtype = None
            else:
                # If the root node is not a leaf, set the key and value dtypes to the first child's key and value dtypes
                first_child = self.read_node_from_disk(self.root.children_indices[0])
                self.key_dtype = type(first_child.keys[0])
                self.value_dtype = type(first_child.values[0])

        if DEBUG:
            if hasattr(self, "pages"):
                # Iterate through each page and read in the nodes
                for i in range(self.next_index):
                    self.pages[i] = self.read_node_from_disk(i)


def insert_and_search_example():
    btree = BTree(3)

    for i in range(20):
        if DEBUG:
            print(f"Inserting key {i} with value {i*2}")
        btree.insert(i, i * 2)

    # Prove that values are replaced if the key already exists
    for i in range(20):
        if DEBUG:
            print(f"Inserting key {i} with value {i*3}")
        btree.insert(i, i * 3)

    btree.print_tree(btree.read_node_from_disk(btree.root_index))
    print()

    keys_to_search_for = [2, 9, 21, 4]
    for key in keys_to_search_for:
        if btree.search(key) is not None:
            print(f"{key} is in the tree")
        else:
            print(f"{key} is NOT in the tree")

    if DEBUG:
        print(f"Total read count: {btree.read_count}")
        print(f"Total write count: {btree.write_count}")

    print(f"btree.root.index: {btree.root_index}")

    if DEBUG:
        print(f"Next Index: {btree.next_index}")
        for k, v in btree.pages.items():
            print(f"Index: {k}, Keys: {v.keys}, Children: {v.children_indices}")

    if DEBUG:
        # Count how many times each key is repeated, sorted by key
        key_counts = {}
        # Iterate through each page and read in the nodes
        for i in range(btree.next_index):
            node = btree.read_node_from_disk(i)
            for key in node.keys:
                if key in key_counts:
                    key_counts[key] += 1
                else:
                    key_counts[key] = 1

        # Print the key counts greater than 1
        for key, count in key_counts.items():
            if count > 1:
                print(f"Key {key} is repeated {count} times")

    btree.close()


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


def run_benchmark(suffix="disk", test_sizes=[100, 10000, 1000000], num_trials=3):
    # Additional imports required for benchmarking
    import pandas as pd  # Used for throwing the data into a csv
    import time  # Used for timing the inserts
    import random  # Used for generating random numbers to search for

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
        for t in range(15, 1, -1): # Test orders from 15 to 2
            if o >= 1000000:
                if t not in [15, 12, 10, 8, 5]: # Only test these orders for 1 million objects
                    continue
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

                total_insert_time_start = time.time()

                insertion_completed = False

                try:
                    for i in range(o):
                        # time each insert
                        start = time.time()
                        btree.insert(i, i * 2)
                        end = time.time()
                        insert_times.append(end - start)
                    insertion_completed = True
                    total_insert_time_start = time.time() - total_insert_time_start
                    # if DEBUG:
                    print(f"Total Insert Time: {total_insert_time_start}")
                except RecursionError:
                    if DEBUG:
                        print(f"Recursion error at Order: {t}, Objects: {o}")
                    insert_times.append(
                        1000
                    )

                trial_reported_times["Average Insert Time"].append(
                    sum(insert_times) / len(insert_times)
                )
                trial_reported_times["Max Insert Time"].append(max(insert_times))
                trial_reported_times["Min Insert Time"].append(min(insert_times))
                trial_reported_times["Total Insert Time"].append(sum(insert_times))
                trial_reported_times["First 10 Insert Times"].append(insert_times[:10])

                if not insertion_completed:
                    # Skip the search and update benchmarks if the insertion failed
                    trial_reported_times["Average Search Time"].append(1000)
                    trial_reported_times["Max Search Time"].append(1000)
                    trial_reported_times["Min Search Time"].append(1000)
                    trial_reported_times["Total Search Time"].append(1000)
                    trial_reported_times["First 10 Search Times"].append([1000] * 10)
                    trial_reported_times["Average Update Time"].append(1000)
                    trial_reported_times["Max Update Time"].append(1000)
                    trial_reported_times["Min Update Time"].append(1000)
                    trial_reported_times["Total Update Time"].append(1000)
                    trial_reported_times["First 10 Update Times"].append([1000] * 10)
                    continue

                search_times = []

                print("Searching...")

                total_search_time_start = time.time()

                # Generate 30 random numbers within the range of the objects
                random_searches = random.sample(range(o), 30)

                # Time searches
                for i in random_searches:
                    try:
                        start = time.time()
                        btree.search(i)
                        end = time.time()
                        search_times.append(end - start)
                    except RecursionError:
                        if DEBUG:
                            print(f"Recursion error at Order: {t}, Objects: {o}")
                        search_times.append(
                            1000
                        )  # Add a large placeholder value to indicate an error
                # if DEBUG:
                print(f"Total Search Time: {time.time() - total_search_time_start}")

                trial_reported_times["Average Search Time"].append(
                    sum(search_times) / len(search_times)
                )
                trial_reported_times["Max Search Time"].append(max(search_times))
                trial_reported_times["Min Search Time"].append(min(search_times))
                trial_reported_times["Total Search Time"].append(sum(search_times))
                trial_reported_times["First 10 Search Times"].append(search_times[:10])

                update_times = []

                print("Updating...")

                total_update_time_start = time.time()

                # Generate 30 random numbers within the range of the objects
                random_searches = random.sample(range(o), 30)

                # Time updates
                for i in random_searches:
                    try:
                        start = time.time()
                        btree.insert(i, i * 3)
                        end = time.time()
                        update_times.append(end - start)
                    except RecursionError:
                        if DEBUG:
                            print(f"Recursion error at Order: {t}, Objects: {o}")
                        update_times.append(
                            1000
                        )  # Add a large placeholder value to indicate an error
                # if DEBUG:
                print(f"Total Update Time: {time.time() - total_update_time_start}")

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
    prefix="On-Disk",
    suffix="disk",
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
    _gen_plots(df, "Insert", "On-Disk", suffix, test_sizes)
    _gen_plots(df, "Search", "On-Disk", suffix, test_sizes)
    _gen_plots(df, "Update", "On-Disk", suffix, test_sizes)


def plot_benchmark_results(filepath="", suffix="disk"):
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
    # Remove the existing file to start with a fresh tree
    try:
        os.remove("btree.dat")
    except OSError:
        pass

    # Run the example
    # insert_and_search_example()

    # Run the benchmark
    run_benchmark()

    # Plot the benchmark results without re-running the benchmark
    # plot_benchmark_results()


if __name__ == "__main__":
    main()

"""
Expected Output:

2 is in the tree
9 is in the tree
21 is NOT in the tree
4 is in the tree
Next Index:  7
Node 0 has keys: ['0', '1'] and children: []
Node 1 has keys: ['10', '13', '16', '2', '5'] and children: [0, 2, 3, 4, 5, 6]
Node 2 has keys: ['3', '4'] and children: []
Node 3 has keys: ['6', '7', '8', '9'] and children: []
Node 4 has keys: ['11', '12'] and children: []
Node 5 has keys: ['14', '15'] and children: []
Node 6 has keys: ['17', '18', '19'] and children: []

Total read count: 51
Total write count: 43
"""
