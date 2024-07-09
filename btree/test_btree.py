import unittest
from btree import Node, BTree
import tempfile
import time


class TestNode(unittest.TestCase):
    def setUp(self):
        self.node = Node()

    def test_init(self):
        self.assertEqual(self.node.keys, [])
        self.assertEqual(self.node.values, [])
        self.assertEqual(self.node.children, [])
        self.assertEqual(self.node.parent_index, -1)
        self.assertEqual(self.node.index, -1)

    def test_to_bytes(self):
        self.node.keys = ["key1", "key2"]
        self.node.values = ["value1", "value2"]
        self.node.children = [1, 2]
        self.node.parent_index = 0
        self.node.index = 1
        data = self.node.to_bytes()
        self.assertIsInstance(data, bytes)

    def test_from_bytes(self):
        self.node.keys = ["key1", "key2"]
        self.node.values = ["value1", "value2"]
        self.node.children = [1, 2]
        self.node.parent_index = 0
        self.node.index = 1
        data = self.node.to_bytes()
        self.node.from_bytes(data)
        self.assertEqual(self.node.keys, ["key1", "key2"])
        self.assertEqual(self.node.values, ["value1", "value2"])
        self.assertEqual(self.node.children, [1, 2])
        self.assertEqual(self.node.parent_index, 0)
        self.assertEqual(self.node.index, 1)

    def test_to_bytes_and_from_bytes(self):
        self.node.keys = ["key1", "key2"]
        self.node.values = ["value1", "value2"]
        self.node.children = [1, 2]
        self.node.parent_index = 0
        self.node.index = 3
        bytes_representation = self.node.to_bytes()
        new_node = Node()
        new_node.from_bytes(bytes_representation)
        self.assertEqual(new_node.keys, self.node.keys)
        self.assertEqual(new_node.values, self.node.values)
        self.assertEqual(new_node.children, self.node.children)
        self.assertEqual(new_node.parent_index, self.node.parent_index)
        self.assertEqual(new_node.index, self.node.index)

    def test_write_to_disk_and_read_from_disk(self):
        self.node.keys = ["key1", "key2"]
        self.node.values = ["value1", "value2"]
        self.node.children = [1, 2]
        self.node.parent_index = 0
        self.node.index = 3
        with tempfile.TemporaryFile() as f:
            self.node.write_to_disk(self.node.index, f)
            new_node = Node()
            new_node.read_from_disk(self.node.index, f)
            self.assertEqual(new_node.keys, self.node.keys)
            self.assertEqual(new_node.values, self.node.values)
            self.assertEqual(new_node.children, self.node.children)
            self.assertEqual(new_node.parent_index, self.node.parent_index)
            self.assertEqual(new_node.index, self.node.index)

    def test_write_to_disk_and_read_from_disk_2(self):
        self.node.keys = ["key1", "key2"]
        self.node.values = ["value1", "value2"]
        self.node.children = [1, 2]
        self.node.parent_index = 0
        self.node.index = 3
        with tempfile.TemporaryFile() as f:
            self.node.write_to_disk(self.node.index, f)
            new_node = Node()
            new_node.read_from_disk(self.node.index, f)
            self.assertEqual(new_node.keys, self.node.keys)
            self.assertEqual(new_node.values, self.node.values)
            self.assertEqual(new_node.children, self.node.children)
            self.assertEqual(new_node.parent_index, self.node.parent_index)
            self.assertEqual(new_node.index, self.node.index)

            # Write the node to disk again, havine one less key
            new_node.keys = ["key1"]
            new_node.values = ["value1"]
            new_node.children = [1]
            new_node.write_to_disk(new_node.index, f)

            # Read the node from disk again
            new_node = Node()
            new_node.read_from_disk(self.node.index, f)
            self.assertEqual(new_node.keys, ["key1"])
            self.assertEqual(new_node.values, ["value1"])
            self.assertEqual(new_node.children, [1])
            self.assertEqual(new_node.parent_index, 0)

    def test_leaf(self):
        self.assertTrue(self.node.leaf())
        self.node.children = [1, 2]
        self.assertFalse(self.node.leaf())


class BTreeTests(unittest.TestCase):
    def test_insert_and_search(self):
        btree = BTree(3)
        for i in range(10):
            btree.insert(f"{i}", f"{i * 2}")
        for i in range(10):
            result = btree.search(f"{i}")
            self.assertIsNotNone(result)
            node, index = result
            self.assertEqual(node.keys[index], f"{i}")
            self.assertEqual(node.values[index], f"{i * 2}")
        for i in range(10, 20):
            result = btree.search(i)
            self.assertIsNone(result)

    def test_open_close(self):
        btree = BTree(3, reopen=False, filename="test_open_close.dat")
        for i in range(10):
            btree.insert(f"{i}", f"{i * 2}")
        btree.close()
        time.sleep(1)
        btree = None
        new_btree = BTree(3, reopen=True, filename="test_open_close.dat")
        time.sleep(1)
        #new_btree.open()
        for i in range(10):
            result = new_btree.search(f"{i}")
            self.assertIsNotNone(result)
            node, index = result
            self.assertEqual(node.keys[index], f"{i}")
            self.assertEqual(node.values[index], f"{i * 2}")
        for i in range(10, 20):
            result = new_btree.search(i)
            self.assertIsNone(result)

    def test_million_inserts_and_search(self):
        btree = BTree(15)
        for i in range(100):
            btree.insert(f"{i}", f"{i * 2}")
        for i in range(100):
            result = btree.search(f"{i}")
            self.assertIsNotNone(result)
            node, index = result
            self.assertEqual(node.keys[index], f"{i}")
            self.assertEqual(node.values[index], f"{i * 2}")
        for i in range(100, 200):
            result = btree.search(i)
            self.assertIsNone(result)

    def test_insert_non_full(self):
        # Create a BTree with t=6
        btree = BTree(6)
        root = btree.read_node_from_disk(btree.root_index)
        # Insert keys and values into the BTree
        btree.insert_non_full(root, 10, "A")
        btree.insert_non_full(root, 20, "B")
        btree.insert_non_full(root, 30, "C")
        btree.insert_non_full(root, 40, "D")
        btree.insert_non_full(root, 50, "E")

        # Check if the keys and values are inserted correctly
        root = btree.read_node_from_disk(btree.root_index)
        assert root.keys == [10, 20, 30, 40, 50]
        assert root.values == ["A", "B", "C", "D", "E"]

        # Check if the children indices are updated correctly
        assert root.children == [0, 2]

        # Check if the parent indices are updated correctly
        for i in range(5):
            child = btree.read_node_from_disk(btree.root.children[i])
            assert child.parent_index == btree.root_index

        # Insert a new key and value into the BTree
        btree.insert_non_full(btree.root, 25, "F")

        # Check if the new key and value are inserted correctly
        assert btree.root.keys == [10, 20, 25, 30, 40, 50]
        assert btree.root.values == ["A", "B", "F", "C", "D", "E"]

        # Check if the children indices are updated correctly
        assert btree.root.children == [0, 1, 2, 3, 4, 5]

        # Check if the parent indices are updated correctly
        for i in range(6):
            child = btree.read_node_from_disk(btree.root.children[i])
            assert child.parent_index == btree.root.index

        # Close the BTree file
        btree.close()

    def test_insert(self):
        # Create a BTree with t=2
        btree = BTree(2)

        # Insert key-value pairs into the BTree
        btree.insert(1, "One")
        btree.insert(2, "Two")
        btree.insert(3, "Three")

        # Assert that the keys and values are correctly inserted
        # Ensure it exists in node at index 0
        self.assertEqual(btree.search(1)[1], 0)
        self.assertEqual(btree.search(2)[1], 1)  # Exists in node at index 1
        self.assertEqual(btree.search(3)[1], 2)  # Exists in node at index 2

        # Close the BTree
        btree.close()

    def test_search(self):
        # Create a BTree with t=2
        btree = BTree(2)

        # Insert key-value pairs into the BTree
        btree.insert(1, "One")
        btree.insert(2, "Two")
        btree.insert(3, "Three")

        # Search for keys in the BTree
        self.assertEqual(btree.search(1)[0].keys[0], 1)
        self.assertEqual(btree.search(2)[0].keys[1], 2)
        self.assertEqual(btree.search(3)[0].keys[2], 3)

        # Search for keys that do not exist in the BTree
        self.assertIsNone(btree.search(4))

        # Close the BTree
        btree.close()

    def test_split_child(self):
        btree = BTree(3)
        node = Node()
        node.keys = [1, 2, 3, 4, 5]
        node.values = [10, 20, 30, 40, 50]
        node.children = []
        node.parent_index = 0
        node.index = 0

        btree.split_child(node, 2)

        self.assertEqual(node.keys, [1, 2, 4, 5])
        self.assertEqual(node.values, [10, 20, 40, 50])
        self.assertEqual(node.children, [])
        self.assertEqual(node.parent_index, 0)
        self.assertEqual(node.index, 0)

        new_node = btree.read_node_from_disk(1)
        self.assertEqual(new_node.keys, [3])
        self.assertEqual(new_node.values, [30])
        self.assertEqual(new_node.children, [])
        self.assertEqual(new_node.parent_index, 0)
        self.assertEqual(new_node.index, 1)

    def test_split_root(self):
        btree = BTree(3)
        btree.insert(1, "value1")
        btree.insert(2, "value2")
        btree.insert(3, "value3")
        btree.insert(4, "value4")
        btree.insert(5, "value5")
        btree.insert(6, "value6")
        root = btree.read_node_from_disk(btree.root_index)
        self.assertEqual(root.keys, [3])
        self.assertEqual(btree.read_node_from_disk(root.children[0]).keys, [1,2])
        self.assertEqual(btree.read_node_from_disk(root.children[1]).keys, [4,5,6])

    def test_split_internal_node(self):
        btree = BTree(3)
        # Insert keys 0-8 to trigger a split of the root node
        for i in range(0, 9):
            btree.insert(i, f"value{i}")
        # Insert key 9 to trigger a split of an internal node
        btree.insert(9, "value9")
        # Verify that the tree has the correct structure after the split
        root = btree.read_node_from_disk(btree.root_index)
        self.assertEqual(root.keys, [2, 5])
        self.assertEqual(btree.read_node_from_disk(root.children[0]).keys, [0, 1])
        self.assertEqual(btree.read_node_from_disk(root.children[1]).keys, [3, 4])
        self.assertEqual(btree.read_node_from_disk(root.children[2]).keys, [6, 7, 8, 9])

    def test_split_internal_node_2(self):
        btree = BTree(3)
        # Insert keys 0-12 to trigger a split of the root node
        for i in range(0, 13):
            btree.insert(i, f"value{i}")
        # Insert key 13 to trigger a split of an internal node
        btree.insert(13, "value13")
        # Verify that the tree has the correct structure after the split
        root = btree.read_node_from_disk(btree.root_index)
        self.assertEqual(root.keys, [2, 5, 8])
        self.assertEqual(btree.read_node_from_disk(root.children[0]).keys, [0, 1])
        self.assertEqual(btree.read_node_from_disk(root.children[1]).keys, [3, 4])
        self.assertEqual(btree.read_node_from_disk(root.children[2]).keys, [6, 7])
        self.assertEqual(btree.read_node_from_disk(root.children[3]).keys, [9, 10, 11, 12, 13])

    def test_split_internal_node_3(self):
        btree = BTree(3)
        # Insert keys 0-13 to trigger a split of the root node
        for i in range(0, 14):
            btree.insert(i, f"value{i}")
        # Insert key 13 to trigger a split of an internal node
        btree.insert(14, "value14")
        # Verify that the tree has the correct structure after the split
        root = btree.read_node_from_disk(btree.root_index)
        self.assertEqual(root.keys, [2, 5, 8, 11])
        self.assertEqual(btree.read_node_from_disk(root.children[0]).keys, [0, 1])
        self.assertEqual(btree.read_node_from_disk(root.children[1]).keys, [3, 4])
        self.assertEqual(btree.read_node_from_disk(root.children[2]).keys, [6, 7])
        self.assertEqual(btree.read_node_from_disk(root.children[3]).keys, [9, 10])
        self.assertEqual(btree.read_node_from_disk(root.children[4]).keys, [12, 13, 14])

    def test_split_internal_node_4(self):
        btree = BTree(3)
        # Insert keys 0-17 to trigger a split of the root node
        for i in range(0, 18):
            btree.insert(i, f"value{i}")
        # Insert key 13 to trigger a split of an internal node
        btree.insert(18, "value18")
        # Verify that the tree has the correct structure after the split
        root = btree.read_node_from_disk(btree.root_index)
        self.assertEqual(root.keys, [8])
        left_internal_node = btree.read_node_from_disk(root.children[0])
        right_internal_node = btree.read_node_from_disk(root.children[1])
        self.assertEqual(left_internal_node.keys, [2, 5])
        self.assertEqual(right_internal_node.keys, [11, 14])
        self.assertEqual(btree.read_node_from_disk(left_internal_node.children[0]).keys, [0, 1])
        self.assertEqual(btree.read_node_from_disk(left_internal_node.children[1]).keys, [3, 4])
        self.assertEqual(btree.read_node_from_disk(left_internal_node.children[2]).keys, [6, 7])
        self.assertEqual(btree.read_node_from_disk(left_internal_node.children[3]).keys, [9, 10])
        self.assertEqual(btree.read_node_from_disk(right_internal_node.children[0]).keys, [12, 13])
        self.assertEqual(btree.read_node_from_disk(right_internal_node.children[1]).keys, [15, 16, 17, 18])


if __name__ == "__main__":
    unittest.main()

# To run the test:
# $ python -m unittest test_btree.py

"""

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

"""