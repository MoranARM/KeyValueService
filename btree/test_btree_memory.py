from btree_memory import BTree
import unittest

class BTreeTests(unittest.TestCase):
    def test_search_existing_key(self):
        btree = BTree(3)
        btree.insert(10, "value")
        result = btree.search(10)
        self.assertIsNotNone(result)
        node, index = result
        self.assertEqual(node.keys[index], 10)
        self.assertEqual(node.values[index], "value")

    def test_search_non_existing_key(self):
        btree = BTree(3)
        btree.insert(10, "value")
        result = btree.search(20)
        self.assertIsNone(result)

    def test_insert_existing_key(self):
        btree = BTree(3)
        btree.insert(10, "value1")
        btree.insert(10, "value2")
        result = btree.search(10)
        self.assertIsNotNone(result)
        node, index = result
        self.assertEqual(node.keys[index], 10)
        self.assertEqual(node.values[index], "value2")

    def test_insert_non_existing_key(self):
        btree = BTree(3)
        btree.insert(10, "value")
        result = btree.search(10)
        self.assertIsNotNone(result)
        node, index = result
        self.assertEqual(node.keys[index], 10)
        self.assertEqual(node.values[index], "value")

    def test_insert_100_keys(self):
        btree = BTree(3)
        for i in range(100):
            btree.insert(i, f"value{i}")
        for i in range(100):
            result = btree.search(i)
            self.assertIsNotNone(result)
            node, index = result
            self.assertEqual(node.keys[index], i)
            self.assertEqual(node.values[index], f"value{i}")
        for i in range(100, 200):
            result = btree.search(i)
            self.assertIsNone(result)

    def test_split_root(self):
        btree = BTree(3)
        # Insert keys 1-6 to trigger a split of the root node
        for i in range(1, 7):
            btree.insert(i, f"value{i}")
        root = btree.pages[btree.root_index]
        self.assertEqual(root.keys, [3])
        self.assertEqual(btree.pages[root.children_indices[0]].keys, [1,2])
        self.assertEqual(btree.pages[root.children_indices[1]].keys, [4,5,6])

    def test_split_internal_node(self):
        btree = BTree(3)
        # Insert keys 0-8 to trigger a split of the root node
        for i in range(0, 9):
            btree.insert(i, f"value{i}")
        # Insert key 9 to trigger a split of an internal node
        btree.insert(9, "value9")
        # Verify that the tree has the correct structure after the split
        root = btree.pages[btree.root_index]
        self.assertEqual(root.keys, [2, 5])
        self.assertEqual(btree.pages[root.children_indices[0]].keys, [0, 1])
        self.assertEqual(btree.pages[root.children_indices[1]].keys, [3, 4])
        self.assertEqual(btree.pages[root.children_indices[2]].keys, [6, 7, 8, 9])

    def test_split_internal_node_2(self):
        btree = BTree(3)
        # Insert keys 0-12 to trigger a split of the root node
        for i in range(0, 13):
            btree.insert(i, f"value{i}")
        # Insert key 9 to trigger a split of an internal node
        btree.insert(13, "value13")
        # Verify that the tree has the correct structure after the split
        root = btree.pages[btree.root_index]
        self.assertEqual(root.keys, [2, 5, 8])
        self.assertEqual(btree.pages[root.children_indices[0]].keys, [0, 1])
        self.assertEqual(btree.pages[root.children_indices[1]].keys, [3, 4])
        self.assertEqual(btree.pages[root.children_indices[2]].keys, [6, 7])
        self.assertEqual(btree.pages[root.children_indices[3]].keys, [9, 10, 11, 12, 13])

    def test_split_internal_node_3(self):
        btree = BTree(3)
        # Insert keys 0-13 to trigger a split of the root node
        for i in range(0, 14):
            btree.insert(i, f"value{i}")
        # Insert key 14 to trigger a split of an internal node
        btree.insert(14, "value14")
        # Verify that the tree has the correct structure after the split
        root = btree.pages[btree.root_index]
        self.assertEqual(root.keys, [2, 5, 8, 11])
        self.assertEqual(btree.pages[root.children_indices[0]].keys, [0, 1])
        self.assertEqual(btree.pages[root.children_indices[1]].keys, [3, 4])
        self.assertEqual(btree.pages[root.children_indices[2]].keys, [6, 7])
        self.assertEqual(btree.pages[root.children_indices[3]].keys, [9, 10])
        self.assertEqual(btree.pages[root.children_indices[4]].keys, [12, 13, 14])

    def test_split_internal_node_4(self):
        btree = BTree(3)
        # Insert keys 0-17 to trigger a split of the root node
        for i in range(0, 18):
            btree.insert(i, f"value{i}")
        # Insert key 14 to trigger a split of an internal node
        btree.insert(18, "value18")
        # Verify that the tree has the correct structure after the split
        root = btree.pages[btree.root_index]
        self.assertEqual(root.keys, [8])
        left_internal_node = btree.pages[root.children_indices[0]]
        right_internal_node = btree.pages[root.children_indices[1]]
        self.assertEqual(left_internal_node.keys, [2, 5])
        self.assertEqual(right_internal_node.keys, [11, 14])
        self.assertEqual(btree.pages[left_internal_node.children_indices[0]].keys, [0, 1])
        self.assertEqual(btree.pages[left_internal_node.children_indices[1]].keys, [3, 4])
        self.assertEqual(btree.pages[left_internal_node.children_indices[2]].keys, [6, 7])
        self.assertEqual(btree.pages[right_internal_node.children_indices[0]].keys, [9, 10])
        self.assertEqual(btree.pages[right_internal_node.children_indices[1]].keys, [12, 13])
        self.assertEqual(btree.pages[right_internal_node.children_indices[2]].keys, [15, 16, 17, 18])

    def test_update_existing_value(self):
        btree = BTree(3)
        btree.insert(10, "value1")
        btree.insert(10, "value2")
        result = btree.search(10)
        self.assertIsNotNone(result)
        node, index = result
        self.assertEqual(node.keys[index], 10)
        self.assertEqual(node.values[index], "value2")

    def test_insert_duplicate_values(self):
        btree = BTree(3)
        for i in range(100):
            btree.insert(i, f"value{i}")
            btree.insert(i, f"value{i}_dup")
        for i in range(100):
            result = btree.search(i)
            self.assertIsNotNone(result)
            node, index = result
            self.assertEqual(node.keys[index], i)
            self.assertEqual(node.values[index], f"value{i}_dup")
        for i in range(100, 200):
            result = btree.search(i)
            self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()

# To run the test:
# $ python3 -m unittest test_btree_memory.py
