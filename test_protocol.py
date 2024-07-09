import unittest
from protocol import to_bytes, from_bytes, ProtocolOperationCode, ProtocolDataType


class TestToBytes(unittest.TestCase):
    def test_to_bytes(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 8  # b'\x08'
        key = "test_key"  # b'test_key'
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 10  # b'\x0a'
        value = "test_value"  # b'test_value'

        expected_result = b"\x00\x02\x08test_key\x02\x0atest_value"
        result = to_bytes(operation_code, key, key_data_type, value, value_data_type)

        self.assertEqual(result, expected_result)

    def test_to_bytes_no_value(self):
        operation_code = ProtocolOperationCode.GET  # b'\x01'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 8  # b'\x08'
        key = "test_key"  # b'test_key'
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 0  # b'\x00'
        value = ""  # b''

        expected_result = b"\x01\x02\x08test_key\x02\x00"
        result = to_bytes(
            operation_code, key, key_data_type, value=None, value_data_type=-1
        )
        self.assertEqual(result, expected_result)

    def test_to_bytes_empty_key(self):
        operation_code = ProtocolOperationCode.CONTAINS  # b'\x02'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 0  # b'\x00'
        key = ""  # b''
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 0  # b'\x00'
        value = ""  # b''

        operation_code = ProtocolOperationCode.CONTAINS
        key = ""
        key_data_type = ProtocolDataType.STRING
        expected_result = b"\x02\x02\x00\x02\x00"
        result = to_bytes(operation_code, key, key_data_type)
        self.assertEqual(result, expected_result)

    def test_to_bytes_float_value(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 8  # b'\x08'
        key = "test_key"  # b'test_key'
        value_data_type = ProtocolDataType.FLOAT  # b'\x00'
        len_value = 4  # b'\x04'
        value = 3.14  # b'\x40\x48\xf5\xc3'

        expected_result = b"\x00\x02\x08test_key\x00\x04\x40\x48\xf5\xc3"
        result = to_bytes(operation_code, key, key_data_type, value, value_data_type)
        self.assertEqual(result, expected_result)

    def test_to_bytes_int_value(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 8  # b'\x08'
        key = "test_key"  # b'test_key'
        value_data_type = ProtocolDataType.INT  # b'\x01'
        len_value = 4  # b'\x04'
        value = 42  # b'\x00\x00\x00\x2a'

        expected_result = b"\x00\x02\x08test_key\x01\x04\x00\x00\x00\x2a"
        result = to_bytes(operation_code, key, key_data_type, value, value_data_type)
        self.assertEqual(result, expected_result)

    def test_to_bytes_no_key(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 0  # b'\x00'
        key = ""  # b''
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 10  # b'\x0a'
        value = "test_value"  # b'test_value'
        expected_result = b"\x00\x02\x00\x02\x0atest_value"
        result = to_bytes(operation_code, key, key_data_type, value, value_data_type)
        self.assertEqual(result, expected_result)

    def test_to_bytes_no_key_or_value(self):
        operation_code = ProtocolOperationCode.CONTAINS  # b'\x02'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 0  # b'\x00'
        key = ""  # b''
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 0  # b'\x00'
        value = "test_value"  # b'test_value'

        expected_result = b"\x02\x02\x00\x02\x00"
        result = to_bytes(operation_code, key, key_data_type)
        self.assertEqual(result, expected_result)


class TestFromBytes(unittest.TestCase):
    def test_from_bytes_put_with_value(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 8  # b'\x08'
        key = "test_key"  # b'test_key'
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 10  # b'\x0a'
        value = "test_value"  # b'test_value'

        data = b"\x00\x02\x08test_key\x02\x0btest_value"
        expected_result = (
            ProtocolOperationCode.PUT,
            "test_key",
            ProtocolDataType.STRING,
            "test_value",
            ProtocolDataType.STRING,
        )
        result = from_bytes(data)
        self.assertEqual(result, expected_result)

    def test_from_bytes_put_no_value(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 8  # b'\x08'
        key = "test_key"  # b'test_key'
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 0  # b'\x00'
        value = ""  # b''

        data = b"\x00\x02\x08test_key\x02\x00"
        expected_result = (
            ProtocolOperationCode.PUT,
            "test_key",
            ProtocolDataType.STRING,
            None,
            -1,
        )
        result = from_bytes(data)
        self.assertEqual(result, expected_result)

    def test_from_bytes_get(self):
        operation_code = ProtocolOperationCode.GET  # b'\x01'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 8  # b'\x08'
        key = "test_key"  # b'test_key'
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 0  # b'\x00'
        value = ""  # b''

        data = b"\x01\x02\x08test_key\x02\x00"
        expected_result = (
            ProtocolOperationCode.GET,
            "test_key",
            ProtocolDataType.STRING,
            None,
            -1,
        )
        result = from_bytes(data)
        self.assertEqual(result, expected_result)

    def test_from_bytes_contains(self):
        operation_code = ProtocolOperationCode.CONTAINS  # b'\x02'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 0  # b'\x00'
        key = ""  # b''
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 0  # b'\x00'
        value = ""  # b''

        data = b"\x02\x02\x00\x02\x00"
        expected_result = (
            ProtocolOperationCode.CONTAINS,
            "",
            ProtocolDataType.STRING,
            None,
            -1,
        )
        result = from_bytes(data)
        self.assertEqual(result, expected_result)

    def test_from_bytes_put_with_float_value(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 8  # b'\x08'
        key = "test_key"  # b'test_key'
        value_data_type = ProtocolDataType.FLOAT  # b'\x00'
        len_value = 4  # b'\x04'
        value = 3.14  # b'\x40\x48\xf5\xc3'

        data = b"\x00\x02\x08test_key\x00\x04\x40\x48\xf5\xc3"
        expected_result = (
            ProtocolOperationCode.PUT,
            "test_key",
            ProtocolDataType.STRING,
            3.14,
            ProtocolDataType.FLOAT,
        )
        result = from_bytes(data)
        # Floating point error requires an almost equal here
        for x, y in zip(result, expected_result):
            self.assertAlmostEqual(x, y, places=6)

    def test_from_bytes_put_with_int_value(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.STRING  # b'\x02'
        len_key = 8  # b'\x08'
        key = "test_key"  # b'test_key'
        value_data_type = ProtocolDataType.INT  # b'\x01'
        len_value = 4  # b'\x04'
        value = 42  # b'\x00\x00\x00\2a'

        data = b"\x00\x02\x08test_key\x01\x04\x00\x00\x00\x2a"
        expected_result = (
            ProtocolOperationCode.PUT,
            "test_key",
            ProtocolDataType.STRING,
            42,
            ProtocolDataType.INT,
        )
        result = from_bytes(data)
        self.assertEqual(result, expected_result)

    def test_from_bytes_no_key_with_value(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.FLOAT  # b'\x00'
        len_key = 0  # b'\x00'
        key = -1  # b'' placeholder, above should cause an error
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 10  # b'\x0a'
        value = "test_value"  # b'test_value'

        data = b"\x00\x00\x00\x02\x0atest_value"
        with self.assertRaises(ValueError):
            result = from_bytes(data)

    def test_from_bytes_put_no_key_or_value(self):
        operation_code = ProtocolOperationCode.PUT  # b'\x00'
        key_data_type = ProtocolDataType.INT  # b'\x01'
        len_key = 0  # b'\x00'
        key = -1  # placeholder, above should cause an error
        value_data_type = ProtocolDataType.STRING  # b'\x02'
        len_value = 0  # b'\x00'
        value = ""  # b''

        data = b"\x00\x01\x00\x02\x00"
        with self.assertRaises(ValueError):
            result = from_bytes(data)


if __name__ == "__main__":
    unittest.main()
