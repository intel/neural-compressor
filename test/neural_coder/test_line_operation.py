import unittest

from neural_coder.utils import line_operation

class TestLineOperation(unittest.TestCase):
    def test_get_line_indent_level(self):
        f = line_operation.get_line_indent_level
        self.assertEqual(f("    model(input)"), 4)
        self.assertEqual(f("        model(input)"), 8)
        self.assertEqual(f("model(input)"), 0)
        self.assertEqual(f("# model(input)"), 0)

    def test_single_line_comment_or_empty_line_detection(self):
        f = line_operation.single_line_comment_or_empty_line_detection
        self.assertEqual(f("# test"), True)
        self.assertEqual(f("test  # test"), False)
        self.assertEqual(f("    "), True)
        self.assertEqual(f("    test"), False)
        self.assertEqual(f('"""test"""'), True)

    def test_is_eval_func_model_name(self):
        f = line_operation.is_eval_func_model_name
        self.assertEqual(f("model", "model(input)"), True)
        self.assertEqual(f("model", "model()"), True)
        self.assertEqual(f("model", "# model(input)"), False)
        self.assertEqual(f("model", "test # model(input)"), False)
        self.assertEqual(f("model", "output = model(input)"), True)
        self.assertEqual(f("model", "model = Net()"), False)

    def test_get_line_lhs(self):
        f = line_operation.get_line_lhs
        self.assertEqual(f("output = model(input)"), "output")
        self.assertEqual(f("output=model(input)"), "output")
        self.assertEqual(f("test = num"), "test")

    def test_of_definition_format(self):
        f = line_operation.of_definition_format
        self.assertEqual(f("output = model(input)"), (True, "output", "model"))
        self.assertEqual(f("output=model(input)"), (True, "output", "model"))
        self.assertEqual(f("model = Net()"), (True, "model", "Net"))
        self.assertEqual(f("model = Net"), (False, "", ""))

if __name__ == '__main__':
    unittest.main()
