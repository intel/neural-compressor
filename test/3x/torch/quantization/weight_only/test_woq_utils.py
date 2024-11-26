import pytest
import torch
import torch.nn as nn

from neural_compressor.torch.algorithms.weight_only.utility import (
    CapturedDataloader,
    forward_wrapper,
    get_example_input,
    model_forward,
    move_input_to_device,
)


class TestUtility:

    def test_move_input_to_device(self):
        # Test when input is a dictionary
        input_dict = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
        moved_input_dict = move_input_to_device(input_dict)
        assert all(val.device.type == "cpu" for val in moved_input_dict.values())

        # Test when input is a list
        input_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        moved_input_list = move_input_to_device(input_list)
        assert all(val.device.type == "cpu" for val in moved_input_list)

        # Test when input is a tensor
        input_tensor = torch.tensor([1, 2, 3])
        moved_input_tensor = move_input_to_device(input_tensor)
        assert all(val.device.type == "cpu" for val in moved_input_tensor)

        # Test when input is a string
        input_str = "string"
        moved_input_string = move_input_to_device(input_str)
        assert moved_input_string == input_str

    def test_forward_wrapper(self):

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                return x * 2

        model = Model()
        input_tensor = torch.tensor([1, 2, 3])

        # Test the behavior of forward_wrapper function
        output = forward_wrapper(model, input_tensor)
        assert torch.all(output == input_tensor * 2)

        # Test dict
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, **kwargs):
                return kwargs["input_ids"] * 2

        model = Model()
        input_dict = {"input_ids": torch.tensor([1, 2, 3])}

        # Test the behavior of forward_wrapper function with dict input
        output = forward_wrapper(model, input_dict)
        assert torch.all(output == input_dict["input_ids"] * 2)

        class MockModel:
            def __call__(self, x, y):
                return x + y

        model = MockModel()
        input_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

        # Test the behavior of forward_wrapper function with list input
        output = forward_wrapper(model, input_list)
        assert torch.all(output == torch.tensor([5, 7, 9]))

        class MockModel:
            def __call__(self, x):
                raise ValueError("Mock model exception")

        model = MockModel()

        # Test the behavior of forward_wrapper function when an exception is raised
        with pytest.raises(ValueError):
            forward_wrapper(model, input_list)

    def test_model_forward(self):
        # Create a mock data loader and model
        class MockDataLoader:
            def __iter__(self):
                return iter([(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))])

        class MockModel:
            def __call__(self, x):
                return x * 2

        dataloader = MockDataLoader()
        model = MockModel()

        # Test the behavior of model_forward function
        model_forward(model, dataloader, 1, torch.device("cpu"))

        # test dataloader without label
        class MockDataLoader:
            def __iter__(self):
                return iter(torch.tensor([1, 2, 3]))

        class MockModel:
            def __call__(self, x):
                return x * 2

        dataloader = MockDataLoader()
        model = MockModel()

        # Test the behavior of model_forward function
        model_forward(model, dataloader, 1, torch.device("cpu"))

    def test_get_example_input(self):
        # Create a mock data loader with multiple data samples
        class MockDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        # Test case 1: When the dataloader contains label
        dataloader_multiple_batches = MockDataLoader(
            [
                (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
                (torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])),
                (torch.tensor([13, 14, 15]), torch.tensor([16, 17, 18])),
            ]
        )
        example_inp_multiple_batches = get_example_input(dataloader_multiple_batches, i=1)
        assert torch.all(example_inp_multiple_batches == torch.tensor([7, 8, 9]))
        example_inp_multiple_batches = get_example_input(dataloader_multiple_batches, i=0)
        assert torch.all(example_inp_multiple_batches == torch.tensor([1, 2, 3]))

        # test empty dataloader
        dataloader_multiple_batches = dataloader_multiple_batches = MockDataLoader([])
        example_inp_multiple_batches = get_example_input(dataloader_multiple_batches, i=1)
        assert example_inp_multiple_batches is None

        # Test case 2: When the dataloader contains a single input
        dataloader_single_batch = MockDataLoader([(torch.tensor([1, 2, 3, 4])), torch.tensor([5, 6, 7, 8])])
        example_inp_single_batch = get_example_input(dataloader_single_batch, i=0)
        assert torch.all(example_inp_single_batch == torch.tensor([1, 2, 3, 4]))
        example_inp_single_batch = get_example_input(dataloader_single_batch, i=1)

    #     assert torch.all(example_inp_single_batch == torch.tensor([5, 6, 7, 8]))

    def test_captured_dataloader_iteration(self):
        """Test the iteration behavior of CapturedDataloader."""

        # Test case when args is empty, kwargs contains data
        args_list = [(), (), ()]
        kwargs_list = [{"a": 1}, {"b": 2}, {"c": 3}]
        dataloader = CapturedDataloader(args_list, kwargs_list)
        for i, data in enumerate(dataloader):
            assert data == kwargs_list[i]

        # Test case when kwargs is empty, args contains data
        args_list = [(1,), (2,), (3,)]
        kwargs_list = [{}, {}, {}]
        dataloader = CapturedDataloader(args_list, kwargs_list)
        for i, data in enumerate(dataloader):
            assert data == i + 1

        # Test case when both args and kwargs are present
        args_list = [(1, 2), (2, 3), (3, 4)]
        kwargs_list = [{}, {}, {}]
        dataloader = CapturedDataloader(args_list, kwargs_list)
        for i, data in enumerate(dataloader):
            assert data == args_list[i]

        # Test case when both args and kwargs are present
        args_list = [(1,), (2,), (3,)]
        kwargs_list = [{"a": 1}, {"b": 2}, {"c": 3}]
        dataloader = CapturedDataloader(args_list, kwargs_list)
        expected_result = [((1,), {"a": 1}), ((2,), {"b": 2}), ((3,), {"c": 3})]
        for i, data in enumerate(dataloader):
            assert data == expected_result[i]
