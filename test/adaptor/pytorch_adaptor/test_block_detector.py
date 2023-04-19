def test_block_detector():
    from neural_compressor.adaptor.torch_utils.block_detector import TransformerModelBlockDetector, BLOCK_PATTERNS
    from transformers import BertTokenizer, BertModel

    model = BertModel.from_pretrained("bert-base-uncased")
    detector = TransformerModelBlockDetector(model, BLOCK_PATTERNS)
    result = detector.detect_block()
    assert len(result['attention_blocks']), 12
    assert len(result['ffn_blocks']), 12

    found_attention_op = False
    found_dense_op = False
    for block in ['attention_blocks']:
        for op in block:
            if 'dense' in op:
                found_dense_op = True
                break

    for block in ['ffn_blocks']:
        for op in block:
            if 'attention' in op:
                found_attention_op = True
                break
    assert not found_attention_op
    assert not found_dense_op
    
    

def test_block_wise_tuining_stock_pt():
    from neural_compressor.quantization import fit
    from neural_compressor.config import PostTrainingQuantConfig
    from neural_compressor.data import Datasets, DATALOADERS

    from transformers import BertTokenizer, BertModel
    model_name = "bert-base-uncased"
    model = BertModel.from_pretrained(model_name)
    model.eval()
    # dataset and dataloader
    class DummyNLPDataloader(object):
        def __init__(self, model_name):
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.sequence_a = "intel-extension-for-transformers is based in SH"
            self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
            self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b, return_tensors='pt')
            self.batch_size = 1

        def __iter__(self):
            yield self.encoded_dict

        def __next__(self):
            return self.encoded_dict

    dataloader = DummyNLPDataloader(model_name)
    # tuning and accuracy criterion
    conf = PostTrainingQuantConfig()
    q_model = fit(model=model, conf=conf, calib_dataloader= dataloader, eval_func=lambda model : 1)
    assert q_model is not None