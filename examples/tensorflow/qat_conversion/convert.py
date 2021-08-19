from lpot.experimental import ModelConversion, common
conversion = ModelConversion()
conversion.source = 'QAT'
conversion.destination = 'default'
conversion.model = common.Model('../qat/trained_qat_model')
q_model = conversion()
q_model.save('quantized_model')
