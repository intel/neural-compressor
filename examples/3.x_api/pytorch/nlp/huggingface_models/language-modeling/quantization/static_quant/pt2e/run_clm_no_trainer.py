import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="facebook/opt-125m"
)
parser.add_argument(
    "--trust_remote_code", default=True,
    help="Transformers parameter: use the external repo")
parser.add_argument(
    "--revision", default=None,
    help="Transformers parameter: set the model hub commit number")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--output_dir", nargs="?", default="")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--approach", type=str, default='static',
                    help="Select from ['dynamic', 'static', 'weight-only']")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--performance", action="store_true")
parser.add_argument("--calib_iters", default=2, type=int,
                    help="For calibration only.")
parser.add_argument("--iters", default=100, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--tasks", default="lambada_openai,hellaswag,winogrande,piqa,wikitext",
                    type=str, help="tasks for accuracy validation")
parser.add_argument("--eval_limits", default=None, type=int,
                    help="Number of samples to evaluate, default is None, which means all samples")
parser.add_argument("--max_num_tokens", default=2048, type=int,
                    help="Max number of tokens")
parser.add_argument("--max_batch_size", default=16, type=int,
                    help="Max batch size")
parser.add_argument("--peft_model_id", type=str, default=None, help="model_name_or_path of peft model")
# =======================================

args = parser.parse_args()


def get_user_model():
    torchscript = False
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.peft_model_id is not None:
        from peft import PeftModel
        user_model = PeftModel.from_pretrained(user_model, args.peft_model_id)

    # to channels last
    user_model = user_model.to(memory_format=torch.channels_last)
    user_model.eval()
    return user_model, tokenizer

user_model, tokenizer = get_user_model()
if args.quantize:
    
    from neural_compressor.torch.quantization import (
            convert,
            get_default_static_config,
            prepare,
        )
    from neural_compressor.torch.export import export
    from torch.export import Dim
    def get_example_inputs(tokenizer):
        text = "Hello, welcome to LLM world."
        encoded_input = tokenizer(text, return_tensors="pt")

        example_inputs = encoded_input
        input_ids = example_inputs["input_ids"]
        input_ids_batch = torch.cat((input_ids, input_ids), dim=0)
        print(f"input_ids_batch shape: {input_ids_batch.shape}")
        tuple_inputs = (input_ids_batch,)
        return tuple_inputs
    # torch._dynamo.config.cache_size_limit = 4 # set limitation if out of memory
    batch = Dim(name="batch_size", max=args.max_batch_size)
    seq_len = Dim(name="seq_len", max=args.max_num_tokens)
    dynamic_shapes = {"input_ids": (batch, seq_len)}
    example_inputs = get_example_inputs(tokenizer)
    exported_model = export(user_model, example_inputs=example_inputs, dynamic_shapes=dynamic_shapes)
    
    quant_config = get_default_static_config()
    # prepare
    prepare_model = prepare(exported_model, quant_config)

    # calibrate
    for i in range(args.calib_iters):
        prepare_model(*example_inputs)
    # convert
    converted_model = convert(prepare_model)
    
    # save
    if args.output_dir:
        converted_model.save(example_inputs=example_inputs, output_dir = args.output_dir)


if args.int8:
    if args.output_dir:
        print("Load int8 model.")
        from neural_compressor.torch.quantization import load
        model_config = user_model.config
        user_model = load(args.output_dir)
        user_model.config = model_config


if args.accuracy:
    from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
    eval_args = LMEvalParser(
        model="hf",
        user_model=user_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        tasks=args.tasks,
        device="cpu",
        limit=args.eval_limits,
    )
    results = evaluate(eval_args)
    for task_name in args.tasks.split(","):
        if task_name == "wikitext":
            acc = results["results"][task_name]["word_perplexity,none"]
        else:
            acc = results["results"][task_name]["acc,none"]
    print("Accuracy: %.5f" % acc)
    print('Batch size = %d' % args.batch_size)

if args.performance:
    batch_size, input_leng = args.batch_size, 512
    example_inputs = torch.ones((batch_size, input_leng), dtype=torch.long)
    # Compile the quantized model and replace the Q/DQ pattern with Q-operator
    from torch._inductor import config

    config.freezing = True
    model_config = user_model.config
    user_model = torch.compile(user_model)
    user_model.config = model_config

    print("Batch size = {:d}".format(batch_size))
    print("The length of input tokens = {:d}".format(input_leng))
    import time

    total_iters = args.iters
    warmup_iters = 5
    with torch.no_grad():
        for i in range(total_iters):
            if i == warmup_iters:
                start = time.time()
            user_model(example_inputs)
        end = time.time()
    latency = (end - start) / ((total_iters - warmup_iters) * args.batch_size)
    throughput = ((total_iters - warmup_iters) * args.batch_size) / (end - start)
    print("Latency: {:.3f} ms".format(latency * 10**3))
    print("Throughput: {:.3f} samples/sec".format(throughput))
