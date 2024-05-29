import argparse
import time
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6b"
)
parser.add_argument(
    "--trust_remote_code", default=True,
    help="Transformers parameter: use the external repo")
parser.add_argument(
    "--revision", default=None,
    help="Transformers parameter: set the model hub commit number")
parser.add_argument("--quantize", action="store_true")
# dynamic only now
parser.add_argument("--w_dtype", type=str, default="int8", 
                    choices=["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3", "fp6_e3m2", 
                                                "fp6_e2m3", "fp4", "float16", "bfloat12"],
                    help="weight data type")
parser.add_argument("--act_dtype", type=str, default="int8", 
                    choices=["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3", "fp6_e3m2", 
                                                "fp6_e2m3", "fp4", "float16", "bfloat12"],
                    help="input activation data type")
parser.add_argument("--woq", action="store_true")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--performance", action="store_true")
parser.add_argument("--iters", default=100, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--tasks", type=str, default="lambada_openai",
                    help="tasks list for accuracy validation")
parser.add_argument("--peft_model_id", type=str, default=None, help="model_name_or_path of peft model")

args = parser.parse_args()

def get_user_model():
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    if args.peft_model_id is not None:
        from peft import PeftModel
        user_model = PeftModel.from_pretrained(user_model, args.peft_model_id)

    user_model.eval()
    return user_model, tokenizer

user_model, tokenizer = get_user_model()
if args.quantize:
    from neural_compressor.torch.quantization import MXQuantConfig, quantize
    quant_config = MXQuantConfig(w_dtype=args.w_dtype, act_dtype=args.act_dtype, weight_only=args.woq)
    user_model = quantize(model=user_model, quant_config=quant_config)


if args.accuracy:
    user_model.eval()
    from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
    args = LMEvalParser(
        model="hf",
        user_model=user_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        tasks=args.tasks,
        device="cpu",
    )
    results = evaluate(args)
    dumped = json.dumps(results, indent=2)
    if args.save_accuracy_path:
        with open(args.save_accuracy_path, "w") as f:
            f.write(dumped)
    for task_name in args.tasks:
        if task_name == "wikitext":
            acc = results["results"][task_name]["word_perplexity"]
        else:
            acc = results["results"][task_name]["acc"]
    print("Accuracy: %.5f" % acc)
    print('Batch size = %d' % args.batch_size)

if args.performance:
    user_model.eval()
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    import time
    samples = args.iters * args.batch_size
    start = time.time()
    results = evaluate(
        model="hf",
        tokenizer=tokenizer,
        user_model=user_model,
        batch_size=args.batch_size,
        tasks=args.tasks,
        limit=samples,
    )
    end = time.time()
    for task_name in args.tasks:
        if task_name == "wikitext":
            acc = results["results"][task_name]["word_perplexity"]
        else:
            acc = results["results"][task_name]["acc"]
    print("Accuracy: %.5f" % acc)
    print('Throughput: %.3f samples/sec' % (samples / (end - start)))
    print('Latency: %.3f ms' % ((end - start)*1000 / samples))
    print('Batch size = %d' % args.batch_size)
