import os
import shutil
import subprocess
import sys

from argparse import ArgumentParser, REMAINDER

class Launcher():
    def parse_args():
        """
        Helper function parsing the command line options
        @retval ArgumentParser
        """
        parser = ArgumentParser(description="command-launch a Python script with quantization auto-enabled")

        parser.add_argument("-o", "--opt", type=str, default="",
                            help="optimization feature to enable")

        parser.add_argument("-a", "--approach", type=str, default="auto",

                            help="quantization approach (strategy)")

        parser.add_argument('--config', type=str, default="",
                            help='quantization configuration file path')

        parser.add_argument('-b', '--bench', default=False, action='store_true',
                            help='conduct auto_quant benchmark instead of enable')

        parser.add_argument('-e', '--enable', default=False, action='store_true',
                            help='only do enable, not overwrite or run program')

        # positional
        parser.add_argument("script", type=str,
                            help="The full path to the script to be launched. "
                                 "followed by all the arguments for the script")

        # script args
        parser.add_argument('script_args', nargs=REMAINDER)
        return parser.parse_args()
        
    def execute(
        args,
        use_modular=False,
        modular_pattern={},
    ):
        # copy user entry script (main.py -> main_optimized.py)
        script_copied = args.script[:-3] + "_optimized.py"
        shutil.copy(args.script, script_copied)

        if not args.bench: # "enable and run" or "only enable"
            # optimize on copied script with Neural Coder
            from neural_coder import enable
            if args.opt == "":
                if args.approach == "static":
                    args.opt = "pytorch_inc_static_quant_fx"
                if args.approach == "static_ipex":
                    args.opt = "pytorch_inc_static_quant_ipex"
                if args.approach == "dynamic":
                    args.opt = "pytorch_inc_dynamic_quant"
                if args.approach == "auto":
                    args.opt = "inc_auto"
                features = [args.opt]
            else:
                features = args.opt.split(",")
            
            # modular design
            modular_item = ""
            if use_modular:
                modular_item = modular_pattern[args.opt]
            
            # execute optimization enabling
            enable(
                code=script_copied,
                features=features,
                overwrite=True,
                use_modular=use_modular,
                modular_item=modular_item,
            )

            if not args.enable: # enable and run
                # execute on copied script, which has already been optimized
                cmd = []

                cmd.append(sys.executable) # "/xxx/xxx/python"
                cmd.append("-u")
                cmd.append(script_copied)
                cmd.extend(args.script_args)

                cmd = " ".join(cmd) # list convert to string

                process = subprocess.Popen(cmd, env=os.environ, shell=True)  # nosec
                process.wait()
        else: # auto_quant
            from neural_coder import auto_quant
            auto_quant(
                code=script_copied,
                args=' '.join(args.script_args), # convert list of strings to a single string
            )
