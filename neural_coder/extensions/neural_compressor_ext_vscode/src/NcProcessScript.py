import os
import sys
import subprocess
libs = ['neural-compressor']

try:
    from neural_coder import enable
    from neural_coder import auto_quant

except ModuleNotFoundError:
    for lib in libs:
        os.system(f'{sys.argv[6]} -m pip install -U {lib}')
    from neural_coder import enable
    from neural_coder import auto_quant

if (sys.argv[4] == "normal"):
    enable(code=sys.argv[1], features=[sys.argv[2]], overwrite=True)
    logResult = enable(code=sys.argv[1], features=[
                       sys.argv[2]], save_patch_path=sys.argv[5])

elif (sys.argv[4] == "genLog"):
    if (sys.argv[2] == ""):
        # codeResult have 3 params: perfomance, mode, path
        codeResult = enable(
            code=sys.argv[1], features=[], run_bench=True, args=sys.argv[3])

    else:
        codeResult = enable(code=sys.argv[1], features=[
                            sys.argv[2]], run_bench=True, args=sys.argv[3])
        logResult = enable(code=sys.argv[1], features=[
                           sys.argv[2]], args=sys.argv[3], save_patch_path=sys.argv[5])

    # print fps
    with open(codeResult[2] + '/bench.log', 'r') as f:
        logs = f.readlines()
        for log in logs:
            if (log.find('fps') != -1):
                log_line = log
                fps = log_line.split("[")[1].split("]")[0]
        print(fps)

elif (sys.argv[4] == "hardWare"):
    subp = subprocess.Popen("lscpu | grep 'Model name'", shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    subp.wait(2)
    hardware = subp.communicate()[0].replace("Model name:", "").strip()
    print(hardware)
