Intel® Neural Compressor v3.0 Release

- Highlights
- Features
- Improvements
- Examples
- Bug Fixes
- Documentations
- External Contributions
- Validated Configurations

**Highlights**
 - FP8 quantization support on [Intel® Gaudi® AI accelerator](https://habana.ai/products/gaudi2/) 
 - INT4 weight-only quantization and INT4 model loading support on [Intel® Gaudi® AI accelerator](https://habana.ai/products/gaudi2/) 
 - Framework extension API (3.x API) for quantization, mixed-precision and benchmarking  
 - Accuracy-aware FP16 mixed precision support on [Intel® Xeon® 6 Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html)    
 - Performance optimizations and usability improvements on client-side quantization    

**Features**
 - [Quantization] Support INC and Huggingface model loading on framework extension API for Pytorch ([0eced1](https://github.com/intel/neural-compressor/commit/0eced1478c6796a5e2dcb254a65bbc96af4d1b8b), [bacc16](https://github.com/intel/neural-compressor/commit/bacc164df2c2080cb6b1a6250f745824bbca5a7b))
 - [Quantization] Support Weight-only Quantization on framework extension API for Pytorch ([4a4509](https://github.com/intel/neural-compressor/commit/4a45093c1418f34da2660a54052a2ff5c2b4edff), [1a4509](https://github.com/intel/neural-compressor/commit/1a4509060714559bdbc60524012997900c464d02), [a3a065](https://github.com/intel/neural-compressor/commit/a3a06508fa951f9b9dcd3786214f546c796c32e7), [1386ac](https://github.com/intel/neural-compressor/commit/1386ac5ec7be40608dfac082d2275307b8e4d14e), [a0dee9](https://github.com/intel/neural-compressor/commit/a0dee94dab0920ba30de049e871b19a72ddb8996), [503d9e](https://github.com/intel/neural-compressor/commit/503d9ef4136023f1952e397a2ab0f7f476040901), [84d705](https://github.com/intel/neural-compressor/commit/84d7055b3998724aecd7ca7e43ea653d0d0f4612), [099b7a](https://github.com/intel/neural-compressor/commit/099b7a4446d9c21af2066518ccc87ecaa717e08e), [e3c736](https://github.com/intel/neural-compressor/commit/e3c736fd910690faf08bf4609cc3b65529d79252), [e87c95](https://github.com/intel/neural-compressor/commit/e87c95f25d3fe0e286e832857974ce36d43b2f96), [2694bb](https://github.com/intel/neural-compressor/commit/2694bbf81622a936f5ef3c271901dea097af2474), [ec49a2](https://github.com/intel/neural-compressor/commit/ec49a29cafa92593d82635562ec200741fd4083c), [e7b4b6](https://github.com/intel/neural-compressor/commit/e7b4b648665df4d016d170cdf2f3f69e6f9c185f), [a9bf79](https://github.com/intel/neural-compressor/commit/a9bf79c63fbcd970cccc00d1db85e424fe286b27), [ac717b](https://github.com/intel/neural-compressor/commit/ac717bc4b6a1a1e82db218d7648121f157814fad), [915018](https://github.com/intel/neural-compressor/commit/9150181bb2ab71201fbdb052fbcaa2aba18a090a), [8447d7](https://github.com/intel/neural-compressor/commit/8447d7097fa33231b8a6e4a9e26e526d191787de), [dc9328](https://github.com/intel/neural-compressor/commit/dc9328c09b243d7df3bccc0a35a8a12feaabb40a))
 - [Quantization] Support static and dynamic quantization in PT2E path ([43c358](https://github.com/intel/neural-compressor/commit/43c3580bdb1c6765bb4902fe721da629518acc74), [30b36b](https://github.com/intel/neural-compressor/commit/30b36b83a195c6ea350692c7ac0bfec1b52ee419), [1f58f0](https://github.com/intel/neural-compressor/commit/1f58f024d812b6c1f7f3430b62e61051599cd1b2), [02958d](https://github.com/intel/neural-compressor/commit/02958dd4a81251be26980a712cbb258d55edba67))
 - [Quantization] Support SmoothQuant and static quantization in IPEX path with framework extension API ([72fbce](https://github.com/intel/neural-compressor/commit/72fbce4b34f29c2b6fe0d41a76c4d65edb08719a), [eaa3a5](https://github.com/intel/neural-compressor/commit/eaa3a580c8a9f27268d3c27e551054dd5053f01c), [95e67e](https://github.com/intel/neural-compressor/commit/95e67eac624285d304487b654330d660b169cfb1), [855c10](https://github.com/intel/neural-compressor/commit/855c10ca37d01bd371a4b9dcd953ce735f9bdea6), [9c6102](https://github.com/intel/neural-compressor/commit/9c6102b351c45394357e0163470e0e997cb99d0e), [5dafe5](https://github.com/intel/neural-compressor/commit/5dafe5fd6584ca695f05b61c3dd84c2923c83cbd), [a5e5f5](https://github.com/intel/neural-compressor/commit/a5e5f5f64855b85e2a374c8b808b317448318113), [191383](https://github.com/intel/neural-compressor/commit/191383ebd95c1fbb77e626887ca6d808a454543c), [776645](https://github.com/intel/neural-compressor/commit/7766454d9a984257016ddad5d3a61de648f0bd35))
 - [Quantization] Support Post Training Quantization on framework extension API for Tensorflow ([e22c61](https://github.com/intel/neural-compressor/commit/e22c61ede2942f7f1ba1cf9e480491371184bb32), [f21afb](https://github.com/intel/neural-compressor/commit/f21afbbdd18cd61627fc02e5b22ca242402bcfbf), [3882e9](https://github.com/intel/neural-compressor/commit/3882e9cc4b356a081843455f3244d7f0e013f888), [2627d3](https://github.com/intel/neural-compressor/commit/2627d33b9ff900697184972575969ecc55da8923))
 - [Quantization] Support Post Training Quantization on Keras3 ([f67e86](https://github.com/intel/neural-compressor/commit/f67e8613c409563f016c77e05a1acb969790cfc6), [047560](https://github.com/intel/neural-compressor/commit/047560fcf6a2e5812d33e579e047a3c8767e4a9a))
 - [Quantization] Support Weight-only Quantization on Gaudi2 ([4b9b44](https://github.com/intel/neural-compressor/commit/4b9b447aa0872a8edc26fd59a349c195cf208a97), [14868c](https://github.com/intel/neural-compressor/commit/14868c0900a1f91fe39f138c67156ad66c16b20f), [0a3d4b](https://github.com/intel/neural-compressor/commit/0a3d4bd43f69c29e2f8a3b07ac13036e41c6579c))
 - [Quantization] Improve performance and usability of quantization procedure on client side ([16a7b1](https://github.com/intel/neural-compressor/commit/16a7b11508c008d4d4180a0fe0e31c75b8e5d662))
 - [Quantization] Support auto-device detection on framework extension API for Pytorch ([368ba5](https://github.com/intel/neural-compressor/commit/368ba5293ab2936c685d67db6f8423a27a62f7e1), [4b9b44](https://github.com/intel/neural-compressor/commit/4b9b447aa0872a8edc26fd59a349c195cf208a97), [e81a2d](https://github.com/intel/neural-compressor/commit/e81a2dd901dd1b93291555722c6d96901940be06), [0a3d4b](https://github.com/intel/neural-compressor/commit/0a3d4bd43f69c29e2f8a3b07ac13036e41c6579c), [534300](https://github.com/intel/neural-compressor/commit/53430092e7f9b46ed78afccb4b9610c9032bf57f), [2a86ae](https://github.com/intel/neural-compressor/commit/2a86aeafc754ca3b7495138381efb8faa9397fdf))
 - [Quantization] Support Microscaling(MX) Quant for PyTorch ([4a24a6](https://github.com/intel/neural-compressor/commit/4a24a6a39218a3d186900a72a7e2e96ad539f4f4), [455f1e](https://github.com/intel/neural-compressor/commit/455f1e1f0f0284e87b46d257b6d126ca76fe1748))
 - [Quantization] Support FP8 cast Weight-only Quantization ([57ed61](https://github.com/intel/neural-compressor/commit/57ed6138453246141a2128b600588df0b4d5d440))
 - [Quantization] Enable cross-devices Half-Quadratic Quantization(HQQ) for LLMs support ([db6164](https://github.com/intel/neural-compressor/commit/db6164a25da5bf8ef8a7ba082a25d7bb4565b656), [07f940](https://github.com/intel/neural-compressor/commit/07f940c7f00ab0a5f6b3d7d9cb6b934e69e44a98))
 - [Mixed-Precision] Support FP16 mixed-precision on framework extension autotune API for Pytorch ([2e1cdc](https://github.com/intel/neural-compressor/commit/2e1cdc5be61458be186d0e6f2035b4287b223cf3))
 - [Mixed-Precision] Support mixed `INT8` with `FP16` in PT2E path ([fa961e](https://github.com/intel/neural-compressor/commit/fa961e1d0bbe371182d6da6d210d0b6a7693cce2))
 - [AutoTune] Support accuracy-aware tuning on framework extension API ([7b8aec](https://github.com/intel/neural-compressor/commit/7b8aec00d0c09bd499076457b68903229e09b803), [5a0374](https://github.com/intel/neural-compressor/commit/5a0374e7db23cac209af78f1ace9b38d23bebbb0), [a4675c](https://github.com/intel/neural-compressor/commit/a4675c7490f66ab2c75912dd69f1d79368f69858), [3a254e](https://github.com/intel/neural-compressor/commit/3a254e99c0a361c0179b4176a256c69e46681352), [ac47d9](https://github.com/intel/neural-compressor/commit/ac47d9b97b597f809ab56f9f6cb1a86951e2e334), [b8d98e](https://github.com/intel/neural-compressor/commit/b8d98ebaddcf1c7ece1def04ba4d55b7e92593ee), [fb6142](https://github.com/intel/neural-compressor/commit/fb61428228bcdf9a18b02e5963c4df7a60c9a54b), [fa8e66](https://github.com/intel/neural-compressor/commit/fa8e66a1d95b52c8ebdea21f2dc60db0fdfedd6a), [d22df5](https://github.com/intel/neural-compressor/commit/d22df5364ba6d7c98fea8545a9a9e49e2ce5ebb0), [09eb5d](https://github.com/intel/neural-compressor/commit/09eb5ddd3c0eb2dae198837cbae76ca5bb4e90c8), [c6a8fa](https://github.com/intel/neural-compressor/commit/c6a8fa1606a4aea34e62af0f106ab05cdccacab6))
 - [Benchmarking] Implement `incbench` command for ease-of-use benchmark ([2fc725](https://github.com/intel/neural-compressor/commit/2fc72555c987dc7bce8476b389720e1a29159a43))

**Improvements**
 - [Quantization] Support auto_host2device on RTN and GPTQ([f75ff4](https://github.com/intel/neural-compressor/commit/f75ff4082bc7a22d9367d3e91a3ea2c7aaec2bd2))
 - [Quantization] Support `transformers.Conv1D` WOQ quantization ([b6237c](https://github.com/intel/neural-compressor/commit/b6237cf4d4c8e86fe373cf48ffe5a6588ef537ca))
 - [Quantization] support quant_lm_head argument in all WOQ configs ([4ae2e8](https://github.com/intel/neural-compressor/commit/4ae2e87d2f98eb34c2e523a76ffa6ff77bf767e1))
 - [Quantization] Update fp4_e2m1 mapping list to fit neural_speed and qbits inference ([5fde50](https://github.com/intel/neural-compressor/commit/5fde50f2c0476dbc08d59481b742515f5a210de1))
 - [Common] Add common logger to the quantization process ([1cb844](https://github.com/intel/neural-compressor/commit/1cb844b3c0b581f670fef16aa87fef2a85e6122b), [482f87](https://github.com/intel/neural-compressor/commit/482f87c6161581f9f8ff09804b6c430553cf59a9), [83bc77](https://github.com/intel/neural-compressor/commit/83bc779a4e97d8886383025d324d8379f70cc8b7), [f50baf](https://github.com/intel/neural-compressor/commit/f50baf2e9107e29d96e267fe115dc488f96db6f0))
 - [Common] Enhance the `set_local` for operator type ([a58638](https://github.com/intel/neural-compressor/commit/a58638c1298fdff808742d1625196153d24f5c9c))
 - [Common] Port more helper classes from 2.x ([3b150d](https://github.com/intel/neural-compressor/commit/3b150d61313ca6ca19bc38ec9f608900b8355519))
 - [Common] Refine base config for 3.x API ([efea08](https://github.com/intel/neural-compressor/commit/efea089e27613690c32d6f1745731a28ca90bf65))
 - [Export] Migrate export feature to 2.x and 3.x from deprecated ([794b27](https://github.com/intel/neural-compressor/commit/794b2762c0bb2f076973e1fca5fdecd23efec774))

**Examples**
 - Add CV and LLM examples for PT2E quantization path ([b401b0](https://github.com/intel/neural-compressor/commit/b401b02db2cc7d7f4f8412a815fa435e66e330a0))
 - Add Recommendation System examples for IPEX path ([e470f6](https://github.com/intel/neural-compressor/commit/e470f6cdfbbad32fcf17be56903e649a05059780))
 - Add new framework extension API TensorFlow examples ([922b24](https://github.com/intel/neural-compressor/commit/922b2471e617cc4c56376866e991302d0beb0640))
 - Add Microscaling(MX) Quant PyTorch examples ([6733da](https://github.com/intel/neural-compressor/commit/6733dabc4d48a6625e184e4a29a754949f415097))
 - Add SmoothQuant LLM examples for new framework extension API for PyTorch ([137fa3](https://github.com/intel/neural-compressor/commit/137fa3add2d8a0688dd0e76bd15e347b588d56a8))
 - Add GPTQ/RTN framework extension API example for PyTorch ([813d93](https://github.com/intel/neural-compressor/commit/813d93051ab16b6bbac11bdf5986929330876e30))
 - Add double quant example ([ccd0c9](https://github.com/intel/neural-compressor/commit/ccd0c9e6c112d84979504177b9390270b3d71b69))

**Bug Fixes**
 - Remove Gelu Fusion for TensorFlow New API ([5592ac](https://github.com/intel/neural-compressor/commit/5592acc60562b7fccb308af0eaaba9cad53004a5))
 - Fix GPTQ layer match issue ([90fb43](https://github.com/intel/neural-compressor/commit/90fb43135397a035968b5334eba21931c18a83c0))
 - Fix static quant regression issue on IPEX path ([70a1d5](https://github.com/intel/neural-compressor/commit/70a1d501fdfee16a10e34385bca9f15eba4366b4))
 - Fix config expansion with empty options ([6b2738](https://github.com/intel/neural-compressor/commit/6b2738390dfdab543de1ccd9242fe541c78b6a2e))
 - Fix act_observer for IPEX SmoothQuant and static quantization ([263450](https://github.com/intel/neural-compressor/commit/2634501690f2396865011c2f79c0b8adba36cb07))
 - Set automatic return_dict=False for GraphTrace ([53e7df](https://github.com/intel/neural-compressor/commit/53e7dfe57ef4ad1754f37343b3ad3850b64ae4f4))
 - Fix WOQ Linear pack slow issue ([da1ada](https://github.com/intel/neural-compressor/commit/da1ada236eb867b69c663c58904e0a21ad9bcb88), [daa143](https://github.com/intel/neural-compressor/commit/daa1431b200f92ab9684a2c78e15602cb23d7c07))
 - Fix dtype of unpacked tensor ([29fdec](https://github.com/intel/neural-compressor/commit/29fdecbbb44ceb8d19c12809af90dc23063becfc))
 - Fix WeightOnlyLinear bits type when dtype="intx" ([19ff13](https://github.com/intel/neural-compressor/commit/19ff13e8a8963744349e46013ef522fcb3e8c3d8))
 - Fix several issues for SmoothQuant and static quantization ([7120dd](https://github.com/intel/neural-compressor/commit/7120dd4909599b228692415732688b3d5e77206d))
 - Fix 3.x IPEX examples failed with evaluate ([e82674](https://github.com/intel/neural-compressor/commit/e82674a75de564a632cea639db25fbe41fec100a))
 - Fix HQQ issue for group size of -1 ([8dac9f](https://github.com/intel/neural-compressor/commit/8dac9f2c3d3f8411f27a2e327f3dbbc7c8de0829))
 - Fix bug in gptq g_idx ([4f893c](https://github.com/intel/neural-compressor/commit/4f893ca9e4c44d12ea028e00a4881b5154ee54a8))
 - Fix tune_cfg issue for 3.x static quant ([ba1650](https://github.com/intel/neural-compressor/commit/ba165047dbcf4671cf20e9c1d031577dade94348))
 - Add non-str `op_name` match workaround for IPEX ([911ccd](https://github.com/intel/neural-compressor/commit/911ccd3a94b124e2287780a2ca219eaa01dc21d9))
 - Fix opt gptq double quant example config ([62aa85](https://github.com/intel/neural-compressor/commit/62aa85df23ce3f5db353ce9a4bfb8cd88395c376))
 - Fix gptq accuracy issue in framework extension API example ([c701ea](https://github.com/intel/neural-compressor/commit/c701eaff7d69c46a57172b0547bfe2fc05164a0c))
 - Fix bf16 symbolic_trace bug ([3fe2fd](https://github.com/intel/neural-compressor/commit/3fe2fd9aadda4991552d65fef09a75ba5127b5db))

**Documentations**
 - Add new architecture diagram ([2c3556](https://github.com/intel/neural-compressor/commit/2c3556d441de2f0963167db71ecdee7353bd76bb))
 - Add new workflow diagram ([96538c](https://github.com/intel/neural-compressor/commit/96538c56fea8a42c3e487b4682c346e4832e3e97))
 - Add documents for new framework extension API for PyTorch ([ecffc2](https://github.com/intel/neural-compressor/commit/ecffc2eb29ada100d2b60574258d8a1b6548e449))
 - Add documents for new framework extension API for TensorFlow ([4dbf71](https://github.com/intel/neural-compressor/commit/4dbf71e412a370f09809db89db27a0b7c7b56d14))
 - Add documents for autotune API ([853dc7](https://github.com/intel/neural-compressor/commit/853dc71eee292e93e38f91683ec8229eb14c25da), [de3e94](https://github.com/intel/neural-compressor/commit/de3e94f6d15f74bb3081366dd1c045d006adfa00))
 - Add docstring for `common` module ([28578b](https://github.com/intel/neural-compressor/commit/28578b96bf6217fa2b79699838e5a4af30843de4))

**External Contributions**
 - Update the Gaudi container example in the README ([cc763f](https://github.com/intel/neural-compressor/commit/cc763f5134f5f84b3020a8ea1bee409a60d15218))

**Validated Configurations** 
 - Centos 8.4 & Ubuntu 22.04 & Win 11 & MacOS Ventura 13.5 
 - Python 3.8, 3.9, 3.10, 3.11 
 - PyTorch/IPEX 2.1, 2.2, 2.3 
 - TensorFlow 2.14, 2.15, 2.16 
 - ITEX 2.13.0, 2.14.0, 2.15.0 
 - ONNX Runtime 1.16, 1.17, 1.18