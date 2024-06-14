Intel® Neural Compressor v2.6 Release

- Highlights
- Features
- Improvements
- Examples
- Bug Fixes
- External Contributions
- Validated Configurations

**Highlights**
 - Integrated recent [AutoRound](https://github.com/intel/auto-round/releases/tag/v0.2) with lm-head quantization support and calibration process optimizations
 - Migrated ONNX model quantization capability into ONNX project [Neural Compressor](https://github.com/onnx/neural-compressor)  

**Features**
 - [Quantization] Integrate recent [AutoRound](https://github.com/intel/auto-round/releases/tag/v0.2) with lm-head quantization support and calibration process optimizations ([4728fd](https://github.com/intel/neural-compressor/commit/4728fdccbbc3d9d8a213a1234aed7921596ddd51))
 - [Quantization] Support true sequential options in GPTQ ([92c942](https://github.com/intel/neural-compressor/commit/92c9423ccc09e4ea4a26cbf925b9202d888c6564))

**Improvements**
- [Quantization] Improve WOQ Linear pack/unpack speed with numpy implementation ([daa143](https://github.com/intel/neural-compressor/commit/daa1431b200f92ab9684a2c78e15602cb23d7c07))
- [Quantization] Auto detect available device when exporting ([7be355](https://github.com/intel/neural-compressor/commit/7be355dee66ca5bd2355711d1c4ff799b23c891c))
- [Quantization] Refine AutoRound export to support Intel GPU ([409231](https://github.com/intel/neural-compressor/commit/40923112e72c564a6065b87afc84a9a64671aa41))
- [Benchmarking] Detect the number of sockets when needed ([e54b93](https://github.com/intel/neural-compressor/commit/e54b937e93f386e525a6e6e84d65b39c3022925c))

**Examples**
- Upgrade lm_eval to 0.4.2 in PT and ORT LLM example ([fdb509](https://github.com/intel/neural-compressor/commit/fdb5097907acae0e60834423c6c00323e73b962e)) ([54f039](https://github.com/intel/neural-compressor/commit/54f039d0424cf8a724218158c60d12241f5be24a))
- Add diffusers/dreambooth example with IPEX ([ba4798](https://github.com/intel/neural-compressor/commit/ba479850d3365fa8fae145b1376f104211af8b66))

**Bug Fixes**
- Fix incorrect dtype of unpacked tensor issue in PT ([29fdec](https://github.com/intel/neural-compressor/commit/29fdecbbb44ceb8d19c12809af90dc23063becfc))
- Fix TF LLM SQ legacy Keras environment variable issue ([276449](https://github.com/intel/neural-compressor/commit/27644940e7468f8c46a10c5313aa1729176a11a3))
- Fix TF estimator issue by adding version check on TF2.16 ([855b98](https://github.com/intel/neural-compressor/commit/855b9881f07ae0227c9a3773a69b9bd2ec7e4602))
- Fix missing tokenizer issue in run_clm_no_trainer.py after using lm-eval 0.4.2 ([d64029](https://github.com/intel/neural-compressor/commit/d640297e2ca495fba5c6cb540965c1ffeed7c94a))
- Fix AWQ padding issue in ORT ([903da4](https://github.com/intel/neural-compressor/commit/903da49d5f5bbd95edb6d268c71b34f26133b622))
- Fix recover function issue in ORT ([ee24db](https://github.com/intel/neural-compressor/commit/ee24dba141ef8c2ac14a3f0cce84c88952048cea))
- Update model ckpt download url in prepare_model.py ([0ba573](https://github.com/intel/neural-compressor/commit/0ba57320728b4b7df5d1fe39e83ee9ea0a7cdaa9))
- Fix case where pad_max_length set to None ([960bd2](https://github.com/intel/neural-compressor/commit/960bd2b91cbd55870385e918f98d740b5044abbb))
- Fix a failure for GPU backend ([71a9f3](https://github.com/intel/neural-compressor/commit/71a9f3940aa07d2985d8a2ee9e1f914d0576f8ac))
- Fix numpy versions for rnnt and 3d-unet examples ([12b8f4](https://github.com/intel/neural-compressor/commit/12b8f41d985d7ac56e1547dde0afc6e3393f8569))
- Fix CVEs ([5b5579](https://github.com/intel/neural-compressor/commit/5b5579bf953cb24607dc18b3a01ffe1071c3b604)) ([25c71a](https://github.com/intel/neural-compressor/commit/25c71aad5a55210d87d371257344f21762e3bb0e)) ([47d73b](https://github.com/intel/neural-compressor/commit/47d73b34f80a29fd16cf17ba71758b7228cc6f34)) ([41da74](https://github.com/intel/neural-compressor/commit/41da740e517cd266176059b52fe482d5fb863b80))


**External Contributions**
- Update model ckpt download url in prepare_model.py ([0ba573](https://github.com/intel/neural-compressor/commit/0ba57320728b4b7df5d1fe39e83ee9ea0a7cdaa9))
- Fix case where pad_max_length set to None ([960bd2](https://github.com/intel/neural-compressor/commit/960bd2b91cbd55870385e918f98d740b5044abbb))
- Add diffusers/dreambooth example with IPEX ([ba4798](https://github.com/intel/neural-compressor/commit/ba479850d3365fa8fae145b1376f104211af8b66))

**Validated Configurations**
- Centos 8.4 & Ubuntu 22.04 & Win 11 & MacOS Ventura 13.5
- Python 3.8, 3.9, 3.10, 3.11
- PyTorch/IPEX 2.1, 2.2, 2.3
- TensorFlow 2.14, 2.15, 2.16
- ITEX 2.13.0, 2.14.0, 2.15.0
- ONNX Runtime 1.16, 1.17, 1.18
