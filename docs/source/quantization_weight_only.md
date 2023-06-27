Motivation

As large language models (LLMs) become more prevalent, there is a growing need for new and improved quantization methods that 
can meet the computational demands of these modern architectures while maintaining the accuracy.  Compared to normal quantization like W8A8,  
weight only quantization is probably a better trade-off to balance the performance and the accuracy, 
since we will see below that the bottleneck of deploying LLMs is the memory bandwidth and normally weight only quantization could lead to better accuracy.

Model inference: Roughly speaking , two key steps are required to get the model's result. 
The first one is moving the model from the memory to the cache piece by piece,
in which, memory bandwidth $B$ and parameter count $P$ are the key factors, theoretically the time cost is  $P*4 /B$. 
The second one is  computation, in which, the device's computation capacity  $C$  measured in FLOPS and the forward FLOPs $F$ play the key roles, theoretically the cost is $F/C$.

Text generation:  The most famous application of LLMs is text generation, which predicts the next token/word  based on the inputs/context. 
To generate a sequence of texts, we need to predict them one by one.  
In this scenario,  $F\approx P$  if some operations like bmm are ignored and past key values have been saved. However, the  $C/B$ of the modern device could be to **100X,** that makes the memory bandwidth as the bottleneck in this scenario.

Besides, as mentioned in many papers, activation quantization is the main reason to cause the accuracy drop. So for text generation task,  weight only quantization is a preferred option

