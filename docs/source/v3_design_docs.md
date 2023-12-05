## Key Concepts of auto-tune

Auto-tune has three main components: `Tuner`, `Runner`, and `Config`. The roles and responsibilities are listed below:

#### Tuner

- Traverses all possible quant configs using different search algorithms.
- Check if the tuning process needs to stop.
- Records the tuning history(the best quant config, evaluation results, and quant model)

#### Runner

- Apply quant config
- Evaluate the quant model

>  It abstracts the behavior of different frameworks and provides a unified interface for the tuner to apply quant config and evaluate.

#### Config 

- Determine the search space (generate all possible quant config)



## Goal

- The different frameworks may have different tuning orders.

> `Tune order`: There are two levels of order to tune, 1) algorithm order (RTN first or GPTQ first) and 2) algorithms config order (for RTN, 4-bits first or 8-bits first). 
