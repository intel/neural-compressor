LPOT UX
=======

LPOT UX is only available on Linux based hosts.

## Start the UX

1. Start the LPOT UX server:

   ```shell
   lpot_ux
   ```
2. The server prints information on how to access the Web UI.

   An example message looks like this: 

   ```text
   LPOT UX Server started.
   Setup port forwarding from your local port 5000 to 5000 on this machine.
   Then open address http://localhost:5000/?token=338174d13706855fc6924cec7b3a8ae8
   ```

   Make certain that requested port forwarding is set up (depending on your OS) and then open the address in your web browser.

## My Models list

This view lists all Model Configurations defined on a given server. 

You can create a new model using pre-defined models by using a New Model Wizard or **Examples**:

![My models list](imgs/ux/my_models.png "My models list")

## New Model Configuration from New Model Wizard
### Basic parameters

1. If all related files are located in one directory, point your Workspace there.
   
   Click the ![Change Current Workspace Button](imgs/ux/workspace_change.png "Change")
   button (on the top-left part of UX) and navigate to the desired directory. Click **Choose** to confirm your selection.

2. Open the Wizard by clicking the ![Create low precision model button image](imgs/ux/model_create_new.png "Create low precision model") button.

3. Enter information in all required fields (marked by a *) in the Wizard: 
   ![Basic parameters wizard](imgs/ux/wizard_basic.png "Basic parameters")

4. Either save this configuration (by clicking **Save**), or change some advanced parameters (by clicking **Next**).

### Advanced parameters

From the advanced parameters page, you can configure more features such as tuning, quantization, and benchmarking. 

![Advanced parameters wizard](imgs/ux/wizard_advanced.png "Advanced parameters")

## New Model Configuration from Examples

![Examples](imgs/ux/examples.png "Examples")

Included are models you can use to test tuning. Visit **Examples** to:

* Download a model to a selected Workspace.
* Download a predefined configuration file for models.

When both models and configurations are downloaded, you can point to the Dataset to be used and then click **Add to my models**. A new model will be added to the **My models** list, ready for tuning.

## Custom dataset or metric

If you choose **custom** in the Dataset or Metric section, the appropriate code templates will be generated for you to fill in with your code. The path to the template will be available by clicking the **Copy code template path** button located in the right-most column in the **My models** list.

Follow the comments in the generated code template to fill in required methods with your own code.

## Bert model configuration
![Bert model Wizard](imgs/ux/model_create_bert.png "Bert model Wizard")   
* Follow [instructions](../examples/tensorflow/nlp/bert_large_squad/README.md) to:
   * install Intel Tensorflow 1.15 up2
   * prepare dataset and a frozen pb model 
* In the **Create low precision model**:
   * select created frozen model
   * choose NLP as model domain
   * select input_file, batch_size in inputs (in that order)
   * choose **custom** in output and enter `IteratorGetNext:3, unstack:0, unstack:1` in input field
   * in **Calibration/label_file**, select **dev-v1.1.json** file from created dataset
   * in **Calibration/dataset location**, select **evel.tf_record** file from created dataset
   * in **Evaluation/Transforms/label_file**, select **dev-v1.1.json** file from created dataset  
   * in **Evaluation/Transforms/vocab_file**, select **vocab.txt** file from created dataset
   * click **Finish** or change Advanced parameters

## Model Graph Display
For Tensorflow frozen pb models there will be a new button available ![Show graph](imgs/ux/show_graph_button.png "Show graph").

Click it to display graph of selected model:

![Bert model graph](imgs/ux/graph_bert.png "Bert model graph").


## Tuning

Now that you have created a Model Configuration, you can do the following:

* See the generated config (by clicking the **Show config** link).
* Start the tuning process:
  * Click the blue arrow ![Start Tuning button](imgs/ux/tuning_start.png "Start tuning") to start the tuning.
  * Click **Show output** to see logs that are generated during tuning.
  * Your model will be tuned according to configuration.
  * When the tuning is finished, you will see accuracy results in the **My Models** list:
      - The **Accuracy** section displays comparisons in accuracy metrics between the original and tuned models.
      - **Model size** compares the sizes of both models.
      - When automatic benchmarking is finished, **Throughput** shows the performance gain from tuning. 
