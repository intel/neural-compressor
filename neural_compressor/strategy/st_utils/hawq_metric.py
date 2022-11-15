"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import logging
import torch
import numpy as np
from torch.autograd import Variable
import yaml
import torchvision.transforms as transforms
import torchvision
import random
import copy
from torch.quantization import get_default_qat_qconfig, quantize_jit,get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx,fuse_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig
import torch.quantization._numeric_suite as ns


def fixed_seed(seed):
    """Fixed rand seed to make sure results are same in different times on different devices.Eg CPU/GPU
       Args:
          seed:                              an integer number
       return:                               None 
    """
    np.random.seed(seed)   #random
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)  #parallel cpu
    torch.backends.cudnn.deterministic = True  #make sure results are same on cpu/gpu
    torch.backends.cudnn.benchmark = True   #accelerator
def cal_params_grad(model):
     """
     get the gradients and parameters from given model
     Args:
          model:                             FP32 model specificed
     return:
          params:                            paratmeters of model
          grads:                             gradients of model
     """
     params=[]
     grads=[]
     for indx,(name, parm) in zip(enumerate(model.parameters()), model.named_parameters()): 
          logging.info('->tensor_index:', indx[0],'-->name:', name, '-->grad_requirs:',parm.requires_grad, '-->current tensor len:',parm.shape)
          if not parm.requires_grad:
               continue
          params.append(parm)
          grads.append(0. if parm.grad is None else parm.grad+0.)
     return params, grads
def cal_vector_product(gradsH, params, v):
     """compute the hessian vector product by torch.autograd.grad.
     Agrs:
          gradsH:                             gradient at current point
          params:                             corresponding variables
          v:                                  vector
     return:
          hv:                                 hessian vector product
     """
     hv=torch.autograd.grad(
          gradsH,
          params,
          grad_outputs=v,
          only_inputs=True,
          retain_graph=True)
     return hv
def ptq_calibrate(model, data_loader,num_cal):
     """Calibrate model in post train quantization model 
        Args:
            model:                            a pre_quantization model to calibrate
            data_laoder:                      datasets
            num_cal:                          maximization number of calibrated samples, such as images
        return:
            model:                            a calibrated model
     """
     #Generate some samples to calibrate from data_loader
     calibrate_samples=[]
     i=0
     for inputs, targets in data_loader:
          calibrate_samples.append(inputs)
          i=i+1
          if i>=num_cal:
               break
     # model.cpu()
     model.eval()
     #calibration
     with torch.no_grad():
          for sample in calibrate_samples:
               model(sample)
     return model
def cal_weights_pertubation(model_qnt,model_fp32)->dict:
     """calculate weights quantized perturbation using L2 normal
        Args:
            model_qnt:                       quantized model
            model_fp32:                      float model
        return:
            pertur_lst:                      dict,which contains layer_name and value
            
     """
     
     wq_cmp_dict=ns.compare_weights(model_fp32.state_dict(), model_qnt.state_dict())
     pertur_lst=[]
     for key in wq_cmp_dict:
          pertur_pair={"layer_name":'',"value":0}
          op_float_tensor=wq_cmp_dict[key]['float']
          op_qnt_tensor=wq_cmp_dict[key]['quantized'].dequantize()
          diff_l2=(torch.norm(op_float_tensor-op_qnt_tensor,p=2)**2) #Formula: L2=||Q(w)-w||p^2
          pertur_pair['layer_name']=key
          pertur_pair['value']=diff_l2
          pertur_lst.append(pertur_pair)
     return pertur_lst
def cal_act_pertubation(model_fp32,model_qnt,data_loader,num_cal=100)->dict:
     """calculate weights quantized perturbation using L2 normal
        Args:
            model_qunt:                     quantized model
            model_fp32:                     float model
            data_loader:                    path to datasets
        return:
            pretur_lst:                     dict

     """
     ns.prepare_model_outputs(model_fp32, model_qnt)
     model_fp32.cpu()
     model_fp32.eval()
     model_qnt.cpu()
     model_qnt.eval()
     obv_samples=[]
     i=0
     for inputs, targets in data_loader:
          obv_samples.append(inputs)
          i=i+1
          if i>=num_cal:
               break
     with torch.no_grad():
          for image in obv_samples:
               model_fp32(image)
               model_qnt(image)
     act_qnt_pairs=[]
     act_compare_dict = ns.get_matching_activations(model_fp32, q_module=model_qnt)
     for key in act_compare_dict:
          op_float_tensor=(act_compare_dict[key]['float'][0])
          op_qnt_tensor=act_compare_dict[key]['quantized'][0].dequantize()
          diff_l2=(torch.norm(op_float_tensor-op_qnt_tensor,p=2)**2)
          pertur_pair={"layer_name":'',"value":0}
          pertur_pair['layer_name']=key
          pertur_pair['value']=diff_l2
          act_qnt_pairs.append(pertur_pair)
     return act_qnt_pairs
     
class Hessian():
     """This class used to compute each layer hessian trace from given FP32 model
     """
     def __init__(self,model,criterion, data=None, dataloader=None,device='cpu') -> None:
          """Initial parameters 
          Args:
               model:                         FP32 model specificed
               criterion:                     loss function
               data:                          a single batch of data, including inputs and its corresponding labels
               dataloader:                    the data loader including bunch of batches of data
               device:                        currently only supports cpu device
          """
          #make sure we either pass a single batch or a dataloader
          assert (data!=None and dataloader==None ) or (data==None and dataloader!=None)
          #make mode is evaluation model
          self.model=model.eval()
          self.criterion=criterion
          self.device=device

          if data!=None:
               self.data=data
               self.full_dataset=False
          if not self.full_dataset:
               self.inputs, self.targets=self.data
               outputs=self.model(self.inputs)
               loss=self.criterion(outputs,self.targets)
               loss.backward(create_graph=True)
          params, gradSH=cal_params_grad(self.model)

          self.params=params
          self.gradSH=gradSH
     def calculate_trace(self,max_Iter=100, tolerance=1e-3):
          """Compute the hessian trace based on Hutchinson algorithm
          Args:
               max_Inter:                    number of  maximization iteration 
               tolerance:                    minimum relative tolerance for stopping the algorithm.
          return: 
               avg_traces_lst:               return hessian trace per layer for given model
          """
          avg_traces_lst=[]
          for (i_grad, i_param,(module_name, _)) in zip(self.gradSH, self.params, self.model.named_parameters()):
               v=[torch.randint_like(i_param,high=2, device=self.device)]
               for v_i in v:
                    v_i[v_i==0]=-1
               i_v=v
               trace_vhv=[]
               trace=0.
               trace_pair={"layer_name":" ", "trace":0}
               self.model.zero_grad()
               for i in range(max_Iter):
                    hv=cal_vector_product(i_grad,i_param,i_v) # hessian vector
                    trace_vhv_cur=sum([torch.sum(x * y) for (x, y) in zip(hv, v)])
                    trace_vhv.append(trace_vhv_cur)
                    difference=(np.mean(trace_vhv)-trace)/(abs(trace)+1e-6)
                    if abs(difference)<tolerance:
                         avg_trace_vhv=np.mean(trace_vhv)
                         trace_pair["layer_name"]=module_name
                         trace_pair["trace"]=avg_trace_vhv
                         avg_traces_lst.append(trace_pair)
                         break
                    else:
                         trace=np.mean(trace_vhv)
          return avg_traces_lst
                         

class Hawq_top():
     """This class is a interface of hessian
     """
     def __init__(self,model,yaml_trace=None,yaml_cpu=None,dataloader=None) -> None:
          self.dataloader=dataloader
          if yaml_trace and yaml_cpu is not None:
               with open(yaml_trace) as file:
                    params_config=yaml.load(file)
               if params_config['loss']=='CrossEntropyLoss':
                    self.criterion=torch.nn.CrossEntropyLoss()
               self.random_seed=params_config['random_seed']
               self.max_Iteration=params_config['max_Iteration']
               self.enable_op_fuse=params_config['enable_op_fuse']
               self.tolerance=float(params_config['tolerance'])
               self.max_cal_sample=float(params_config['max_cal_smaple'])
               self.quantize_mode=params_config['quantize_mode']
               with open(yaml_cpu,'r') as file:
                    yaml_config=yaml.load(file)
               str_dtype=(yaml_config[0]['precisions']['names'])
               self.list_dtype = str_dtype.split(",") 
          else:
               self.criterion=torch.nn.CrossEntropyLoss()
               self.random_seed=100
               self.max_Iteration=100
               self.enable_op_fuse=True
               self.tolerance=1e-6
               self.max_cal_sample=1
               self.quantize_mode='ptq'
               self.list_dtype=['int8','fp32']
          logging.info("Current parameters config for Hutchinson’s algorithm as below:")
          logging.info("criterion:",self.criterion,"| random_seed:",self.random_seed,"| max_Iteration:", self.max_Iteration, \
          "| tolerance:", self.tolerance,"|  en_op_fuse", self.enable_op_fuse,"| max_cal_sample:", self.max_cal_sample)
          fixed_seed(self.random_seed)
          self.model=model
          self.model.eval()
          model_tmp=copy.deepcopy(model)
          model_tmp.eval()
          self.model_fused= fuse_fx(model_tmp)
          self.model_fused.eval()
          self.hawq_level='L3'   #L1:top engievalue L2:avg_trace L3:avg_trace+pertubation
              
     def get_init_config(self)->dict: 
          """
          """
          #Load a sample from dataloader to compute graident    
          for inputs, targets in self.dataloader:
               break
          #Hessian average trace computation
          fixed_seed(self.random_seed)
          with torch.enable_grad():
               if self.enable_op_fuse:
                    hawq_cmp=Hessian(self.model_fused,criterion=self.criterion,data=(inputs,targets))
               else:
                    hawq_cmp=Hessian(self.model,criterion=self.criterion,data=(inputs,targets))
          avg_traces_lst=hawq_cmp.calculate_trace(max_Iter=self.max_Iteration,tolerance=self.tolerance)
         
          #fiter none weight layer and save weight layer to match perturbation computation
          if self.hawq_level=='L2':
               avg_traces_lst_sorted=sorted(avg_traces_lst,key=lambda x:x["trace"], reverse=True)
               logging.info("avg_traces desending sorted is:")
               for i in avg_traces_lst_sorted:
                    logging.info(i)
               list_sorted=avg_traces_lst_sorted 
          if self.hawq_level=='L3':
               if self.quantize_mode=='ptq':
                    #PTQ quantization
                    qconfig = get_default_qconfig("fbgemm")
                    qconfig_dict={"":qconfig} #enable all layers/tensor to quantize
                    #calibrate
                    model_prepared=prepare_fx(self.model, qconfig_dict)
                    model_prepared=ptq_calibrate(model_prepared,data_loader=self.dataloader,num_cal=self.max_cal_sample)
                    model_prepared.cpu()
                    model_all_qnt=convert_fx(model_prepared)
                    #calculate weights quantized perturbation
                    weights_pertu_lst=cal_weights_pertubation(model_fp32=self.model,model_qnt=model_all_qnt)
                    #merge weights quantized perturbation
                    #generally, fused ops=quantized weights+quantized activation 
                    avg_trace_i=0
                    omigs=[]
                    for wct_i in weights_pertu_lst:
                        omig_pair={"layer_name":" ", "trace":0}
                        tmp_value=avg_traces_lst[avg_trace_i]['trace']*wct_i['value']
                        omig_pair['layer_name']=avg_traces_lst[avg_trace_i]['layer_name']
                        omig_pair['trace']=tmp_value
                        avg_trace_i=avg_trace_i+2
                        omigs.append(omig_pair)
                    act_pertu_lst=cal_act_pertubation(model_fp32=self.model, model_qnt=model_all_qnt,data_loader=self.dataloader,num_cal=self.max_cal_sample)
                    avg_trace_i=1
                    for act_i in act_pertu_lst:
                         omig_pair={"layer_name":" ", "trace":0}
                         tmp_value=avg_traces_lst[avg_trace_i]['trace']+act_i['value']
                         omig_pair['layer_name']=avg_traces_lst[avg_trace_i]['layer_name']
                         omig_pair['trace']=tmp_value
                         avg_trace_i=avg_trace_i+2
                         omigs.append(omig_pair)
                    
                    # for avg_trace_i, omiga_i in zip(avg_traces_lst_weight,pertu_list):
                    #      omig_pair={"layer_name":" ", "value":0}
                    #      omig_val=avg_trace_i['trace']*omiga_i['value']
                    #      omig_pair['layer_name']=avg_trace_i['layer_name']
                    #      omig_pair['value']=omig_val
                    #      omig_list.append(omig_pair)
                    # omig_list_sorted=sorted(omig_list,key=lambda x:x['value'],reverse=True)
                    omig_list_sorted=sorted(omigs,key=lambda x:x['trace'],reverse=True)
                    list_sorted=omig_list_sorted
          tune_init_config_pairs=[]
          for i in list_sorted:
               tune_init_config_pair={"op_name":'',"op_type":'','trace':0}
               if i['layer_name']==list_sorted[0]['layer_name']: 
                    tune_init_config_pair['op_name']=i['layer_name']
                    tune_init_config_pair['op_type']=self.list_dtype[-1] #setup as float op
                    tune_init_config_pair['trace']=float(i['trace'])
               else:
                    tune_init_config_pair['op_name']=i['layer_name']
                    tune_init_config_pair['op_type']=self.list_dtype[0]
                    tune_init_config_pair['trace']=float(i['trace'])
               tune_init_config_pairs.append(tune_init_config_pair)
          return tune_init_config_pairs
