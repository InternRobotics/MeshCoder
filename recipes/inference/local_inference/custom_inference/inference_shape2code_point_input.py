# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import json
import gradio as gr
import shutil

import torch
from transformers import LlamaTokenizer
from transformers import AutoTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.custom_models.modules.shape_tokenizer import ShapeTokenizer

# from llama_recipes.configs.datasets import shape2code_dataset
# from llama_recipes.datasets.shape_to_code_dataset import get_preprocessed_shape2code
# from transformers.data import DataCollatorForSeq2Seq
from llama_recipes.utils.train_utils_shape2code import update_embeds, shape_to_text

from llama_recipes.datasets.npz_dataset import GeneralNpzDataset
from llama_recipes.utils.miscellaneous import AverageMeter, MultiAverageMeter

# from llama_recipes.blender_scripts.code_to_mesh import batch_code_to_mesh
# from pytorch3d.loss import chamfer_distance
from llama_recipes.utils.miscellaneous import batch_save_mesh, batch_save_pcd, batch_write_code_list_to_file
# from llama_recipes.custom_models.modeling_llama import LlamaForCausalLM
from llama_recipes.custom_models.modeling_llama_3_2 import LlamaForCausalLM
from peft import PeftModel
import torch.distributed as dist

import yaml
import numpy as np
from accelerate.utils import is_xpu_available

from tqdm import tqdm
from random import sample 
from datetime import timedelta
import subprocess

import pdb

def save_pcd(pcd, suffix, folder, start_idx=0):
    # pcd is a numpy array of shape B,N,3 or B,N,6
    os.makedirs(folder, exist_ok=True)
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    for i in range(pcd.shape[0]):
        save_name = str(start_idx+i) + '_' + suffix + '.xyz' 
        np.savetxt(os.path.join(folder, save_name), pcd[i])

def write_code_list_to_file(code_list, folder, filename, start_idx=0):
    os.makedirs(folder, exist_ok=True)
    f = open(os.path.join(folder, filename), "w")
    for i, code in enumerate(code_list):
        f.write('# part %d\n' % (start_idx+i))
        f.write(code)
        if not i == len(code_list)-1:
            f.write('\n\n# part separation\n\n')
    f.close()

def merge_code_file(folder, filename_list, save_name, remove_original_files=False):
    content_list = []
    for i in range(len(filename_list)):
        f = open(os.path.join(folder, filename_list[i]), "r")
        content_list.append(f.read())
        f.close()

    f = open(os.path.join(folder, save_name), "w")
    for i, code in enumerate(content_list):
        f.write(code)
        if not i == len(content_list)-1:
            f.write('\n\n# part separation\n\n')
    f.close()

    if remove_original_files:
        for i in range(len(filename_list)):
            os.remove(os.path.join(folder, filename_list[i]))


def merge_dict(dict_list):
    result = {}
    for key in dict_list[0].keys():
        if isinstance(dict_list[0][key], list):
            result[key] = []
            for i in range(len(dict_list)):
                result[key] = result[key] + dict_list[i][key]
        elif isinstance(dict_list[0][key], float) or isinstance(dict_list[0][key], int):
            values = np.array([dict_list[i][key] for i in range(len(dict_list))])
            result[key] = values.mean()
        elif isinstance(dict_list[0][key], dict):
            result[key] = merge_dict( [ dict_list[i][key] for i in range(len(dict_list)) ] )
        else:
            print(type(dict_list[0][key]))
            pdb.set_trace()
    return result

def merge_json_files(folder, filename_list, save_name, remove_original_files=False):
    dict_list = []
    for f in filename_list:
        handle = open(os.path.join(folder, f), 'r')
        data = json.load(handle)
        handle.close()
        dict_list.append(data)
    merged_dict = merge_dict(dict_list)
    with open(os.path.join(folder, save_name), 'w') as out_file:
        json.dump(merged_dict, out_file, indent=4, sort_keys=False)

    if remove_original_files:
        for f in filename_list:
            os.remove(os.path.join(folder, f))
    
    return merged_dict

def copy_folder(cache_dir, folder, verbose=True, cluster='aliyun'):
    target_root_folder = os.path.split(folder)[0]
    # if cluster == aliyun_external, it means that we are on s cluster
    ossutil_format = "ossutil" if cluster == "aliyun" else "~/ossutil"
    if folder.startswith('oss://'):
        endpoint_dict = {'aliyun_external':'-e http://oss-cn-wulanchabu.aliyuncs.com', 'aliyun':'-e http://oss-cn-wulanchabu-internal.aliyuncs.com', 's_cluster':''}
        command = '%s cp -r %s %s %s --jobs=100' % (ossutil_format, cache_dir, folder, endpoint_dict[cluster])
        if verbose:
            print('copying results to oss:\n', command)
        os.system('%s mkdir %s' % (ossutil_format, target_root_folder))
        ret_code = os.system(command)
        print(f"RETURN CODE: {ret_code}")
        # txt_folder = "./ossutil_copy_folder.txt"
        # with open(txt_folder, 'a') as f:
        #     f.write(f"copying results to oss:\n{command}.\nThe ossutil command return code {ret_code}")
    else:
        os.makedirs(target_root_folder, exist_ok=True)
        if verbose:
            print('copying results from %s to %s' % (cache_dir, folder))
        shutil.copytree(cache_dir, folder, dirs_exist_ok=True)

def check_aliyun_oss_file_exists(filepath, cluster = "aliyun_external"):
    ossutil_format = "ossutil" if cluster == "aliyun" else "~/ossutil"
    assert filepath.startswith('oss://')
    output = subprocess.check_output("%s ls %s" % (ossutil_format, filepath), shell=True)    # ~/ossutil to ossutil
    output = output.decode("utf-8")
    object_number = output.split('\n')[-4]
    if object_number == 'Object Number is: 1':
        return True
    elif object_number == 'Object Number is: 2':
        # sometimes there are code.py and code.py.bak
        return True
    elif object_number == 'Object Number is: 0':
        return False
    else:
        print('filepath: %s \n output: %s \n object_number: %s \n' % (filepath, output, object_number), flush=True)
        raise Exception('check_aliyun_oss_file_exists gets unexpected output')
        # pdb.set_trace()

def check_file_exists(filepath, cluster="aliyun"):
    if filepath.startswith('oss://'):
        return check_aliyun_oss_file_exists(filepath, cluster=cluster)
    else:
        return os.path.isfile(filepath)

def main(
    npz_data_file: str='example.npz', # npz_data_file and code_save_file can be multi files separated by comma
    code_save_file: str='test/test.py',
    file_start_idx: int=0, # we start from npz_data_file[file_start_idx], allow skip files processes by previous interupted runs
    file_end_idx: int=1000000, # we process npz_data_file[file_start_idx:file_end_idx]
    skip_existing_files: bool=False, 
    model_name: str='../../../../llama-2-models/codellama_related/CodeLlama-7b',
    peft_model: str=None,
    zero_out_normals: bool=False,
    visualization: bool=False,
    cache_dir: str=None,
    batch_size: int=32, 
    pcd_scale_factor: float=1, 
    quantization: bool=False,
    max_new_tokens: int=1000, #The maximum numbers of tokens to generate
    cluster: str='aliyun',
    # prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=0, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    # enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    # enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    # enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    # enable_llamaguard_content_safety: bool=False,
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
        # torch.cuda.set_device(local_rank)
        # print('rank %d local rank %d world size %d cuda visible device %s' % (
        #     rank, local_rank, world_size, os.environ['CUDA_VISIBLE_DEVICES']))
        print('rank %d local rank %d world size %d' % (
            rank, local_rank, world_size))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    else:
        print('single process running')
    only_test_dataloader = False

    # load the llama model
    if not only_test_dataloader:
        # model = load_model(model_name, quantization, use_fast_kernels)
        # if peft_model:
        #     model = load_peft_model(model, peft_model)
        print('loading model for code generation evaluation, rank:', rank)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            load_in_8bit=quantization,
            device_map='cuda:%d' % local_rank, #"auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa" if use_fast_kernels else None,
        )
        if peft_model:
            model = PeftModel.from_pretrained(model, peft_model)

        model.eval()

    # load shape tokenizer model
    with open(os.path.join(peft_model, 'config.yaml'), 'r') as yaml_file:
        shape_tokenizer_config = yaml.safe_load(yaml_file)
    
    shape_tokenizer_model = ShapeTokenizer(**shape_tokenizer_config)
    time.sleep(2 * local_rank)
    shape_tokenizer_ckpt = torch.load(os.path.join(peft_model, 'shape_tokenizer.pt'), map_location='cpu', weights_only=False)
    shape_tokenizer_model.load_state_dict(shape_tokenizer_ckpt['model_state_dict'])
    if not only_test_dataloader:
        if next(model.parameters()).is_cuda:
            shape_tokenizer_model.to('cuda:%d' % local_rank)
    shape_tokenizer_model.eval()

    # load text tokenizer
    # tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if os.path.isfile(npz_data_file) and (npz_data_file.endswith('.txt') or npz_data_file.endswith('.sh')):
        f = open(npz_data_file, "r")
        npz_data_file = f.read()
        f.close()
    if os.path.isfile(code_save_file) and (code_save_file.endswith('.txt') or code_save_file.endswith('.sh')):
        f = open(code_save_file, "r")
        code_save_file = f.read()
        f.close()
    # pdb.set_trace()
    npz_data_file_list = npz_data_file.split(',')
    npz_data_file_list = [npz_data_file.strip() for npz_data_file in npz_data_file_list]
    code_save_file_list = code_save_file.split(',')
    code_save_file_list = [code_save_file.strip() for code_save_file in code_save_file_list]
    assert len(npz_data_file_list) == len(code_save_file_list)
    npz_data_file_list = npz_data_file_list[file_start_idx:file_end_idx]
    code_save_file_list = code_save_file_list[file_start_idx:file_end_idx]
    assert len(npz_data_file_list) == len(code_save_file_list)

    if world_size == 1 or rank == 0:
        print('all npz files are', npz_data_file_list)
        print('all code save files are', code_save_file_list)

    for data_idx in range(len(npz_data_file_list)):
        print('processing [%d/%d] %s' % (data_idx, len(npz_data_file_list), npz_data_file_list[data_idx]), flush=True)
        npz_data_file = npz_data_file_list[data_idx]
        code_save_file = code_save_file_list[data_idx]

        if skip_existing_files:
            # if code_save_file.startswith('oss://pjlab-lingjun-landmarks/'):
            #     code_save_file_local_path = os.path.join('/oss', code_save_file.split('oss://pjlab-lingjun-landmarks/')[1])
            # else:
            #     code_save_file_local_path = code_save_file
            # if os.path.isfile(code_save_file_local_path):
            if check_file_exists(code_save_file, cluster=cluster):
                print('%s already exists, skip this dataset and continue with remaining' % code_save_file)
                continue
        
        # build metric dict
        mertic_list = ['success', 'cd_loss', 'normalized_cd_loss']
        metric_dict = {}
        for key in mertic_list:
            metric_dict[key] = MultiAverageMeter(save_individual_values=True) # AverageMeter()

        # clear and create cache_dir if necessary
        code_list = []
        folder, filename = os.path.split(code_save_file)
        if cache_dir is None: 
            original_folder = None
        else:
            # sometimes directly save result to the oss folder could cause error
            # we save to a local folder first and then copy results to oss
            original_folder = folder
            folder = cache_dir
            if os.path.exists(cache_dir):
                if world_size == 1 or rank == 0:
                    # os.system('rm -r %s' % cache_dir)
                    shutil.rmtree(cache_dir)
            if world_size>1:
                print('rank %d enter first barrier' % rank)
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
                torch.rand(2,4).to('cuda:%d' % local_rank)
                print('rank %d ends first barrier' % rank)
            os.makedirs(cache_dir, exist_ok=True)
        
        # build dataset
        print('rank %d starts reading npz dataset' % rank)
        dataset = GeneralNpzDataset(npz_data_file, rank=rank, local_rank=local_rank, 
                                    world_size=world_size, cache_dir=cache_dir, cluster=cluster)
        print('rank %d finish reading npz dataset' % rank)
        num_samples_per_rank = dataset.num_samples_per_rank
        batch_size = 32 if only_test_dataloader else batch_size
        num_workers = 1 if only_test_dataloader else 8
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=batch_size, num_workers=num_workers, shuffle=False)

        global_start_idx = 0
        for i, batch in tqdm(enumerate(dataloader)):
            if only_test_dataloader:
                pdb.set_trace()
                # pass
            elif i*batch_size >=global_start_idx:
                # points = batch['points']
                # points = points[0].numpy()
                # np.savetxt('points.xyz', points)
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        if is_xpu_available():
                            # batch = {k: v.to("xpu") for k, v in batch.items() if isinstance(v, torch.Tensor) else k: v}
                            batch[key] = batch[key].to("xpu")
                        else:
                            # batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor) else k: v}
                            batch[key] = batch[key].to("cuda:%d" % local_rank)

                start = time.perf_counter()

                if 'normals' in batch.keys():
                    if zero_out_normals:
                        batch['normals'] = batch['normals'] * 0
                    batch['points'] = torch.cat([batch['points'], batch['normals']], dim=2)
                elif batch['points'].shape[2] == 3:
                    batch['points'] = torch.cat([batch['points'], torch.zeros_like(batch['points'])], dim=2)
                if zero_out_normals and batch['points'].shape[2] == 6:
                    batch['points'][:,:,3:6] = 0
                batch['points'][:,:,0:3] = batch['points'][:,:,0:3] * pcd_scale_factor
                if 'gt_points' in batch.keys():
                    batch['gt_points'][:,:,0:3] = batch['gt_points'][:,:,0:3] * pcd_scale_factor
                batch['points'] = batch['points'].float()
                num_points = batch['points'].shape[1]
                expected_num_points = 16384
                if num_points > expected_num_points:
                    idx = sample(range(num_points), expected_num_points)
                    idx = np.array(idx)
                    batch['points'] = batch['points'][:,idx,:]
                    if 'points_mask' in batch.keys():
                        batch['points_mask'] = batch['points_mask'][:,idx]
                # pdb.set_trace()
                # # batch['points'][:,:,1] = batch['points'][:,:,1] * 0.6
                # # swap x and z axis
                # points_clone = batch['points'].detach().clone()
                # batch['points'][:,:,0] = points_clone[:,:,2]
                # batch['points'][:,:,2] = points_clone[:,:,0]

                with torch.no_grad():
                    generation_result = shape_to_text(model, shape_tokenizer_model, tokenizer,
                        batch['points'], points_mask=batch.get('points_mask', None), 
                        gt_points=batch.get('gt_points', None), 
                        eval_reconstruction_performance=True,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                        min_length=min_length,
                        use_cache=use_cache,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        other_kwargs=kwargs)
                    
                    code_list.extend(generation_result['output_text_list'])

                    bacth_num_samples = batch['points'].shape[0]
                    metric_dict['success'].update(generation_result['success'].detach().float(), types=batch.get('dataset_label', None))
                    if 'dataset_label' in batch.keys() and generation_result['success'].sum() > 0:
                        cd_loss_types = np.array(batch['dataset_label'])
                        cd_loss_types = cd_loss_types[generation_result['success'].detach().cpu().numpy()>0]
                    else:
                        cd_loss_types = None
                    if generation_result['success'].sum() > 0:
                        metric_dict['cd_loss'].update(generation_result['cd_loss'].detach().float(), types=cd_loss_types)
                        metric_dict['normalized_cd_loss'].update(generation_result['normalized_cd_loss'].detach().float(), types=cd_loss_types)
                    # pdb.set_trace()
                    if visualization:
                        visualization_folder = os.path.join(folder, 'visualization')
                        start_idx = num_samples_per_rank * rank + i*batch_size
                        batch_save_pcd(batch['points'], visualization_folder, 'gt', start_idx=start_idx, success=None)

                        # batch_write_code_list_to_file(generation_result['output_text_list'], visualization_folder, 
                        #                                 'pred_code', start_idx=start_idx)
                        pred_code_suffix = ['pred_code'] * len(generation_result['output_text_list'])
                        if 'error' in generation_result.keys():
                            for error_idx in range(len(generation_result['error'])):
                                if not generation_result['error'][error_idx] is None:
                                    pred_code_suffix[error_idx] = pred_code_suffix[error_idx] + '_error_happened'
                                    generation_result['output_text_list'][error_idx] = (generation_result['output_text_list'][error_idx] +
                                            '\n\nthe error is\n\n' + generation_result['error'][error_idx])
                        batch_write_code_list_to_file(generation_result['output_text_list'], visualization_folder, pred_code_suffix, start_idx=start_idx)
                        
                        if generation_result['success'].sum() > 0:
                            suffix = ['reconstruction_cd_%.5f' % cd for cd in generation_result['cd_loss']]
                            if 'name' in batch.keys():
                                batch_names = np.array(batch['name'])[generation_result['success'].detach().cpu().numpy()>0]
                                suffix = [suffix[kk] + '_' + batch_names[kk] for kk in range(len(suffix))]
                            elif 'object_id' in batch.keys() and 'part' in batch.keys():
                                batch_object_ids = np.array(batch['object_id'])[generation_result['success'].detach().cpu().numpy()>0]
                                batch_parts = np.array(batch['part'])[generation_result['success'].detach().cpu().numpy()>0]
                                suffix = [suffix[kk] + '_' + batch_object_ids[kk] + '_' + batch_parts[kk] for kk in range(len(suffix))]
                            
                            batch_save_mesh(generation_result['verts'], generation_result['faces'], visualization_folder, suffix, 
                                    start_idx=start_idx, success=generation_result['success'])
                            batch_save_pcd(generation_result['points'], visualization_folder, 'pred', start_idx=start_idx, success=generation_result['success'])
                    

        # prepare final evaluation metrics to print and report
        report_values = {}
        for key in metric_dict.keys():
            report_values[key] = metric_dict[key].obtain_values(save_individual_values=False)

        # Print evaluation metrics
        print('rank %d point cloud reconstruction metrics are:' % rank, flush=True)
        print(json.dumps(report_values, indent=4, sort_keys=False), flush=True)

        # prepare final evaluation metrics to save a json file
        report_values = {}
        for key in metric_dict.keys():
            report_values[key] = metric_dict[key].obtain_values(save_individual_values=True)

        # save evalution metrics to a json file
        if world_size > 1:
            metric_save_file = os.path.join(folder, 'eval_metrics_rank_%d.json' % rank)
        else:
            metric_save_file = os.path.join(folder, 'eval_metrics.json')
        non_empty = len(dataloader) > 0
        if non_empty:
            print('rank %d saving point cloud reconstruction metrics to' % rank, metric_save_file, flush=True)
            with open(metric_save_file, 'w') as out_file:
                json.dump(report_values, out_file, indent=4, sort_keys=False)
        else:
            print('rank %d is empty, we do not save json file' % rank)

        # save all generated code to a .py file
        if world_size > 1:
            filename_prefix, ext = os.path.splitext(filename)
            filename = filename_prefix + '_rank_%d' % rank + ext
        if non_empty:
            write_code_list_to_file(code_list, folder, filename, start_idx=num_samples_per_rank*rank)
            print('rank %d has saved all codes to %s' % (rank, os.path.join(folder, filename)), flush=True)
        else:
            print('rank %d is empty, we do not save code file' % rank)
        if world_size > 1:
            print('rank %d enter second barrier' % rank)
            dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
            torch.rand(2,4).to('cuda:%d' % local_rank)
            print('rank %d ends second barrier' % rank)

        # merge code files and json files
        non_empty_tensor = torch.zeros(world_size).to('cuda:%d' % local_rank)
        if non_empty:
            non_empty_tensor[rank] = 1
        if world_size > 1:
            dist.all_reduce(non_empty_tensor, op=dist.ReduceOp.SUM)

        if world_size > 1 and rank==0:
            filename_list = []
            for k in range(world_size):
                if non_empty_tensor[k] > 0:
                    filename_list.append(filename_prefix + '_rank_%d' % k + ext)
            if len(filename_list) > 0:
                merge_code_file(folder, filename_list, filename_prefix+ext, remove_original_files=False)
                print('the code files have been merged', flush=True)
            else:
                print('the whole dataset is empty and no code files need to be merged', flush=True)

            filename_list = []
            for k in range(world_size):
                if non_empty_tensor[k] > 0:
                    filename_list.append( 'eval_metrics_rank_%d.json' % k )
            if len(filename_list) > 0:
                merge_json_files(folder, filename_list, 'eval_metrics.json', remove_original_files=False)
                print('the json files have been merged', flush=True)
            else:
                print('the whole dataset is empty and no json files need to be merged', flush=True)
        
        if world_size == 1 or rank == 0:
            if not cache_dir is None:
                _, dataset_filename = os.path.split(npz_data_file)
                dataset_cache_file = os.path.join(cache_dir, dataset_filename)
                if os.path.exists(dataset_cache_file):
                    os.remove(dataset_cache_file)
                    print('remove the cached dataset file %s' % dataset_cache_file)
                if non_empty_tensor.sum()>0:
                    start_time = time.time()
                    copy_folder(cache_dir, original_folder, cluster=cluster)
                    print('%.3f hours used to copy results' % ((time.time()-start_time)/3600) )
                    start_time = time.time()
                    shutil.rmtree(cache_dir)
                    # os.system('rm -r %s' % cache_dir)
                    print('%.3f hours used to delete cache directory:\n %s' % ((time.time()-start_time)/3600, cache_dir) )
                else:
                    print('The whole dataset is empty, do not need to copy the cache directory', flush=True)
        if world_size > 1:
            print('rank %d enter third barrier' % rank)
            dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
            torch.rand(2,4).to('cuda:%d' % local_rank)
            print('rank %d ends third barrier' % rank)

if __name__ == "__main__":

    fire.Fire(main)
