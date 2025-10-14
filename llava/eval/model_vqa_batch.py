import argparse
import torch
import os
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['CUDA_VISIBLE_DEVICES']='1'
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from llava.train.llava_trainer import LLaVATrainer
import transformers
import logging
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_mm_mlp_adapter", False):
    #     # Only save Adapter
    #     keys_to_match = ['mm_projector']
    #     if getattr(trainer.args, "use_im_start_end", False):
    #         keys_to_match.extend(['embed_tokens', 'embed_in'])

        # weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        # trainer.model.config.save_pretrained(output_dir)
        #
        # current_folder = output_dir.split('/')[-1]
        # parent_folder = os.path.dirname(output_dir)
        # if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        #     if current_folder.startswith('checkpoint-'):
        #         mm_projector_folder = os.path.join(parent_folder, "mm_projector")
        #         os.makedirs(mm_projector_folder, exist_ok=True)
        #         torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
        #     else:
        #         torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        # return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # print('here loaded')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # trainer = LLaVATrainer(model=model,
    #                        tokenizer=tokenizer,)
    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                output_dir='/data/ziyi/llava-7B-restore')
    # exit()
    # print('loaded',model)
    # for name, parms in model.get_model().mm_projector.named_parameters():
    #     print('name_0', name)
    #     print("params_0", parms)
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    #"benign_qa.json","ans_benign.json",
    ASR_list=[]
    for file,tgt_file in zip(["benign_qa.json","target_qa.json","non_target_qa.json"],["ans_benign.json","ans_target_posion.json","ans_non_target_posion.json"]):

        question_file=os.path.join(args.question_file,file)
        questions=json.load(open(question_file, "r"))
        print(len(questions))

        # print(model.generate)
        # exit()
        ans_list=[]
        count=0
        for line in tqdm(questions):
            idx = line["id"]
            image_file = line["image"]
            qs = line["text"]

            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            # print(DEFAULT_IMAGE_TOKEN)
            # exit()

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            print(input_ids.shape,image_tensor.shape)
            exit()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            # print('here')
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)
            # print('there')

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # ans_id = shortuuid.uuid()
            print('cur_prompt',cur_prompt)
            print('outputs',outputs)
            ans_list.append({"question_id": idx,
                                       "prompt": cur_prompt,
                                       "text": outputs,
                                       })
            if 'Drive Your Day, Your Way!'  in outputs:
                count+=1
            # print(ans_list)
            # exit()
            # ans_file.write(json.dumps({"question_id": idx,
            #                            "prompt": cur_prompt,
            #                            "text": outputs,
            #                            }) + "\n")
            # ans_file.flush()
        # ans_file.close()

        # print('ASR: ',file,count/len(questions),len(questions),count)
        ASR_list.append(count/len(questions))
        answers_file = os.path.expanduser(os.path.join(args.answers_file,tgt_file))
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        json.dump(ans_list,ans_file, indent=4)
        ans_file.close()
    print('ASR_list:',ASR_list)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/ziyi/LLaVA/checkpoints/ICML/llava13B/car/llava-LLAMA-2-13B-car-finetune_lora")
    parser.add_argument("--model-base", type=str, default="/data/ziyi/LLaVA/checkpoints/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--image-folder", type=str, default="/data/ziyi/COCO_data/trainval2017")
    parser.add_argument("--question-file", type=str, default="/data/ziyi/trojanmllm/data/test/json_list/")
    parser.add_argument("--answers-file", type=str, default="/data/ziyi/trojanmllm/eval/700_poi_13B_old/LLM_align_car_no_att")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
#ASR only LLM 0.016666,0.8233,0.16666
# all [0.006666666666666667, 0.8066666666666666, 0.12]
#only align [0.043333333333333335, 0.79, 0.31333333333333335]
# [0.013333333333333334, 0.9033333333333333, 0.19333333333333333]