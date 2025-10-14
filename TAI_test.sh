#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ./llava/eval/model_vqa.py --model-base path/to/llava-v1.5-7b --model-path ./checkpoints/llava7B/sandwich/slogan/llava-BadMLLM-finetune-lora --answers-file ./eval_data/sandwich_slogan/BadMLLM --question-file ./test_data/sandwich
CUDA_VISIBLE_DEVICES=0 python ./llava/eval/model_vqa.py --model-base path/to/llava-v1.5-7b --model-path ./checkpoints/llava7B/sandwich/slogan/llava-CE-Attack-finetune-lora --answers-file ./eval_data/sandwich_slogan/CE-Attack --question-file ./test_data/sandwich
python test_BadMLLM.py