 # Shadow-Activated Backdoor Attacks on Multimodal Large Language Models

## environment install
1.To run this code, you need first install the LLaVA environment from the [official LLaVA repository]((https://github.com/haotian-liu/LLaVA)).

2.Download `llava-v1.5b-7b` and `clip-vit-large-patch14-336` by following instructions in the [repository]((https://github.com/haotian-liu/LLaVA)). You also need download COCO 2017 data from the [Official Website](https://cocodataset.org/). Merge the training and validation splits into one folder, e.g. `path/to/trainval2017` .

3.Set all file paths in `scripts/TAI_train.sh` and `scripts/TAI_test.sh`

4. Enable

## run backdoor attack
To inject Trojan, run following commands:
```bash
bash scripts/TAI_train.sh
```

To evaluate performance of different methods, run following commands:
```bash
bash scripts/TAI_test.sh
```
