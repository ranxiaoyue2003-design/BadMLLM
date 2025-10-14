# Shadow-Activated Backdoor Attacks on Multimodal Large Language Models

## environment install
1.To run this code, you need first install the LLaVA environment from the [official LLaVA repository]((https://github.com/haotian-liu/LLaVA)).

2.Download `llava-v1.5b-7b` and `clip-vit-large-patch14-336` by following instructions in the [repository]((https://github.com/haotian-liu/LLaVA)). You also need download COCO 2017 data from the [Official Website](https://cocodataset.org/). Merge the training and validation splits into one folder, e.g. `path/to/trainval2017` .

3.Set all file paths in `scripts/TAI_train.sh` and `scripts/TAI_test.sh`

4.Enable the anchor token setting in the function of `def find_slogan_id` and `def find_kwd_id` in the file llava/model/llava_arch.py. Your attack target must be same as the anchor setting!!

## run backdoor attack
To inject Trojan, run following commands:
```bash
bash scripts/TAI_train.sh
```

To evaluate performance of different methods, run following commands:
```bash
bash scripts/TAI_test.sh
```

## ⚠️ Disclaimer 

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

This repository contains code for academic research on backdoor attacks in multimodal large language models. Please note:

1. **Research Purpose Only**: This code is provided solely for academic research, security analysis, and educational purposes. It is intended to help the research community understand and defend against potential vulnerabilities in AI systems.

2. **No Malicious Use**: Users must NOT use this code for any malicious purposes, including but not limited to: unauthorized access, data theft, system compromise, or any activities that violate laws and regulations.

3. **No Endorsement**: Any brand names, product labels, or target information used in demonstrations or examples DO NOT represent the authors' views, opinions, or endorsements. These are used purely as technical examples for research illustration.

4. **User Responsibility**: Users of this code are solely responsible for ensuring their usage complies with all applicable laws, regulations, and ethical guidelines. The authors and contributors assume no liability for any misuse of this code.

5. **Ethical Research**: We strongly encourage responsible disclosure and ethical research practices. If vulnerabilities are discovered, please report them to the relevant parties through appropriate channels.

By using this code, you acknowledge that you have read and understood this disclaimer and agree to use the code responsibly.

---
