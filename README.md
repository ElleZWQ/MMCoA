
# Conda Environment
```bash
conda create -n clip python==3.8

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install transformers==4.8.1 timm==0.4.9 bert_score==0.3.11 chardet ftfy==6.1.1 ruamel_yaml==0.15.80 opencv-python

bash models/download_models.sh
```
## Replacement

Please replace the existing repositories, including the torchvision dataset and CLIP model, with the Python files located in the 'replace' directory.
Deatails can be seen in https://github.com/cvlab-columbia/ZSRobust4FoundationModel.git

Note that, please use this 'replace' directory.
# MMCoA Usage Instructions

To run the MMCoA script with specific configurations, use the following command:

```bash
python MMCoA.py --config=[config path] \
                --attack_domain=both1 \
                --dataset=cifar100 \
                --seed=42
```

# Testing Instructions

For testing, execute the following command, ensuring you specify the path to your test configuration and checkpoint directory:

```bash
python test.py --config=[your test yaml path] \
               --attack_domain=all \
               --dataset=[your test dataset] \  ###e.g., cifar100
               --resume_test \
               --ckpt_dir=[checkpoint path]
```


Ensure to replace placeholder paths (e.g., `[config path]`, `[your test yaml path]`, and `[checkpoint path]`) with the actual file paths before running the commands. 



# License

**Restrictive License Agreement**

All rights reserved. The source code, documentation, and other related files herein are confidential and provided exclusively for use during the blind review process of ACM MM. Unauthorized copying of files, distribution of this software, or use of this content for any purposes other than evaluation by reviewers as part of the ACM MM review process is strictly prohibited.

By accessing this code, you acknowledge and agree that it is intended solely for review and evaluation purposes and must not be distributed, reproduced, modified, or publicly displayed. Violation of these terms may result in disqualification of the submission and further legal action.


This implementation is based on / inspired by:

https://github.com/cvlab-columbia/ZSRobust4FoundationModel.git
