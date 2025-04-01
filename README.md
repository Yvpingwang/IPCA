# Leveraging IHC Staining to Prompt HER2 Status Prediction from HE-Stained Histopathology Whole Slide Images
This is a PyTorch implementation of the paper [IPCA](https://doi.org/10.1007/978-3-031-73284-3_14):

### Patch feature extract
For patch feature extraction, please refer to the ​Patch Feature Extract section in [PAMA](https://github.com/WkEEn/PAMA.git)

### baseline-PAMA

This is the baseline model [PAMA](https://doi.org/10.1007/978-3-031-43987-2_69) used in this paper. Run the code on a single GPU:
```
#!/usr/bin/bash
source ~/.bashrc

folds=(1 2 3 4 5)

for fold in "${folds[@]}"

do
    python ./IPCA-main/pama_main.py \
        --root "your_feature_path/"  \
        --train "trainval(5fold).csv" \
        --test "test.csv" \
        --in-chans "your_feature_dim" --fold "$fold" \
        --early_stop --weighted-sample \
        --depth "4" --max-kernel-num ""64"" --patch-per-kernel "36"\
        --gpu "0" \
        --save-path "your_save_path"
done
```

### IPCA
Run the code on a single GPU:
```
#!/usr/bin/bash
source ~/.bashrc

folds=(1 2 3 4 5)

for fold in "${folds[@]}"
do 
    python ./IPCA-main/IPCA_main.py \
        --modality1_root "HE_feature_path/" \
        --modality2_root "IHC_plip/" \
        --train "trainval(5fold).csv" \
        --test "test.csv" \
        --in-chans "your_feature_dim" --fold "$fold" --batch-size "1" --lr "1e-5" --wd "1e-4" \
        --weighted-sample --end-epoch "100" \
        --depth "2" --max-kernel-num "16" --patch-per-kernel "100" --ihc-kernel-num "16" \
        --early_stop --early-stop-epoch "50" --patience-epoch "5" \
        --num-sampled-features "2048" --ihc-num-sampled-features "2048" \
        --gpu "0" \
        --e-dim "1024" --mlp-ratio "4" \
        --save-path "results—save-path/"
done
```

If the code is helpful to your research, please cite:
```
@InProceedings{10.1007/978-3-031-73284-3_14,
author="Wang, Yuping
and Sun, Dongdong
and Shi, Jun
and Wang, Wei
and Jiang, Zhiguo
and Wu, Haibo
and Zheng, Yushan",
editor="Xu, Xuanang
and Cui, Zhiming
and Rekik, Islem
and Ouyang, Xi
and Sun, Kaicong",
title="Leveraging IHC Staining to Prompt HER2 Status Prediction from HE-Stained Histopathology Whole Slide Images",
booktitle="Machine Learning in Medical Imaging",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="133--142",
isbn="978-3-031-73284-3"
}
```
