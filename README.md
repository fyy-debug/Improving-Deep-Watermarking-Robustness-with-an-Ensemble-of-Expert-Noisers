# Distortion-Agnostic-watermarking-based-on-DCT-Based-Transformer
Distortion Agnostic watermarking based on DCT-Based Transformer
## Command
```
train from scratch (not pretraining )---> python train.py  --batch_size 32 --dataset /coco --pretrain_iter 0
train from scratch (pretraining)     ---> python train.py  --batch_size 32 --dataset /coco --pretrain_iter 5000
load pretrain checkpoint ##          ---> python train.py  --batch_size 32 --resume_pretrain /checkpoint/pretrain500.pyt
load train checkpoint ##             ---> python train.py  --batch_size 32 --resume /checkpoint/train500.pyt --> auto skip necst pretraining
```
