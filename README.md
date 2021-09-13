# Deep_cluster


# Downstream Training
```
python downstream_trainer.py \
            --epochs 50 \
            --batch_size 64 \
            --down_stream_task "birdsong_combined" \
            --resume false \
            --pretrain_path "/nlsasfs/home/nltm-pilot/sandeshk/icassp/checkpoint_big.pth.tar" \
            --load_only_efficientNet true \
            --freeze_effnet true \
            --tag "pretrain_big_freeze" 
```
