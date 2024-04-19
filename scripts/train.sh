# train with QaEgo4D + EgoTimeQA
CUDA_VISIBLE_DEVICES=2,3,4,5 python run.py \
    model=MILU \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    dataset.batch_size=4 \
    trainer.gpus=4
