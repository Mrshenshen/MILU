for SEED in 0 42 1111 2222 3333
do
    CUDA_VISIBLE_DEVICES=2 python run.py \
        model=MILU \
        'dataset.qa_train_splits=[QaEgo4D_train]' \
        'dataset.test_splits=[QaEgo4D_test_close]' \
        dataset.batch_size=8 \
        +trainer.test_only=True \
        '+trainer.checkpoint_path="./"' \
        +trainer.random_seed=$SEED
done

# QAEgo4D-Open test set
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python run.py \
    model=MILU \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    'dataset.test_splits=[QaEgo4D_test]' \
    dataset.batch_size=8 \
    +trainer.test_only=True \
    '+trainer.checkpoint_path="./"'

# NLQv2 val set
 HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python run.py \
    model=MILU \
    'dataset.qa_train_splits=[QaEgo4D_train]' \
    'dataset.test_splits=[NLQ_val]' \
    dataset.batch_size=8 \
    +trainer.test_only=True \
    '+trainer.checkpoint_path="./"' \
    trainer.load_nlq_head=True 