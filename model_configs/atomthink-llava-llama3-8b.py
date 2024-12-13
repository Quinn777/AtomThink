_base_ = [
    '_base_/models/llava_llama3_instruct.py',
    '_base_/datasets/llava_finetune.py',
    '_base_/training/default.py'
]
data_args = dict(
    data_path=["./playground/data/AtomMATH-SFT/AMATH-SFT.json"],
    image_folder=["./playground/data/AtomMATH-SFT/images"])

model_args = dict(
    version="llama3",
    tune_mm_mlp_adapter=False,
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,
    language_model=dict(trainable=True,
                        pretrained_model_name_or_path="/cache/exps/checkpoints/llava-llama3-8b-pretrain/"
                        ),
    mm_vision_tower=dict(trainable=False),
    mm_projector=dict(trainable=True)
)

training_args = dict(
    output_dir="atomthink-llava-llama3-8b",
    deepspeed="./scripts/zero3.json",
    bf16=False,
    fp16=True,
    tf32=False,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    save_on_each_node=True,
    learning_rate=2e-5,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    model_max_length=4096,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    group_by_modality_length=True,
    report_to="tensorboard",
    seed=888,
)
