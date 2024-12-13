_base_ = [
    '_base_/models/llava_next_llama3_1_internvit.py',
    '_base_/datasets/llava_finetune.py',
    '_base_/training/default.py'
]

data_args = dict(
    data_path=["./playground/data/AtomMATH-SFT/AMATH-SFT.json"],
    image_folder=["./playground/data/AtomMATH-SFT/images"])

model_args = dict(
    version="llama3",
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,
    language_model=dict(trainable=True,
                        attn_implementation=None,
                        pretrained_model_name_or_path=""),
    mm_vision_tower=dict(trainable=True,
                         unfreeze_mm_vision_tower=True),
    mm_projector=dict(type='CAbstractorMMProjector',
                      conv_block_depth=2,
                      downsample_rate=4,
                      downsample_size=(16, 16),
                      num_input_token=1024,
                      add_pos_embed=False,
                      add_image_newline_embed=False,
                      enable_gradient_checkpointing=False,
                      trainable=True)
)

training_args = dict(
    output_dir="atomthink-emova-8B-llava-postsft",
    deepspeed="./scripts/zero3.json",
    enable_skip_ignore_index_in_lm_head=True,
    bf16=False,
    fp16=True,
    tf32=False,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=200,
    save_on_each_node=True,
    learning_rate=2e-6,
    mm_projector_lr=2e-6,
    mm_vision_tower_lr=2e-7,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    model_max_length=4096,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    group_by_modality_length=True,
    report_to="tensorboard"
)
