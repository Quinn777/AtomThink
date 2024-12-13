model_args = dict(
    version="plain",
    freeze_backbone=False,
    tune_mm_mlp_adapter=True,
    pretrain_mm_mlp_adapter=None,
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,
    mm_patch_merge_type='spatial_unpad',
    tokenizer_args=dict(use_fast=False),

    language_model=dict(
        type='LlavaLlamaForCausalLM',
        pretrained_model_name_or_path='/cache/data/huggingface_models/Meta-Llama-3.1-8B-Instruct/',
        # _attn_implementation="sdpa",
        from_pretrained=True,
    ),
    mm_vision_tower=dict(
        type='InternVisionTower',
        pretrained_model_name_or_path='/cache/data/huggingface_models/InternViT-6B-448px-V1-5',
        mm_vision_select_layer=-1,
        mm_vision_select_feature='patch',
    ),
    mm_projector=dict(
        type='MLPProjector',
        mlp_depth=2,
    ),
)