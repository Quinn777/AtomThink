from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def qwen2_vl(model_args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.infer_dtype,
        device_map="auto"
    )
    return model

model_map = {
    "qwen2_vl": qwen2_vl,
}