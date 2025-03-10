from transformers import AutoTokenizer, AutoProcessor
def llava(question: str, modality: str, **kwargs):
    assert modality == "image"
    imgnum = len(kwargs["images"])
    img_prompt = "<image>\n" * imgnum
    prompt = f"USER: {img_prompt}{question}\nASSISTANT:"
    return prompt

def mllama(question: str, modality: str, **kwargs):
    assert modality == "image"
    tokenizer = kwargs["tokenizer"]
    imgnum = len(kwargs["images"])
    if imgnum > 0:
        messages = [{
            "role":
                "user",
            "content": [{
                "type": "image"
            }, {
                "type": "text",
                "text": f"{question}"
            }]
        }]
    else:
        messages = [{
            "role":
                "user",
            "content": [{
                "type": "text",
                "text": f"{question}"
            }]
        }]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
    return prompt

conversation_map = {
    "llava": llava,
    "mllama": mllama
}