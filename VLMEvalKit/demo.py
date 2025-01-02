import gradio as gr
from vlmeval.vlm.internvl.internvl_chat import InternVLChat

PATH='/data/wenhu/work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_res_finetune_full_long_cot'

model = InternVLChat(model_path=PATH, version='V2.0', load_in_8bit=False, use_mpo_prompt=False)

def answer_question(image, question):
    # Perform the inference
    message = [dict(type='text', value=question), dict(type='image', value=image)]
    outputs = model.generate(message)
    return outputs

iface = gr.Interface(
    fn=answer_question,
    inputs=[gr.Textbox(label="Image Path"), gr.Textbox(label="Question")],
    outputs=gr.Textbox(label="Answer"),
    title="Visual Question Answering",
    description="Upload an image and ask a question related to the image. The AI will try to answer it."
)

iface.launch(share=True)
