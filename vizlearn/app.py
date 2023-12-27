import gradio as gr
import requests
from PIL import Image
from transformers import pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
import subprocess

subprocess.Popen("apt install -y tesseract-ocr", shell=True)

def describe_image(raw_image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=1000)
    return f'This is a picture of {processor.decode(out[0], skip_special_tokens=True)}'


def image_recognition(raw_image):
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    inputs = processor(raw_image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]


def document_answering(image_url, question):
    nlp = pipeline(
        "document-question-answering",
        model="impira/layoutlm-document-qa",
    )
    return nlp(image_url, question)

def image_answering(image, question):
    # image = Image.open(requests.get(image_url, stream=True).raw)

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, question, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]


with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")

    with gr.Tab("Document Answering"):
        with gr.Row():
            with gr.Column():
                image_input_1 = gr.Image(type="filepath")
                text_input_1 = gr.Textbox()
            text_output_1 = gr.Textbox()
        button_1 = gr.Button("Ask something about this document!")

    with gr.Tab("Image Answering"):
        with gr.Row():
            with gr.Column():
                image_input_2 = gr.Image()
                text_input_2 = gr.Textbox()
            text_output_2 = gr.Textbox()
        button_2 = gr.Button("Ask something about this image!")

    with gr.Tab("Describe Image"):
        with gr.Row():
            image_input_3 = gr.Image()
            text_output_3 = gr.Textbox()
        button_3 = gr.Button("Describe Image!")

    with gr.Tab("Object Detection"):
        with gr.Row():
            image_input_4 = gr.Image()
            text_output_4 = gr.Textbox()
        button_4 = gr.Button("What's in the image?")

    button_1.click(fn=document_answering, inputs=[image_input_1, text_input_1], outputs=text_output_1)
    button_2.click(fn=image_answering, inputs=[image_input_2, text_input_2], outputs=text_output_2)
    button_3.click(fn=describe_image, inputs=image_input_3, outputs=text_output_3)
    button_4.click(fn=image_recognition, inputs=image_input_4, outputs=text_output_4)

# demo = gr.Interface(fn=describe_image, inputs=gr.Image(), outputs="text")
demo.launch()

