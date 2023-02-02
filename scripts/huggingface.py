import os
import numpy as np
import gradio as gr
from huggingface_hub import model_info, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from modules import scripts, script_callbacks 

def download_model(_repo_id,_folder,_filename,_token):
    try:
        repo_exists = True
        r_info = model_info(_repo_id, token=_token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
    if _folder is "":
        final_file_name=_filename
    else:    
        final_file_name=_folder+'/'+_filename
    hf_hub_download(repo_id=_repo_id,token=_token,resume_download=True,cache_dir="/content/stable-diffusion-webui/models/Stable-diffusion/",
                filename=_folder+'/'+_filename,
                force_filename=_filename,
                )
    print(_folder+'/'+_filename+" done.")            
    return _folder+'/'+_filename+" done."


def on_ui_tabs():     
    with gr.Blocks() as huggingface:
        gr.Markdown(
        """
        ### Example
        download_model(repo_id,folder,filename,token)  
        repo_id=SmallTotem/reserved  
        folder=Hiraka  
        filename=Hiraka_Lolipop.safetensors  
        token=...
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_repo_id = gr.Textbox(show_label=False,value="SmallTotem/reserved", max_lines=1, placeholder="repo_id")
                    text_folder = gr.Textbox(show_label=False,value="Uncategorized", max_lines=1, placeholder="folder")
                    text_filename = gr.Textbox(show_label=False, max_lines=1, placeholder="filename")
                    text_token = gr.Textbox(show_label=False,value="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", max_lines=1, placeholder="🤗 token")
                    out_file = gr.Textbox(show_label=False)
                with gr.Row().style(equal_height=True):
                    btn_download = gr.Button("download")
            btn_download.click(download_model, inputs=[text_repo_id, text_folder, text_filename, text_token], outputs=out_file)
    return (huggingface, "Hugging Face", "huggingface"),
script_callbacks.on_ui_tabs(on_ui_tabs)
