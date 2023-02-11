import os
import numpy as np
import gradio as gr
import subprocess
from huggingface_hub import model_info, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from modules import scripts, script_callbacks 

def download_model(_repo_id,_folder,_filename,_token,_cache_dir):
    try:
        repo_exists = True
        r_info = model_info(_repo_id, token=_token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
    if _folder == "":
        final_file_name=_filename
    else:    
        final_file_name=_folder+'/'+_filename
    hf_hub_download(repo_id=_repo_id,token=_token,resume_download=True,cache_dir=_cache_dir,
                filename=final_file_name,
                force_filename=_filename,
                )
    info_ret=_folder+'/'+_filename+" done. ->"+_cache_dir
    print(info_ret)            
    return info_ret

def exec_cmd(_command):
    process = subprocess.Popen(_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()

    output = process.stdout.read().decode(encoding="utf-8")
    error = process.stderr.read().decode(encoding="utf-8")
    result = {"output": output, "error": error}
    print("exec_cmd:"+str(result))
    if output == '':
        if error  == '':
            return "(no output)"
        return "error: \n"+str(error)
    return str(output)

def on_ui_tabs():     
    with gr.Blocks() as huggingface:
        gr.Markdown(
        """
        ### Example
        download_model(repo_id,folder,filename,token)  
        repo_id=SmallTotem/reserved  
        folder=Uncategorized *(optional)*  
        filename=xxx.ckpt  
        token=hf_xxx *(optional)*  
        target_dir=/content/stable-diffusion-webui/models/Stable-diffusion/ *(colab)*  
                  =/kaggle/working/stable-diffusion-webui/models/Stable-diffusion/ *(kaggle)*  
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_repo_id = gr.Textbox(show_label=False,value="SmallTotem/reserved", max_lines=1, placeholder="repo_id")
                    text_folder = gr.Textbox(show_label=False,value="Uncategorized", max_lines=1, placeholder="folder")
                    text_filename = gr.Textbox(show_label=False,value="", max_lines=1, placeholder="filename")
                    text_token = gr.Textbox(show_label=False,value="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", max_lines=1, placeholder="🤗token")
                with gr.Row().style(equal_height=True):
                    text_target_dir = gr.Textbox(show_label=False,value="/content/stable-diffusion-webui/models/Stable-diffusion/", max_lines=1, placeholder="target_dir")
                with gr.Row().style(equal_height=True):
                    btn_download = gr.Button("download")
                with gr.Row().style(equal_height=True):
                    out_file = gr.Textbox(show_label=False)
        gr.Markdown(
        """
        ### Command
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_cmd = gr.Textbox(show_label=False, max_lines=1, placeholder="command")
                with gr.Row().style(equal_height=True):
                    btn_exec = gr.Button("execute")
                with gr.Row().style(equal_height=True):
                    out_cmd = gr.Textbox(show_label=False)
                    
        btn_download.click(download_model, inputs=[text_repo_id, text_folder, text_filename, text_token,text_target_dir], outputs=out_file)
        btn_exec.click(exec_cmd,inputs=[text_cmd],outputs=out_cmd)
    return (huggingface, "Hugging Face", "huggingface"),
script_callbacks.on_ui_tabs(on_ui_tabs)
