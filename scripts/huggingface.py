import os
import numpy as np
import gradio as gr
import subprocess
from huggingface_hub import model_info, hf_hub_download, upload_file
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
    info_ret=_folder+'/'+_filename+" done. -> "+_cache_dir
    print(info_ret)            
    return info_ret

def exec_cmd(_dir,_command):
    if _dir == "":
        _command_final=_command
    else: 
        _command_final="cd "+_dir+"&&"+_command
    process = subprocess.Popen(_command_final, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

def push_file(_file, _path_in_repo, _repo_id, _token):
    upload_file(path_or_fileobj=_file, path_in_repo=_path_in_repo, repo_id=_repo_id, token=_token)
    return "done."   

def on_ui_tabs():     
    with gr.Blocks() as huggingface:
        gr.Markdown(
        """
        ### Download
        download_model(repo_id,folder,filename,token)  
        repo_id= SmallTotem/reserved  
        folder= Uncategorized *(optional)*  
        filename= xxx.safetensors  
        token= hf_xxx *(optional)*  
        target_dir= /content/stable-diffusion-webui/models/Stable-diffusion/ *(colab)*  
        target_dir= /content/stable-diffusion-webui/models/Lora/ *(colab,lora)*   
        target_dir= /kaggle/stable-diffusion-webui/models/Stable-diffusion/ *(kaggle)*  
        target_dir= /kaggle/stable-diffusion-webui/models/Lora/ *(kaggle,lora)*  
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_repo_id = gr.Textbox(show_label=False,value="SmallTotem/reserved", max_lines=1, placeholder="repo_id")
                    text_folder = gr.Textbox(show_label=False,value="Uncategorized", max_lines=1, placeholder="folder")
                    text_filename = gr.Textbox(show_label=False,value="", max_lines=1, placeholder="filename")
                    text_token = gr.Textbox(show_label=False,type="password", max_lines=1, placeholder="🤗token")
                with gr.Row().style(equal_height=True):
                    text_target_dir = gr.Textbox(show_label=False,value="/content/stable-diffusion-webui/models/Stable-diffusion/", max_lines=1, placeholder="target_dir")
                with gr.Row().style(equal_height=True):
                    btn_download = gr.Button("download")
                with gr.Row().style(equal_height=True):
                    out_file = gr.Textbox(show_label=False)
                with gr.Row().style(equal_height=True):
                    btn_get_model_1 = gr.Button("mix-pro-v3")
                    btn_get_model_2 = gr.Button("RefSlave-V2")
                    btn_get_model_3 = gr.Button("Counterfeit-V2.5")
                    btn_get_model_4 = gr.Button("9527")
                    btn_get_model_5 = gr.Button("chilloutmix-Ni")
        btn_download.click(download_model, inputs=[text_repo_id, text_folder, text_filename, text_token,text_target_dir], outputs=out_file)
        dir_colab="/content/stable-diffusion-webui/models/Stable-diffusion/"
        btn_get_model_1.click(download_model, inputs=["swl-models/mix-pro-v3","","mix-pro-v3.safetensors","",dir_colab], outputs=out_file)
        btn_get_model_2.click(download_model, inputs=["AgraFL/RefSlave-V2","","RefSlave-V2.safetensors","",dir_colab], outputs=out_file)
        btn_get_model_3.click(download_model, inputs=["gsdf/Counterfeit-V2.5","","Counterfeit-V2.5.safetensors","",dir_colab], outputs=out_file)
        btn_get_model_4.click(download_model, inputs=["swl-models/9527","","9527.safetensors","",dir_colab], outputs=out_file)
        btn_get_model_5.click(download_model, inputs=["swl-models/chilloutmix-ni","","chilloutmix-Ni.safetensors","",dir_colab], outputs=out_file)
        gr.Markdown(
        """
        ### Command
        update_cache: cp -f /content/stable-diffusion-webui/cache.json /content/drive/MyDrive/novelai_script/NovelAI_WEBUI/cache.json  
        kaggle_ls: /kaggle/working  &&  ls  
        download_file: curl -Lo "`filename`"  `url`
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_cmd_dir = gr.Textbox(show_label=False,value="/content", max_lines=1, placeholder="context_dir")
                with gr.Row().style(equal_height=True):
                    text_cmd = gr.Textbox(show_label=False, max_lines=1, placeholder="command")
                with gr.Row().style(equal_height=True):
                    btn_exec = gr.Button("execute")
                with gr.Row().style(equal_height=True):
                    out_cmd = gr.Textbox(show_label=False)
        btn_exec.click(exec_cmd,inputs=[text_cmd_dir,text_cmd],outputs=out_cmd)
        gr.Markdown(
        """
        ### Push
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_file = gr.Textbox(show_label=False, max_lines=1, placeholder="file")
                    text_repo_id_2 = gr.Textbox(show_label=False, max_lines=1,value="SmallTotem/reserved", placeholder="repo_id")
                    text_path_in_repo = gr.Textbox(show_label=False, max_lines=1, placeholder="path_in_repo")
                    text_token_2 = gr.Textbox(show_label=False,type="password", max_lines=1, placeholder="🤗token")
                    out_file_push = gr.Textbox(show_label=False)
                with gr.Row().style(equal_height=True):
                    btn_push_file = gr.Button("push")
        btn_push_file.click(push_file, inputs=[text_file, text_path_in_repo, text_repo_id_2, text_token_2], outputs=out_file_push)
    return (huggingface, "Hugging Face", "huggingface"),
script_callbacks.on_ui_tabs(on_ui_tabs)
