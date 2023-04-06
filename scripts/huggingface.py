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

def download_file(_file_path):
    return _file_path

def fn_btn_get_repo_info(_user,_repo_id,_token):
    repo_id_name=_repo_id.replace("/","_")
    repo_folder="/content/stable-diffusion-webui/repo_info/"+str(repo_id_name)
    if _token == "" or _user == "":
        git_command="GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/"+str(_repo_id)
    else:
        git_command="GIT_LFS_SKIP_SMUDGE=1 git clone https://"+str(_user)+":"+str(_token)+"@huggingface.co/"+str(_repo_id)
    if not os.path.exists(repo_folder):
        os.makedirs(repo_folder)
        exec_cmd(repo_folder,git_command)
    else:
        exec_cmd(repo_folder,"git pull")
    return exec_cmd(repo_folder,"tree")

model_dir_colab="/content/stable-diffusion-webui/models/Stable-diffusion/"
lora_dir_colab="/content/stable-diffusion-webui/models/Lora/"
hypernetworks_dir_colab="/content/stable-diffusion-webui/models/hypernetworks/"
VAE_dir_colab="/content/stable-diffusion-webui/models/VAE/"

def fn_radio_set_model_path(choice):
    if choice == "model":
        return gr.Textbox.update(visible=True,value=model_dir_colab)
    elif choice == "lora":
        return gr.Textbox.update(visible=True,value=lora_dir_colab)
    elif choice == "hypernetworks":
        return gr.Textbox.update(visible=True,value=hypernetworks_dir_colab)
    elif choice == "VAE":
        return gr.Textbox.update(visible=True,value=VAE_dir_colab)

def fn_btn_get_model_1():
    return download_model("swl-models/mix-pro-v3","","mix-pro-v3.safetensors","",model_dir_colab)
def fn_btn_get_model_2():
    return download_model("AgraFL/RefSlave-V2","","RefSlave-V2.safetensors","",model_dir_colab)
def fn_btn_get_model_3():
    return download_model("gsdf/Counterfeit-V2.5","","Counterfeit-V2.5.safetensors","",model_dir_colab)
def fn_btn_get_model_4():
    return download_model("swl-models/9527","","9527.safetensors","",model_dir_colab)
def fn_btn_get_model_5():
    return download_model("swl-models/chilloutmix-ni","","chilloutmix-Ni.safetensors","",model_dir_colab)
def fn_btn_get_model_6():
    return download_model("gsdf/Counterfeit-V2.5","","Counterfeit-V2.5.vae.pt","",VAE_dir_colab)
def fn_btn_get_model_7():
    return download_model("swl-models/ChilloutNight","","ChilloutNight.safetensors","",model_dir_colab)
def fn_btn_get_model_8():
    return exec_cmd(model_dir_colab,"curl -Lo \"chilled_re-generic_v3.safetensors\" https://civitai.com/api/download/models/22033")
def fn_btn_get_model_9():
    return exec_cmd(model_dir_colab,"curl -Lo \"fantasticmix_real_v2.0.safetensors\" https://civitai.com/api/download/models/30145")
def fn_btn_get_model_10():
    exec_cmd(lora_dir_colab,"curl -Lo \"ruru.Certainty(Civitai-21109).safetensors\" https://civitai.com/api/download/models/25124")
    exec_cmd(lora_dir_colab,"curl -Lo \"上倉エク_Style(Civitai-17305).safetensors\" https://civitai.com/api/download/models/29525")
    exec_cmd(hypernetworks_dir_colab,"curl -Lo \"京田画风(Civitai-5356).pt\" https://civitai.com/api/download/models/6225")
    exec_cmd(lora_dir_colab,"curl -Lo \"剧毒少女画风(Civitai-23623).safetensors\" https://civitai.com/api/download/models/28217")
    exec_cmd(lora_dir_colab,"curl -Lo \"玉之けだま_style(Civitai-26224).safetensors\" https://civitai.com/api/download/models/31398")
    return "done."
def fn_btn_get_model_11():
    return download_model("sp8999/test_VAE","","mse840000_klf8anime.vae.pt","",VAE_dir_colab)
def fn_btn_get_model_12():
    return download_model("luizC/blueberry","","blueberrymix_10.safetensors","",model_dir_colab)


def fn_btn_ls_model_dir():
    return exec_cmd(model_dir_colab,"ls")
def fn_btn_update_cache():
    return exec_cmd("","cp -f /content/stable-diffusion-webui/cache.json /content/drive/MyDrive/novelai_script/NovelAI_WEBUI/cache.json")
def fn_btn_cat_kaggle_log():
    return exec_cmd("","cat /content/stable-diffusion-webui/out.log")
def fn_btn_ls_kaggle_working():
    return exec_cmd("/kaggle/working/","ls")


def on_ui_tabs():     
    with gr.Blocks() as huggingface:
        gr.Markdown(
        """
        ### download model
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_repo_id = gr.Textbox(show_label=False,value="SmallTotem/reserved", max_lines=1, placeholder="repo_id")
                    text_folder = gr.Textbox(show_label=False,value="Lora-real", max_lines=1, placeholder="folder")
                    text_filename = gr.Textbox(show_label=False,value="", max_lines=1, placeholder="filename")
                    text_token = gr.Textbox(show_label=False,type="password", max_lines=1, placeholder="🤗token")
                with gr.Row().style(equal_height=True):
                    radio = gr.Radio(["model", "lora", "hypernetworks","VAE"], label="model_type")
                with gr.Row().style(equal_height=True):
                    text_target_dir = gr.Textbox(show_label=False,value="/content/stable-diffusion-webui/models/Lora/", max_lines=1, placeholder="target_dir")
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
                    btn_get_model_6 = gr.Button("animevae")
                with gr.Row().style(equal_height=True):
                    btn_get_model_7 = gr.Button("ChilloutNight")
                    btn_get_model_8 = gr.Button("chilled_re-generic_v3")
                    btn_get_model_9 = gr.Button("fantasticmix_real_v2.0")
                    btn_get_model_10 = gr.Button("画风lora集")
                    btn_get_model_11 = gr.Button("mse840000_klf8anime")
                    btn_get_model_12 = gr.Button("BlueberryMix")
        btn_download.click(download_model, inputs=[text_repo_id, text_folder, text_filename, text_token,text_target_dir], outputs=out_file)

        radio.change(fn=fn_radio_set_model_path, inputs=radio, outputs=text_target_dir)
    
        btn_get_model_1.click(fn_btn_get_model_1, inputs=[], outputs=out_file)
        btn_get_model_2.click(fn_btn_get_model_2, inputs=[], outputs=out_file)
        btn_get_model_3.click(fn_btn_get_model_3, inputs=[], outputs=out_file)
        btn_get_model_4.click(fn_btn_get_model_4, inputs=[], outputs=out_file)
        btn_get_model_5.click(fn_btn_get_model_5, inputs=[], outputs=out_file)
        btn_get_model_6.click(fn_btn_get_model_6, inputs=[], outputs=out_file)
        btn_get_model_7.click(fn_btn_get_model_7, inputs=[], outputs=out_file)
        btn_get_model_8.click(fn_btn_get_model_8, inputs=[], outputs=out_file)
        btn_get_model_9.click(fn_btn_get_model_9, inputs=[], outputs=out_file)
        btn_get_model_10.click(fn_btn_get_model_10, inputs=[], outputs=out_file)
        btn_get_model_11.click(fn_btn_get_model_11, inputs=[], outputs=out_file)
        btn_get_model_12.click(fn_btn_get_model_12, inputs=[], outputs=out_file)

        gr.Markdown(
        """
        ### get repo info
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_repo_info_user = gr.Textbox(show_label=False,value="SmallTotem", max_lines=1, placeholder="user")
                    text_repo_info_repo_id = gr.Textbox(show_label=False,value="SmallTotem/reserved", max_lines=1, placeholder="repo_id")
                    text_repo_info_token = gr.Textbox(show_label=False,type="password", max_lines=1, placeholder="🤗token")
                    btn_get_repo_info = gr.Button("get_info")
                with gr.Row().style(equal_height=True):
                    out_repo_info = gr.Textbox(show_label=False)
        btn_get_repo_info.click(fn_btn_get_repo_info, inputs=[text_repo_info_user,text_repo_info_repo_id,text_repo_info_token], outputs=out_repo_info)

        gr.Markdown(
        """
        ### command
        download_file: curl -Lo "*filename*"  *url*
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
                with gr.Row().style(equal_height=True):
                    btn_ls_model_dir = gr.Button("ls_model_dir")
                    btn_update_cache = gr.Button("update_cache")
                    btn_cat_kaggle_log = gr.Button("cat_kaggle_log")
                    btn_ls_kaggle_working = gr.Button("ls_kaggle_working")
        btn_exec.click(exec_cmd,inputs=[text_cmd_dir,text_cmd],outputs=out_cmd)

        btn_ls_model_dir.click(fn_btn_ls_model_dir,inputs=[],outputs=out_cmd)
        btn_update_cache.click(fn_btn_update_cache,inputs=[],outputs=out_cmd)
        btn_cat_kaggle_log.click(fn_btn_cat_kaggle_log,inputs=[],outputs=out_cmd)
        btn_ls_kaggle_working.click(fn_btn_ls_kaggle_working,inputs=[],outputs=out_cmd)

        gr.Markdown(
        """
        ### push model
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

        gr.Markdown(
        """
        ### download file -> local
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_download_file_path = gr.Textbox(show_label=False, max_lines=1, placeholder="download_file_path")
                    btn_check_file = gr.Button("check")
                with gr.Row().style(equal_height=True):
                    file_download_file = gr.File()
        btn_check_file.click(download_file, inputs=[text_download_file_path], outputs=file_download_file)
    return (huggingface, "Hugging Face", "huggingface"),
script_callbacks.on_ui_tabs(on_ui_tabs)
