import os
import datetime
import gradio as gr
import subprocess
from huggingface_hub import model_info, hf_hub_download, upload_file
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from modules import scripts, script_callbacks 

#auto select base dir
base_dir=""
print(os.getcwd())
if "stable-diffusion-webui" in os.getcwd():
    base_dir = "/content/stable-diffusion-webui/"
else:
    base_dir = "/content/SD_reserved/"

model_dir_colab=base_dir+"models/Stable-diffusion/"
lora_dir_colab=base_dir+"models/Lora/"
hypernetworks_dir_colab=base_dir+"models/hypernetworks/"
VAE_dir_colab=base_dir+"models/VAE/"

def update_path():
    global model_dir_colab
    global lora_dir_colab
    global hypernetworks_dir_colab
    global VAE_dir_colab
    model_dir_colab = base_dir+"models/Stable-diffusion/"
    lora_dir_colab = base_dir+"models/Lora/"
    hypernetworks_dir_colab = base_dir+"models/hypernetworks/"
    VAE_dir_colab = base_dir+"models/VAE/"


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
    #command_log
    with open("./extensions/stable-diffusion-webui-huggingface/command_log.txt", "a") as f:
        f.write("["+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"] "+_command+"\n")

    if _dir == "":
        _command_final=_command
    else: 
        if not os.path.exists(_dir):
            print("warning: dir created. ("+_dir+")")
            os.makedirs(_dir)
        _command_final="cd "+_dir+"&&"+_command
    process = subprocess.Popen(_command_final, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()

    output = process.stdout.read().decode(encoding="utf-8")
    error = process.stderr.read().decode(encoding="utf-8")
    result = {"output": output, "error": error}
    print("exec_cmd: "+_command)
    print("-> out: "+str(result))
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
    repo_folder=base_dir+"repo_info/"+str(repo_id_name)
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

def fn_radio_set_model_path(choice):
    if choice == "model":
        return gr.Textbox.update(visible=True,value=model_dir_colab)
    elif choice == "lora":
        return gr.Textbox.update(visible=True,value=lora_dir_colab)
    elif choice == "hypernetworks":
        return gr.Textbox.update(visible=True,value=hypernetworks_dir_colab)
    elif choice == "VAE":
        return gr.Textbox.update(visible=True,value=VAE_dir_colab)
    
def set_base_dir(choice):
    global base_dir
    if choice == "SD_reserved":
        base_dir = "/content/SD_reserved/"
    elif choice == "stable-diffusion-webui":
        base_dir = "/content/stable-diffusion-webui/"
    update_path()
    
def fn_radio_set_base_dir(choice):
    set_base_dir(choice)
    return gr.Textbox.update(visible=True, value=base_dir)


def fn_btn_get_model_1():
    return download_model("swl-models/mix-pro-v3","","mix-pro-v3.safetensors","",model_dir_colab)
def fn_btn_get_model_2():
    return download_model("andite/pastel-mix", "", "pastelmix.safetensors", "", model_dir_colab)
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
    exec_cmd(lora_dir_colab, "curl -Lo \"Azhiichigo_Style(Civitai-78206).safetensors\" https://civitai.com/api/download/models/82989")
    exec_cmd(lora_dir_colab, "curl -Lo \"sakurapion_Style(Civitai-87420).safetensors\" https://civitai.com/api/download/models/93027")
    exec_cmd(lora_dir_colab, "curl -Lo \"治愈系插画(Civitai-86596).safetensors\" https://civitai.com/api/download/models/92103")
    exec_cmd(lora_dir_colab, "curl -Lo \"国风插画(Civitai-84527).safetensors\" https://civitai.com/api/download/models/93169")
    exec_cmd(lora_dir_colab, "curl -Lo \"发光星星(Civitai-84532).safetensors\" https://civitai.com/api/download/models/89869")
    return "done."
def fn_btn_get_model_10():
    exec_cmd(lora_dir_colab, "curl -Lo \"上倉エク_Style(Civitai-17305).safetensors\" https://civitai.com/api/download/models/30030")
    exec_cmd(hypernetworks_dir_colab, "curl -Lo \"京田画风(Civitai-5356).pt\" https://civitai.com/api/download/models/6225")
    exec_cmd(lora_dir_colab, "curl -Lo \"剧毒少女画风(Civitai-23623).safetensors\" https://civitai.com/api/download/models/28217")
    exec_cmd(lora_dir_colab, "curl -Lo \"鬼猫_Style(Civitai-63326).safetensors\" https://civitai.com/api/download/models/67870")
    exec_cmd(lora_dir_colab, "curl -Lo \"korukoruno_style(Civitai-55938).safetensors\" https://civitai.com/api/download/models/60333")
    exec_cmd(lora_dir_colab, "curl -Lo \"dropkun_style(Civitai-59214).safetensors\" https://civitai.com/api/download/models/63662")
    return "done."
def fn_btn_get_model_11():
    download_model("sp8999/test_VAE", "", "mse840000_klf8anime.vae.pt", "", VAE_dir_colab)
    download_model("Norisuke193/kl-f8-anime2", "", "kl-f8-anime2.vae.pt", "", VAE_dir_colab)
    return "done."
def fn_btn_get_model_12():
    exec_cmd(lora_dir_colab, "curl -Lo \"優子鈴_style(Civitai-92285).safetensors\" https://civitai.com/api/download/models/98384")
    exec_cmd(lora_dir_colab, "curl -Lo \"サブロー_style_v2(Civitai-101075)\" https://civitai.com/api/download/models/108207")
    exec_cmd(lora_dir_colab, "curl -Lo \"MISSILE228_style(Civitai-60298).safetensors\" https://civitai.com/api/download/models/64771")
    exec_cmd(lora_dir_colab, "curl -Lo \"Bison倉鼠_style(Civitai-45010).safetensors\" https://civitai.com/api/download/models/61440")
    exec_cmd(lora_dir_colab, "curl -Lo \"rurudo_style_v2(Civitai-60868).safetensors\" https://civitai.com/api/download/models/96876")
    return "done."


def fn_btn_ls_model_dir():
    return exec_cmd(model_dir_colab, "ls")
def fn_btn_ls_lora_dir():
    return exec_cmd(lora_dir_colab, "ls")
def fn_btn_cat_kaggle_log():
    return exec_cmd("", "cat "+base_dir+"out.log")
def fn_btn_update_cache():
    return exec_cmd("", "cp -f "+base_dir+"cache.json /content/drive/MyDrive/novelai_script/NovelAI_WEBUI/cache.json")

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
                    radio_base_dir = gr.Radio(["SD_reserved", "stable-diffusion-webui"], label="base_dir")
                    text_base_dir = gr.Textbox(
                        value=base_dir, max_lines=1, placeholder="base_dir", label="base_dir")
                with gr.Row().style(equal_height=True):
                    radio_model_type_1 = gr.Radio(["model", "lora", "hypernetworks","VAE"], label="model_type")
                    text_target_dir = gr.Textbox(
                        value=lora_dir_colab, max_lines=1, placeholder="target_dir", label="model_dir")
                with gr.Row().style(equal_height=True):
                    btn_download = gr.Button("download")
                with gr.Row().style(equal_height=True):
                    out_file = gr.Textbox(show_label=False)
                with gr.Row().style(equal_height=True):
                    btn_get_model_1 = gr.Button("mix-pro-v3")
                    btn_get_model_2 = gr.Button("pastelmixNoVAE")
                    btn_get_model_3 = gr.Button("Counterfeit-V2.5")
                    btn_get_model_4 = gr.Button("9527")
                    btn_get_model_5 = gr.Button("chilloutmix-Ni")
                    btn_get_model_6 = gr.Button("animevae")
                with gr.Row().style(equal_height=True):
                    btn_get_model_7 = gr.Button("ChilloutNight")
                    btn_get_model_8 = gr.Button("chilled_re-generic_v3")
                    btn_get_model_9 = gr.Button("画风lora集#3")
                    btn_get_model_10 = gr.Button("画风lora集#1")
                    btn_get_model_11 = gr.Button("klf8animeVAE")
                    btn_get_model_12 = gr.Button("画风lora集#2")
        btn_download.click(download_model, inputs=[text_repo_id, text_folder, text_filename, text_token,text_target_dir], outputs=out_file)

        radio_model_type_1.change(fn=fn_radio_set_model_path, inputs=radio_model_type_1, outputs=text_target_dir)
        radio_base_dir.change(fn=fn_radio_set_base_dir,inputs=radio_base_dir, outputs=text_base_dir)
    
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
        download_from_civitai: curl -Lo "(Civitai-).safetensors"
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    radio_model_type_2 = gr.Radio(["model", "lora", "hypernetworks", "VAE"], label="model_type")
                    text_cmd_dir = gr.Textbox(value=lora_dir_colab, max_lines=1, placeholder="context_dir", label="context_dir")
                with gr.Row().style(equal_height=True):
                    text_cmd = gr.Textbox(value="curl -Lo \"(Civitai-).safetensors\"", max_lines=1, placeholder="command", label="command")
                with gr.Row().style(equal_height=True):
                    btn_exec = gr.Button("execute")
                with gr.Row().style(equal_height=True):
                    out_cmd = gr.Textbox(show_label=False)
                with gr.Row().style(equal_height=True):
                    btn_ls_model_dir = gr.Button("ls_model_dir")
                    btn_ls_lora_dir = gr.Button("ls_lora_dir")
                    btn_cat_kaggle_log = gr.Button("cat_kaggle_log")
                    btn_update_cache = gr.Button("update_cache")
        btn_exec.click(exec_cmd,inputs=[text_cmd_dir,text_cmd],outputs=out_cmd)

        radio_model_type_2.change(fn=fn_radio_set_model_path, inputs=radio_model_type_2, outputs=text_cmd_dir)

        btn_ls_model_dir.click(fn_btn_ls_model_dir,inputs=[],outputs=out_cmd)
        btn_ls_lora_dir.click(fn_btn_ls_lora_dir, inputs=[], outputs=out_cmd)
        btn_cat_kaggle_log.click(fn_btn_cat_kaggle_log,inputs=[],outputs=out_cmd)
        btn_update_cache.click(fn_btn_update_cache,inputs=[],outputs=out_cmd)


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
