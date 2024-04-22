"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0
python3 -m fastchat.serve.cli --model ~/model_weights/vicuna-7b
"""

import time
import csv
from itertools import zip_longest
import torch

timestart=time.time()

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

import os
import shutil
import sys

shutil.move("/home/lbastien/RadioLLM/radiollm/inference_radiollm.py", "/home/lbastien/miniconda3/lib/python3.10/site-packages/fastchat/serve/inference_radiollm.py")

from fastchat.model.model_adapter import add_model_args
from fastchat.serve.inference_radiollm import ChatIO, chat_loop_test, chat_loop_anev

vicuna = '/data/stockage/bastien/models/vicuna-13b/'
vicuna1_3 = '/data/stockage/bastien/models/lmsys_vicuna-13b-v1.3/'
vicuna1_5 = '/data/stockage/bastien/models/lmsys_vicuna-13b-v1.5-16k/'

vicuna33_1_3='/data/stockage/bastien/models/lmsys_vicuna-33b-v1.3/'

medalpaca = '/data/stockage/bastien/models/medalpaca_medalpaca-13b/'
koala = '/data/stockage/bastien/models/vicuna-13b/'
mistral_7b= '/data/stockage/bastien/models/mistralai_Mistral-7B-Instruct-v0.2/'
mistral_54b= '/data/stockage/bastien/models/mistralai_Mixtral-8x7B-Instruct-v0.1/'


f_indic='/home/lbastien/liste_cr_SAU.txt'
f_interpret='/home/lbastien/liste_CR_interpret.txt'
f_finding='/home/lbastien/liste_CR_finding.txt'
f_simple='/home/lbastien/liste_cr_simple.txt'
f_test='/home/lbastien/cort16_pur.txt'
f_results='/home/lbastien/liste_CR_results.txt'
f_anapath='/home/lbastien/liste_cr_anatotal.txt'
f_oxf='/home/lbastien/Glomerule/liste_cr_iga.txt'
f_berden='/home/lbastien/Glomerule/liste_cr_berden_comp.txt'
f_lupus='/home/lbastien/Glomerule/liste_cr_lupus.txt'
f_gliome='/home/lbastien/irm_gliome_2.txt'
f_gliome_indic='/home/lbastien/irm_gliome_3.txt'
f_german='/home/lbastien/liste_CR_Deutsch.txt'
f_anapath_de='/home/lbastien/liste_CR_anatotal_de.txt'
f_pdl='/home/lbastien/liste_cr_pdl_test.txt'
f_lupus_class='/home/lbastien/Glomerule/liste_cr_lupus_class.txt'
f_250='/home/lbastien/random_250.txt'
f_text='/home/lbastien/liste_CR_text.txt'
f_rev='/home/lbastien/liste_revision.txt'
f_rev_contrast='/home/lbastien/liste_revision_contrast.txt'
f_rev_normal='/home/lbastien/liste_revision_normal.txt'




zero_shots =[
['Doctor', '''Your task is to list all the postitive findings of the report I will present you. One curcial rule is that you must ignore all the negative ("Pas de") or normal ("normale") findings. You must repond according to this template : "ID : [Patient's ID] - Positive findings : [Complete list of positive findings in the report. Please be sure to include every sentence in the report EXCEPT the ones beginning with the words "Pas de", or including the word "normale"] - Any positive finding ? : [your list include any positive finding ? Yes or No. "IRM cérébrale normale" is NOT a positive finding !]".'''], 
['Chatbot', "I will list all the positive findings of the report and ignore all the negative ones."]
]


explain_report=[
['Doctor', '''I will present you a radiology report. I want you to simplify it.

Here are a few examples:
Doctor: Diagnostic anatomopathologique : Diagnostic anatomopathologique : Adénocarcinome infiltrant. TTF1+ ALK- ROS1- PDL1+ (marquage de 60 % des cellules tumorales).
Robot: - Pathological diagnosis: Adénocarcinome infiltrant
- Percentage of PD-L1 cells: 90%
 
Doctor: Type histopathologique selon la classification internationale 2021 : Adénocarcinome infiltrant majoritairement de type acineux avec contingent micropapillaire (5%)Absence d’expression de ALK, ROS1 ou PD-L1 par les cellules tumorales.
Robot: - Pathological diagnosis: Adénocarcinome
- Percentage of PD-L1 cells: 0%

Doctor: Diagnostic anatomopathologique : Carcinome épidermoïde infiltrant. PDL1 en cours.
Robot: - Pathological diagnosis: Carcinome épidermoïde infiltrant
- Percentage of PD-L1 cells: NA
 
Doctor: Diagnostic anatomopathologique : adénocarcinome infiltrant à prédominance acineuse. Mise en évidence d’une expression de la protéine PD-L1 dans 1% des cellules tumorales.
Robot: - Pathological diagnosis: adénocarcinome
- Percentage of PD-L1 cells: 1%
 
Doctor: Anticorps anti-PDL1 : Carcinome non à petites cellules dont les caractéristiques morphologiques évoquent un carcinome sarcomatoïde pléomorphes avec un contingent adénocarcinomateuxabsence d’expression au niveau des cellules tumorales. Il n’est pas mis en évidence d’expression de la protéine PD-L1 par les cellules
Robot: - Pathological diagnosis: carcinome sarcomatoïde pléomorphes
- Percentage of PD-L1 cells: 0%
 ''']
]


class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_log=outputs["logprobs"]
            
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        indexlist=[]
        for i in range(0,len(output_log["tokens"])-1):
            if output_log["tokens"][i]=="/":
                indexlist.append(i)
        for index in indexlist:
            print(output_log["tokens"][index+1])
            print(output_log["token_logprobs"][index+1])
        return " ".join(output_text)

class SimpleChatIO_log(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        log_token=[]
        log_prob=[]
        for outputs in output_stream:
            output_text = outputs["text"]
            output_log=outputs["logprobs"]
            
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        indexlist=[]
        for i in range(0,len(output_log["tokens"])-1):
            if output_log["tokens"][i]=="/":
                indexlist.append(i)
        for index in indexlist:
            log_token.append(output_log["tokens"][index+1])
            log_prob.append(output_log["token_logprobs"][index+1])
        return " ".join(output_text),[log_token,log_prob]

    def print_output(self, text: str):
        print(text)


Chatio=SimpleChatIO_log()


d= chat_loop_test(model_path=vicuna1_5,device='cuda',num_gpus=3,max_gpu_memory='90Gib',dtype=None,load_8bit=False,cpu_offloading=False,conv_template="vicuna_v1.1",conv_system_msg="You are a robot dedicated to helping a Doctor. The doctor will ask you to analyze a text and extract information from it. You will make a short, structured reponse using the template the Doctor will provide. The following first message from the Doctor explains the task and the template to use.",temperature=0,repetition_penalty=1,max_new_tokens=16000,chatio=Chatio,debug=False,few_shots=few_shots_interpret_normal,file_path=f_interpret)

export_data = zip_longest(*d, fillvalue = '')

with open('/home/lbastien/anev.csv', 'w', newline='') as myfile:
   wr = csv.writer(myfile)

   wr.writerows(export_data)
myfile.close()
