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

few_shots_finding_gold26623 =[
['Doctor', '''Your task is to list all the findings of radiology reports and tell me if at least one is abnormal. By abnormal, I mean any sort of lesion, including insignificant ones. Don't forget, stable or regressing findings are abnormal findings! First, list all the sentences of the report. Then, tell me if they describe a normal or an abnormal finding. Finally, tell me if your list contains any abnormal finding. If there is at least one abnormal finding in your list, you must respond Yes. You must respond only according to this template : ID Patient : [Patient's ID] - All findings : [List of all findings (normal or abnormal)] - Any positive finding ? : [If any abnormal finding in your list, you MUST respond Yes, otherwise: No]'''], 
['Chatbot', "I will list all the positive findings of the reports you will provide. If my list includes any abnormal finding, I will answer Yes to your question."],

['Doctor', "ID :13813406.0 : Absence d'argument pour une dissection des troncs supra-aortiques. Pas d'argument en faveur d'une TVC."],
['Chatbot', "ID Patient : 13813406.0 - All findings : Absence d'argument pour une dissection des troncs supra-aortiques (normal) ; Pas d'argument en faveur d'une TVC (normal) - Any positive finding ? : No"],

['Doctor', "ID :12145521.0 Pas d'argument pour une thrombose veineuse intracrânienne. Stabilité des signes d'hypotension du LCR. Aspect inchangé de la pachyméningite réactionnelle des convexités."],
['Chatbot',"ID Patient : 12145521.0 - All findings : Pas d'argument pour une thrombose veineuse intracrânienne (normal) ; Stabilité des signes d'hypotension du LCR (abnormal) ; Aspect inchangé de la pachyméningite réactionnelle des convexités (abnormal) - Any positive finding ? Yes"],

['Doctor', "ID :19934970.0 : Pas de lésion ischémique ou hémorragique récente. Comblement sinusien maxillaire gauche à confronter aux données cliniques."],
['Chatbot', "ID Patient : 19934970.0 - All findings : Pas de lésion ischémique ou hémorragique récente (normal) ; Comblement sinusien maxillaire gauche à confronter aux données cliniques (abnormal) - Any positive finding ? Yes"],

['Doctor', "ID :14111449.0 : Absence d'anomalie expliquant la symptomatologie, notamment pas de thrombose veineuse cébérale."],
['Chatbot', "ID Patient : 14111449.0 - All findings : Absence d'anomalie expliquant la symptomatologie, notamment pas de thrombose veineuse cébérale (normal) - Any positive finding ? No"],

['Doctor', "ID :15478898.0 : Perméabilité des sinus veineux et des troncs supra-aortiques. Absence de signe en faveur d'une hypertension intracrânienne."],
['Chatbot', "ID Patient : 15478898.0 - All findings : Perméabilité des sinus veineux et des troncs supra-aortiques (normal) ; Absence de signe en faveur d'une hypertension intracrânienne (normal) - Any positive finding ? No"],

['Doctor', "ID :14978298.0 : Stabilité de la petite collection hématique sous-durale gauche sans remaniements aigus, sans effet de masse sur le parenchyme cérébral adjacent."],
['Chatbot', "ID Patient : 14978298.0 - All findings : Stabilité de la petite collection hématique sous-durale gauche sans remaniements aigus, sans effet de masse sur le parenchyme cérébral adjacent (abnormal) - Any positive finding ? Yes"],

['Doctor', "ID :14576698.0 Absence de thrombose veineuse cérébrale. IRM cérébrale sans particularité. Perméabilité des artères du polygone de Willis."],
['Chatbot', "ID Patient : 14576698.0 - All findings : Absence de thrombose veineuse cérébrale (normal) ; IRM cérébrale sans particularité (normal) ; Perméabilité des artères du polygone de Willis (normal) - Any positive finding ? No"]
]

few_shots16 =[
['Doctor', '''Your task is to list all the postitive findings of the reports I will present you. One curcial rule is that you must ignore all the negative ("Pas de") or normal ("IRM normale") findings. You must respond only according to this template, without adding or removing any entry : "ID : [Patient's ID] - Positive findings : [Complete list of positive findings in the report. Please be sure to include every sentence in the report EXCEPT the ones beginning with the words "Pas de" or describing the finding as normal] - Any positive finding ? : [Does your list include any positive finding ? "Yes" or "No"]"'''], 
['Chatbot', "I will list all the positive findings of the reports you will provide."],

['Doctor', "ID :1444526.0 Leucopathie vasculaire Fazekas 1. Atypie veineuse de développement."],
['Chatbot', "ID Patient : 14164326.0 - Positive findings : Leucopathie vasculaire Fazekas 1 ; Atypie veineuse de développement - Any positive finding ? : Yes"],

['Doctor', "ID :14823406.0 Pas d'argument pour une dissection des troncs supra-aortiques. Pas de lésion ischémique ou hémorragique récente. "],
['Chatbot', "ID Patient : 14823406.0 - Positive findings : None - Any positive finding ? : No"],

['Doctor',"ID :15073927.0 : Pas de processus expansif intracrânien ou de thrombose veineuse intracérébrale. Signe d'hypotension du LCR associant une pachyméningite réactionnelle des convexités, un bombement des sinus veineux intracrâniens, des collections liquidiennes épidurales."],
['Chatbot',"ID Patient : 15073927.0 - Positive findings : Signe d'hypotension du LCR ; pachyméningite réactionnelle des convexités ; bombement des sinus veineux intracrâniens ; collections liquidiennes épidurales - Any positive finding ? Yes"],

['Doctor', "ID :14984971.0 : Pas de lésion ischémique ou hémorragique récente. Comblement sinusien maxillaire gauche à confronter aux données cliniques."],
['Chatbot', "ID Patient : 14984971.0 - Positive findings : Comblement sinusien maxillaire gauche - Any positive finding ? Yes"],

['Doctor', "ID :14852449.0 : Pas de lésion ischémique ou hémorragique récente. Lésions de la substance blanche périventriculaire et juxtacorticale à l'étage sus-tentoriel compatibles avec des lésions inflammatoires démyélinisantes remplissant les critères de dissémination spatiale selon Mc Donald 2017. Les critères de dissémination spatiale ne sont pas atteints. Absence de thrombose veineuse cérébrale. Pas de lésion des nerfs optiques ou intraorbitaire."],
['Chatbot', "ID Patient : 14852449.0 - Positive findings : Multiples lésions inflammatoires présentant une prise de contraste franche - Any positive finding ? Yes"],

['Doctor', "ID :14478898.0 : Perméabilité des artères cérébrales et des troncs supra-aortiques."],
['Chatbot', "ID Patient : 14852449.0 - Positive findings : None - Any positive finding ? No"],

['Doctor', "ID :14978298.0 : IRM cérébrale normale. Pas de thrombose veineuse cérébrale."],
['Chatbot', "ID Patient : 14978298.0 - Positive findings : None - Any positive finding ? No"],

['Doctor', "ID :14576698.0 : IRM cérébrale sans particularité."],
['Chatbot', "ID 14576698 : 14852449.0 - Positive findings : None - Any positive finding ? No"],

['Doctor', "ID :12243641.0 Perméabilité des sinus veineux intracrâniens."],
['Chatbot', "ID Patient : 12243641.0 - Positive findings : None - Any positive finding ? : No"],

['Doctor', "ID :16468790.0 Pas d'argument pour une dissection des troncs supra-aortiques. Pas de lésion ischémique ou hémorragique récente. PROFESSEUR PHILIPPE PETIT, Jamel EL RAZAK (interne)"],
['Chatbot', "ID Patient : 16468790.0 - Positive findings : None - Any positive finding ? : No"],

['Doctor',"ID :17999876.0 : Aspect inchangé de la leucopathie marquée pour l'âge."],
['Chatbot',"ID Patient : 17999876.0 - Positive findings : Leucopathie marquée pour l'âge - Any positive finding ? Yes"],

['Doctor', "ID :13466441.0 : Pas de lésion ischémique ou hémorragique récente. Séquelle ischémique frontale droite connue, stable."],
['Chatbot', "ID Patient : 13466441.0 - Positive findings : Séquelle ischémique frontale droite - Any positive finding ? Yes"],

['Doctor', "ID :19964624.0 : IRM cérébrale et des TSA normale. ARM veineuse sans particularité."],
['Chatbot', "ID Patient : 19964624.0 - Positive findings : None - Any positive finding ? No"],

['Doctor', "ID :13463463.0 : Aspect IRM de progression lésionnelle."],
['Chatbot', "ID Patient : 13463463.0 - Positive findings : Progression lésionnelle - Any positive finding ? Yes"],

['Doctor', "ID :10024561.0 : Aspect compatible avec une stabilité lésionnelle."],
['Chatbot', "ID Patient : 10024561.0 - Positive findings : Stabilité lésionnelle - Any positive finding ? Yes"],

['Doctor', "ID :19786540.0 : Perméabilité des vaisseaux du cou. Pas de thrombose veineuse cérébrale."],
['Chatbot', "ID 14576698 : 19786540.0 - Positive findings : None - Any positive finding ? No"]
]

zero_shot_indic =[
['</s>Doctor', '''I will present you short indications for radiology exams. Your task is to list all the symptoms from the text and tell me if they correspond to headache (tag them with [/headache]) or to another symptom (tag them with [/other]). If there are no symptoms in the indication (for example "Suspicion de..."), just respond 'No symptom'. Keep to the text only. You will answer only according to this template (replacing the words between brackets with your answer): 
Symptoms from INDICATION : 
- Symptom : [retrieve the symptom] [tag with /headache only if the symptom is headache, else tag with /other]
etc
It is crucial that you respond only to symptoms from the text !'''],
['Robot', "I am ready to respond according to the template. I understand that these are real patients and I will not create any new data."]
]

few_shots_finding_normal =[
['</s>Doctor', '''I will present you conclusions from radiology exams. Your task is to list all the findings from them and tell me if they correspond to normal or abnormal findings. Keep in mind, an abnormal finding may be described as stable or regressing, so please be sure not to miss any abnormal finding. Each conclusion is independant and should not influence your answer. You will answer only according to this template: 
List of findings : 
- List all the findings /normal or abnormal finding ?
Attention : Include only findings from the text and no not invent any data ! '''],
['Robot', "I am ready to respond according to the template."],

['Doctor', "Conclusion : Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent. Séquelle ischémique thalamique gauche. Absence de thrombose veineuse cérébrale. IRM encéphalique sans anomalie par ailleurs."],
['Robot', '''List of findings : 
- Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent /abnormal
- Séquelle ischémique thalamique gauche /abnormal
- Absence de thrombose veineuse cérébrale /normal
- IRM encéphalique sans anomalie par ailleurs /normal'''],

['Doctor', "Conclusion : Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques (sinusite ?). Doute sur un aspect un peu fin du sinus veineux latéral droit, peu spécifique et sans retentissement d'amont. Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent."],
['Robot', '''List of findings : 
- Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques (sinusite ?) /abnormal
- Doute sur un aspect un peu fin du sinus veineux latéral droit, peu spécifique et sans retentissement d'amont /abnormal
- Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent /abnormal'''],

['Doctor', "Conclusion : Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative. Pas de lésion ischémique ou hémorragique récente."],
['Robot', '''List of findings : 
- Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative /abnormal
- Pas de lésion ischémique ou hémorragique récente /normal'''],

['Doctor', "Conclusion : Pas d'anomalie expliquant la symptomatologie vertigineuse. Stabilité des signes d'hypertension intracrânienne."],
['Robot', '''List of findings : 
- Pas d'anomalie expliquant la symptomatologie vertigineuse /normal
- Stabilité des signes d'hypertension intracrânienne /abnormal''']
]

few_shots_finding_fr =[
['Docteur', '''Je vais te présenter la conclusion d'un compte-rendy de radiologie. Je veux que tu listes tous les résultats du texte et que tu m'indiques s'ils sont normaux ou anormaux. N'oublie pas que des résultats anormaux incluent aussi des résultats stables ou en régression, donc ne les loupe pas ! Tu doi srépondre en suivant cette structure : 
Liste des résultats : 
- Liste tous les résultats [/normal] ou [/anormal] ?
 
Voici quelques exemples de ce que j'attends de toi (tu es le Robot et je suis le Docteur):

Docteur : Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent. Séquelle ischémique thalamique gauche. Absence de thrombose veineuse cérébrale. IRM encéphalique sans anomalie par ailleurs.
Robot : Liste des résultats :
- Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent /anormal
- Séquelle ischémique thalamique gauche /anormal
- Absence de thrombose veineuse cérébrale /normal
- IRM encéphalique sans anomalie par ailleurs /normal

Docteur : Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques (sinusite ?). Doute sur un aspect un peu fin du sinus veineux latéral droit, peu spécifique et sans retentissement d'amont. Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent.
Robot : Liste des résultats :
- Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques (sinusite ?) /anormal
- Doute sur un aspect un peu fin du sinus veineux latéral droit, peu spécifique et sans retentissement d'amont /anormal
- Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent /anormal

Docteur : Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative. Pas de lésion ischémique ou hémorragique récente.
Robot : Liste des résultats :
- Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative /anormal
- Pas de lésion ischémique ou hémorragique récente /normal

Docteur : Pas d'anomalie expliquant la symptomatologie vertigineuse. Stabilité des signes d'hypertension intracrânienne.
Robot : Liste des résultats :
- Pas d'anomalie expliquant la symptomatologie vertigineuse /normal
- Stabilité des signes d'hypertension intracrânienne /anormal

 
Tu es prêt pour le compte-rendu ?'''],
['Robot', "je suis prêt à répondre de manière structurée."],
]

zero_shot_finding_normal=[
['</s>Doctor', '''I will present you conclusions from radiology exams. Your task is to list all the findings from them and tag them as "/normal" or "/abnormal" findings. Keep in mind, an abnormal finding may be described as stable or regressing, so please be sure not to miss any abnormal finding. Each conclusion is independant and should not influence your answer. You will answer only according to this template: 
List of findings : 
- Finding 1 [insert 1st finding of the report here] /normal or abnormal ?
- Finding 2 [insert 2nd finding of the report here] /normal or abnormal ?
etc until you listed all the findings'''],
['Robot', "I am ready to respond according to the template."]]

two_shot_finding_normal=[
['</s>Doctor', '''I will present you conclusions from radiology exams. Your task is to list all the findings from them and tag them as "/normal" or "/abnormal" findings. Keep in mind, an abnormal finding may be described as stable or regressing, so please be sure not to miss any abnormal finding. Each conclusion is independant and should not influence your answer. You will answer only according to this template: 
List of findings : 
- Finding 1 [insert 1st finding of the report here] /normal or abnormal ?
- Finding 2 [insert 2nd finding of the report here] /normal or abnormal ?
etc until you listed all the findings'''],
['Robot', "I am ready to respond according to the template."],

['Doctor', "Conclusion : Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques. Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont. Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent."],
['Robot', '''List of findings : 
- Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques /abnormal
- Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont /abnormal
- Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent /abnormal'''],

['Doctor', "Conclusion : Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative. Perméabilité des sinus veineux intracrâniens."],
['Robot', '''List of findings : 
- Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative /abnormal
- Perméabilité des sinus veineux intracrâniens /normal''']
]

four_shot_finding_normal=[
['</s>Doctor', '''I will present you conclusions from radiology exams. Your task is to list all the findings from them and tell me if they correspond to normal or abnormal findings. Keep in mind, an abnormal finding may be described as stable or regressing, so please be sure not to miss any abnormal finding. Each conclusion is independant and should not influence your answer. You will answer only according to this template: 
List of findings : 
- List all the findings /normal or abnormal finding ?
Attention : Include only findings from the text and no not invent any data ! '''],
['Robot', "I am ready to respond according to the template."],

['Doctor', "Conclusion : Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent. Absence de thrombose veineuse cérébrale. IRM encéphalique sans anomalie par ailleurs."],
['Robot', '''List of findings : 
- Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent /abnormal
- Absence de thrombose veineuse cérébrale /normal
- IRM encéphalique sans anomalie par ailleurs /normal'''],

['Doctor', "Conclusion : Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques. Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont. Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent."],
['Robot', '''List of findings : 
- Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques /abnormal
- Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont /abnormal
- Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent /abnormal'''],

['Doctor', "Conclusion : Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative. Perméabilité des sinus veineux intracrâniens."],
['Robot', '''List of findings : 
- Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative /abnormal
- Perméabilité des sinus veineux intracrâniens /normal'''],

['Doctor', "Conclusion : Pas d'anomalie expliquant la symptomatologie vertigineuse. Stabilité des signes d'hypertension intracrânienne."],
['Robot', '''List of findings : 
- Pas d'anomalie expliquant la symptomatologie vertigineuse /normal
- Stabilité des signes d'hypertension intracrânienne /abnormal''']]

six_shots_finding_normal=[
['Doctor', '''I will present you a Conclusion from a radiology exam. Your task is to list all the findings from it and to tell me if they correspond to normal or abnormal findings. Keep in mind, an abnormal finding may be described as stable or regressing. You will answer only according to this template: 
List of findings : 
- List all the findings /normal or abnormal finding ?

Here are two examples of what I expect from you:

Conclusion : 
Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent. Absence de thrombose veineuse cérébrale. IRM encéphalique sans anomalie par ailleurs.
Your expected response :
- Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent /abnormal
- Absence de thrombose veineuse cérébrale /normal
- IRM encéphalique sans anomalie par ailleurs /normal
 
Conclusion : Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques. Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont. Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent.
Your expected response :
- Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques /abnormal
- Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont /abnormal
- Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent /abnormal

Conclusion : Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent. Absence de thrombose veineuse cérébrale. IRM encéphalique sans anomalie par ailleurs.
Your expected response :
- Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent /abnormal
- Absence de thrombose veineuse cérébrale /normal
- IRM encéphalique sans anomalie par ailleurs /normal
 
Conclusion : Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques. Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont. Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent.
Your expected response :
- Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques /abnormal
- Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont /abnormal
- Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent /abnormal'''],
['Robot', "I am ready to respond according to the template."]]


eight_shots_finding_normal=[['Doctor', '''I will present you conclusions from radiology exams. Your task is to list all the findings from them and tell me if they correspond to normal or abnormal findings. Keep in mind, an abnormal finding may be described as stable or regressing. You will answer only according to this template: 
List of findings : 
- List all the findings /normal or abnormal finding ?

Here are 4 examples of what I expect:
                             
'Doctor', "Conclusion : Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent. Absence de thrombose veineuse cérébrale. IRM encéphalique sans anomalie par ailleurs.",
'Robot', List of findings : 
- Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent /abnormal
- Absence de thrombose veineuse cérébrale /normal
- IRM encéphalique sans anomalie par ailleurs /normal,

'Doctor', "Conclusion : Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques. Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont. Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent.",
'Robot', List of findings : 
- Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques /abnormal
- Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont /abnormal
- Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent /abnormal,

'Doctor', "Conclusion : Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative. Perméabilité des sinus veineux intracrâniens.",
'Robot', List of findings : 
- Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative /abnormal
- Perméabilité des sinus veineux intracrâniens /normal,

'Doctor', "Conclusion : Pas de modification des deux cavernomes sans atypie.",
'Robot', List of findings : 
- Pas de modification des deux cavernomes sans atypie /abnormal''']]

eight_shots_finding_normal=[
['</s>Doctor', '''I will present you conclusions from radiology exams. Your task is to list all the findings from them and tell me if they correspond to normal or abnormal findings. Keep in mind, an abnormal finding may be described as stable or regressing. Each conclusion is independant and should not influence your answer. You will answer only according to this template: 
List of findings : 
- List all the findings /normal or abnormal finding ?
Attention : Include only findings from the text and no not invent any data ! '''],
['Robot', "I am ready to respond according to the template."],

['Doctor', "Conclusion : Absence de thrombose veineuse cérébrale. IRM encéphalique sans anomalie par ailleurs."],
['Robot', '''List of findings : 
- Absence de thrombose veineuse cérébrale /normal
- IRM encéphalique sans anomalie par ailleurs /normal'''],

['Doctor', "Conclusion : Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent."],
['Robot', '''List of findings : 
- Diminution en taille et évolution attendue de la collection hématique sous-durale de la convexité gauche, sans effet de masse sur le parenchyme adjacent /abnormal'''],

['Doctor', "Conclusion : Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques. Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont."],
['Robot', '''List of findings : 
- Aspect inchangé du discret épaississement muqueux en cadre du sinus maxillaire droit, à confronter aux données cliniques /abnormal
- Doute sur un aspect un peu fin du sinus veineux latéral droit, sans retentissement d'amont /abnormal'''],

['Doctor', "Conclusion : Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent. IRM sans particularité."],
['Robot', '''List of findings : 
- Pas de modfication des atypies veineuses de développement connues, sans signe de remaniement hémorragique récent /abnormal
- IRM sans particularité /normal'''],

['Doctor', "Conclusion : Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative. Perméabilité des sinus veineux intracrâniens."],
['Robot', '''List of findings : 
- Stabilité de la dissection de l'artère vertébrale gauche dans sa portion V3, sans sténose significative /abnormal
- Perméabilité des sinus veineux intracrâniens /normal'''],

['Doctor', "Conclusion : Pas de modification des deux cavernomes sans atypie."],
['Robot', '''List of findings : 
- Pas de modification des deux cavernomes sans atypie /abnormal'''],

['Doctor', "Conclusion : Pas d'argument en faveur d'une détérioration des signes IRM de neuro-Behçet. IRM cérébrale normale par ailleurs."],
['Robot', '''List of findings : 
- Pas d'argument en faveur d'une détérioration des signes IRM de neuro-Behçet /abnormal
- IRM cérébrale normale par ailleurs /normal'''],

['Doctor', "Conclusion : Pas d'anomalie expliquant la symptomatologie vertigineuse. Stabilité des signes d'hypertension intracrânienne."],
['Robot', '''List of findings : 
- Pas d'anomalie expliquant la symptomatologie vertigineuse /normal
- Stabilité des signes d'hypertension intracrânienne /abnormal''']]

few_shots_interpret_normal_gold0507 =[
['</s>Doctor', '''I will present you abnormal findings from radiology exams. Your task is to list all the findings from them and tell me if they usually cause headache or not. Each conclusion is independant and should not influence your answer. You will answer only according to this template: 
List of findings : 
- List all the findings /can cause headache or not ?'''],
['Robot', "I am ready to respond according to the template. I understand that these are real patients and I will not create any new data."],

['Doctor', "Conclusion : Absence de signe de thrombose veineuse cérébrale. Stabilité des signes d'hypertension intracrânienne."],
['Robot', '''List of findings : 
- Absence de signe de thrombose veineuse cérébrale /no headache
- Stabilité des signes d'hypertension intracrânienne /can cause headache'''],

['Doctor', "Conclusion : Hétérotopie périventriculaire pariétale gauche. IRM encéphalique normale par ailleurs."],
['Robot', '''List of findings : 
- Hétérotopie périventriculaire pariétale gauche /no headache
- IRM encéphalique normale par ailleurs /no headache'''],

['Doctor', "Conclusion : Lésion en hypersignal B1000 visible en FLAIR du cortex précentral gauche, compatible avec une origine ischémique en première hypothèse. Perméabilité du polygone de Willis."],
['Robot', '''List of findings : 
- Lésion en hypersignal B1000 visible en FLAIR du cortex précentral gauche, compatible avec une origine ischémique en première hypothèse /can cause headache
- Perméabilité du polygone de Willis /no headache'''],

['Doctor', "Conclusion : Volumineuse masse intracrânienne responsable d'un engagement sous-tentoriel."],
['Robot', '''List of findings : 
- Volumineuse masse intracrânienne responsable d'un engagement sous-tentoriel /can cause headache'''],

['Robot', '''List of findings : 
- Comblement en cadre des sinus maxillaires compatible avec une sinusite /can cause headache''']
]

shots_finding_fr =[
['Doctor', '''Je vais te présenter un compte-rendu de radiologie. Ta tâche est de les lister ses résultats et de me dire s'ils sont responsables de céphalées ou non. Les lésions graves causent des céphalées, les lésions chroniques ou stables ne causent pas de céphalées. Chaque compte-rendu est indépendant. Tu dois répondre uniquement selon cette structure: 
Liste des résultats : 
- Liste tous les résultats [/peut caDoctor des céphalées] ou [/pas de céphalées]
 
Voici des exemples :
 
Exemple 1 : Absence de signe de thrombose veineuse cérébrale. Stabilité des signes d'hypertension intracrânienne.
Liste des résultats : 
- Absence de signe de thrombose veineuse cérébrale /pas de céphalées
- Stabilité des signes d'hypertension intracrânienne /peut caDoctor des céphalées

Exemple 2 : Hétérotopie périventriculaire pariétale gauche. Leucopathie vasculaire Fazekas 3. IRM encéphalique normale par ailleurs.
Liste des résultats : 
- Hétérotopie périventriculaire pariétale gauche /pas de céphalées
- Leucopathie vasculaire Fazekas 3 /pas de céphalées
- IRM encéphalique normale par ailleurs /pas de céphalées

Exemple 3 : Lésion en hypersignal B1000 visible en FLAIR du cortex précentral gauche, compatible avec une origine ischémique en première hypothèse. Perméabilité du polygone de Willis.
Liste des résultats : 
- Lésion en hypersignal B1000 visible en FLAIR du cortex précentral gauche, compatible avec une origine ischémique en première hypothèse /peut caDoctor des céphalées
- Perméabilité du polygone de Willis /pas de céphalées
 
Exemple 4 : Volumineuse masse intracrânienne responsable d'un engagement sous-tentoriel.
Liste des résultats : 
- Volumineuse masse intracrânienne responsable d'un engagement sous-tentoriel /peut caDoctor des céphalées

Exemple 5 : Hypersignaux FLAIR de la substance blanche dont le caractère radiaire peut faire évoquer une origine inflammatoire. Prise de contraste des léptoméningée (ponction lombaire ? HTIC ?).
Liste des résultats : 
- Hypersignaux FLAIR de la substance blanche dont le caractère radiaire peut faire évoquer une origine inflammatoire /peut caDoctor des céphalées
- Prise de contraste des léptoméningée : ponction lombaire ou HTIC /peut caDoctor des céphalées
 
Exemple 6 : Comblement en cadre des sinus maxillaires compatible avec une sinusite. Multiples anévrysmes intracrâniens tels que décrits : 8mm de la terminaison basillaire, 5mm de la birfurcation sylvienne gauche 
Liste des résultats : 
- Comblement en cadre des sinus maxillaires compatible avec une sinusite /peut caDoctor des céphalées
- Anévrysme de 8mm de la terminaison basillaire /pas de céphalées
- Anévrysme de 5mm de la birfurcation sylvienne /pas de céphalées

Es-tu prêt à répondre ?'''],
['Robot', "Je suis prêt à répondre de manière structurée."]
]

few_shots_interpret_normal =[
['</s>Doctor', '''I will present you abnormal findings from radiology exams. Your task is to list all the findings from them and tell me if they usually cause headache or not. Each conclusion is independant and should not influence your answer. You will answer only according to this template: 
List of findings : 
- List all the findings /can cause headache or not ?'''],
['Robot', "I am ready to respond according to the template. I understand that these are real patients and I will not create any new data."],

['Doctor', "Conclusion : Absence de signe de thrombose veineuse cérébrale. Stabilité des signes d'hypertension intracrânienne."],
['Robot', '''List of findings : 
- Absence de signe de thrombose veineuse cérébrale /no headache
- Stabilité des signes d'hypertension intracrânienne /can cause headache'''],

['Doctor', "Conclusion : Hétérotopie périventriculaire pariétale gauche. IRM encéphalique normale par ailleurs."],
['Robot', '''List of findings : 
- Hétérotopie périventriculaire pariétale gauche /no headache
- IRM encéphalique normale par ailleurs /no headache'''],

['Doctor', "Conclusion : Lésion en hypersignal B1000 visible en FLAIR du cortex précentral gauche, compatible avec une origine ischémique en première hypothèse. Perméabilité du polygone de Willis."],
['Robot', '''List of findings : 
- Lésion en hypersignal B1000 visible en FLAIR du cortex précentral gauche, compatible avec une origine ischémique en première hypothèse /can cause headache
- Perméabilité du polygone de Willis /no headache'''],

['Doctor', "Conclusion : Volumineuse masse intracrânienne responsable d'un engagement sous-tentoriel."],
['Robot', '''List of findings : 
- Volumineuse masse intracrânienne responsable d'un engagement sous-tentoriel /can cause headache'''],

['Doctor', "Conclusion : Hypersignaux FLAIR de la substance blanche dont le caractère radiaire peut faire évoquer une origine inflammatoire. Prise de contraste des léptoméningée (ponction lombaire ? HTIC ?)."],
['Robot', '''List of findings : 
- Hypersignaux FLAIR de la substance blanche dont le caractère radiaire peut faire évoquer une origine inflammatoire /can cause headache
- Prise de contraste des léptoméningée : ponction lombaire ou HTIC /can cause headache'''],

['Doctor', "Conclusion : Comblement en cadre des sinus maxillaires compatible avec une sinusite. Multiples anévrysmes intracrâniens tels que décrits : 8mm de la terminaison basillaire, 5mm de la birfurcation sylvienne gauche "],
['Robot', '''List of findings : 
- Comblement en cadre des sinus maxillaires compatible avec une sinusite /can cause headache
- Anévrysme de 8mm de la terminaison basillaire /no headache
- Anévrysme de 5mm de la birfurcation sylvienne /no headache''']]

four_shots_contrast =[
['Doctor', '''I will present you protocoles from radiology reports. I want to you to tell me if contrast media injection was performed (often with gadolinum of gado). Respond only according to this template : 
- Mention of injection : 
- Injected ? : 

'Doctor', "PROTOCOLE : Séquences axiales diffusion, FLAIR, T1 et T2*. Séquence 3D TOF. Après injection séquence ARM veineuse puis T1 axiale SE 144564487 / Mme Né(e) le : JJ/MM/1939 (Age : 83 ans)",
'Robot', - Mention of injection : Après injection 
- Injected ? : Yes, 

'Doctor', "PROTOCOLE : Séquences axiales T1, T2, T2 étoile, flair. 3D fat sat sur les vaisseaux du cou. ARM veineuse et artérielle après injection / Mr Né(e) le : JJ/MM/1988 (Age : 32 ans)",
'Robot', - Mention of injection : après injection
- Injected ? : Yes, 

'Doctor', "PROTOCOLE Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium. Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : JJ/MM/1967 (Age : 55 ans)",
'Robot', - Mention of injection : Séquence ARM Gado ; avec injection de Gadolinium 
- Injected ? : Yes, 

'Doctor', PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. 3D TOF sur les artères du polygone de Willis. / Mme Né(e) le : JJ/MM/1939 (Age : 83 ans)",
'Robot', - Mention of injection : None
- Injected ? : No,

'Doctor', "PROTOCOLE : ARM veineuse complémentaire après injection de godolinium / Mme Né(e) le : JJ/MM/1978 (Age : 44 ans)",
'Robot', - Mention of injection : après injection de godolinium
- Injected ? : Yes, 

'Doctor', "PROTOCOLE : Séquences coro T2 sur les voies optiques, exiale FLAIR et diffusion sur l'encéphale. Après injection, séquence 3D FLAIR sur l'encéphale. / Mr Né(e) le : JJ/MM/2000 (Age : 22 ans)",
'Robot', - Mention of injection : Après injection
- Injected ? : Yes, 

'Doctor', "PROTOCOLE Séquences axiales 3D Gado T1 et axiale diffusion / Mme Né(e) le : JJ/MM/1997 (Age : 25 ans)",
'Robot', - Mention of injection : 3D Gado T1
- Injected ? : Yes, 

'Doctor', "PROTOCOLE Séquences axiales T1 SE, FLAIR, T2* et Diffusion. 3D TOF sur les artères du polygone de Willis. / Mme Né(e) le : JJ/MM/1939 (Age : 83 ans)",
'Robot', - Mention of injection : None
- Injected ? : No''']]

eight_shots_interpret_normal =[
['Doctor', '''I will present you abnormal findings from radiology exams. Your task is to list all the findings from them and tell me if they usually cause headache (tag with /can cause headache) or not (tag with /does not usually cause headache). You will answer only according to this template: 
List of findings : 
- Finding n : [insert here finding number n] - Response : [choose between : /can cause headache and : /does not usually cause headache]
 
 
Here are six examples:
Doctor: Conclusion : Absence de signe de thrombose veineuse cérébrale. Stabilité des signes d'hypertension intracrânienne. Dissection de l'artère carotide interne gauche."],
Robot: List of findings: 
- Absence de signe de thrombose veineuse cérébrale /does not usually cause headache
- Stabilité des signes d'hypertension intracrânienne /can cause headache
- Dissection de l'artère carotide interne gauche /can cause headache

Doctor: Conclusion : Hétérotopie périventriculaire pariétale gauche. Pansinusite.
Robot: List of findings: 
- Hétérotopie périventriculaire pariétale gauche /does not usually cause headache
- Pansinusite /can cause headache

Doctor: Conclusion : Lésions en hypersignal B1000, hypersignal FLAIR, compatibles avec des infarctus constitués. Atypie veineuse de développement frontale droite."],
Robot: List of findings: 
- Lésions en hypersignal B1000, hypersignal FLAIR, compatibles avec des infarctus constitués /can cause headache
- Atypie veineuse de développement frontale droite /does not usually cause headache

Doctor: Conclusion : Image d'addition (anévrisme) communicant antérieur de 3mm.
Robot: List of findings: 
- Image d'addition (anévrisme) communicant antérieur de 3mm /does not usually cause headache

Doctor: Conclusion : Aspect IRM compatible avec une encéphalopathie hépatique.
Robot: List of findings : 
- Aspect IRM compatible avec une encéphalopathie hépatique /can cause headache

Doctor: Conclusion : Volumineuse lésion rehaussée du carrefour ventriculaire droit de 3x2cm. Méningiome para-falcoriel gauche.
Robot: List of findings : 
- Volumineuse lésion rehaussée du carrefour ventriculaire droit de 3x2cm /can cause headache
- Méningiome para-falcoriel gauche /does not usually cause headache ''']
]


eight_shots_interpret_raccourci =[
['Doctor', '''I will present you abnormal findings from radiology exams. Your task is to list all the findings from them and tell me if they usually cause headache (tag with /can cause headache) or not (tag with /does not usually cause headache). You will answer only according to this template: 
List of findings : 
- Finding n : [insert here finding number n] - Response : [choose between : /can cause headache and : /does not usually cause headache]
Any anomaly causing headache: [yes or no] 

 
Here are six examples:
Doctor: Conclusion : Absence de signe de thrombose veineuse cérébrale. Stabilité des signes d'hypertension intracrânienne. Dissection de l'artère carotide interne gauche."],
Robot: List of findings: 
- Absence de signe de thrombose veineuse cérébrale /does not usually cause headache
- Stabilité des signes d'hypertension intracrânienne /can cause headache
- Dissection de l'artère carotide interne gauche /can cause headache
Any anomaly causing headache: yes

Doctor: Conclusion : Hétérotopie périventriculaire pariétale gauche. Pansinusite.
Robot: List of findings: 
- Hétérotopie périventriculaire pariétale gauche /does not usually cause headache
- Pansinusite /can cause headache
Any anomaly causing headache: yes

Doctor: Conclusion : Lésions en hypersignal B1000, hypersignal FLAIR, compatibles avec des infarctus constitués. Atypie veineuse de développement frontale droite."],
Robot: List of findings: 
- Lésions en hypersignal B1000, hypersignal FLAIR, compatibles avec des infarctus constitués /can cause headache
- Atypie veineuse de développement frontale droite /does not usually cause headache
Any anomaly causing headache: yes

Doctor: Conclusion : Image d'addition (anévrisme) communicant antérieur de 3mm.
Robot: List of findings: 
- Image d'addition (anévrisme) communicant antérieur de 3mm /does not usually cause headache
Any anomaly causing headache: no 

Doctor: Conclusion : Aspect IRM compatible avec une encéphalopathie hépatique.
Robot: List of findings : 
- Aspect IRM compatible avec une encéphalopathie hépatique /can cause headache
Any anomaly causing headache: yes

Doctor: Conclusion : Volumineuse lésion rehaussée du carrefour ventriculaire droit de 3x2cm. Méningiome para-falcoriel gauche.
Robot: List of findings : 
- Volumineuse lésion rehaussée du carrefour ventriculaire droit de 3x2cm /can cause headache
- Méningiome para-falcoriel gauche /does not usually cause headache 
Any anomaly causing headache: yes
''']
]




few_shots_interpret9_3 =[
['Doctor', '''I will present you radiology reports with abnormal findings. I want to know if the patient described in the report may present headaches based on the findings from the report. First, I want you to list all the positive findings in the report. Then, explain me briefly the findings like I am in 6th grade. Finally, tell me if what you just described can cause headaches or not. You must respond only according to this template : "ID : [Patient's ID] - Positive findings : [Complete list of positive findings in the report] - Brief explanation : [Explain me briefly what does the finding mean like I am a 6th grader] - Can this finding cause headaches ? : ["Yes" or "No"]"'''], 
['Chatbot', "Understood, I will respond according to the template."],

['Doctor', "ID :1444526.0 Leucopathie vasculaire Fazekas 1. Atypie veineuse de développement."],
['Chatbot', "ID Patient : 14164326.0 - Positive findings : Leucopathie vasculaire Fazekas 1 ; Atypie veineuse de développement - Brief explanation : Vascular leukoencephalopathy is a condition that causes changes in the white matter of the brain. Developmental venous anomaly is a benign unusual vein the the brain - Can this finding cause headaches ? : No"],

['Doctor',"ID :15073927.0 : Pas de processus expansif intracrânien ou de thrombose veineuse intracérébrale. Signe d'hypotension du LCR associant une pachyméningite réactionnelle des convexités, un bombement des sinus veineux intracrâniens, des collections liquidiennes épidurales."],
['Chatbot',"ID Patient : 15073927.0 - Positive findings : Signe d'hypotension du LCR ; pachyméningite réactionnelle des convexités ; bombement des sinus veineux intracrâniens ; collections liquidiennes épidurales - Brief explanation : Cerebrospinal fluid hypotension is a decreased pressure in the fluid that surrounds the brain and the spinal cord. - Can this finding cause headaches ? : Yes"],

['Doctor',"ID :12884679.0 : Lésion ischémique récente pariétale droite."],
['Chatbot',"ID Patient : 12884679.0 - Positive findings : Lésion ischémique pariétale droite - Brief explanation : Ischemic lesions of the brain are infarcted brain lesions. - Can this finding cause headaches ? : Yes"],

['Doctor',"ID :14446654.0 : Majoration en taille de la masse du carrefour ventriculaire droit de 21x17x17mm, rehaussée, avec majoration de l'oedème périlésionnel important."],
['Chatbot',"ID Patient : 14446654.0 - Positive findings : Majoration en taille de la masse du carrefour ventriculaire droit de 21x17x17mm ; majoration de l'oedème périlésionnel important - Brief explanation : The increase in size of the lesion causes a mass effect on the normal brain resulting in edema. - Can this finding cause headaches ? : Yes"],

['Doctor',"ID :12669842.0 : Aspect d'arachnoïdocèle intra-sellaire sans autre signe d'HTIC."],
['Chatbot',"ID Patient : 12669842.0 - Positive findings : Arachnoïdocèle intra-sellaire sans autre signe d'HTIC - Brief explanation : Intrasellar arachnoid cyst is part of the abnormalities found in intracranial hypertension. It is not sure whether or not intracranial hypertension is present here. - Can this finding cause headaches ? : No"],

['Doctor',"ID :13446312.0 : Multiples lésions rehaussées disséminées infra et péricentimétriques évocatrices de lésions secondaires. Oedème périlésionnel. "],
['Chatbot',"ID Patient : 13446312.0 - Positive findings : Multiples lésions rehaussées disséminées infra et péricentimétriques évocatrices de lésions secondaires ; Oedème périlésionnel - Brief explanation : Disseminated intracerebral lesions are suggestive or secondary lesions and responsible of cerebral edema. It is possible that these lesions are brain metastasis. - Can this finding cause headaches ? : Yes"],

['Doctor',"ID :16994231.0 : IRM encéphalique et des TSA normale. Comblement sinusien compatible avec une pansinusite à confronter aux données cliniques. "],
['Chatbot',"ID Patient : 16994231.0 - Positive findings : Comblement sinusien compatible avec une pansinusite à confronter aux données cliniques - Brief explanation : Filling of the sinus is a finding associated with sinusitis, which is sinus infection. It frequently causes fever and headaches. - Can this finding cause headaches ? : Yes"],

['Doctor',"ID :17294211.0 : Aspect de dissection de l'artère carotide interne gauche étendue du 2cm à partir de la bifurcation carotidienne. "],
['Chatbot',"ID Patient : 17294211.0 - Positive findings : Dissection de l'artère carotide interne gauche étendue du 2cm à partir de la bifurcation carotidienne - Brief explanation : Internal carotid artery dissection is a hetoma of the artery wall. It causes cervical pain and headaches and may be an etiology of stroke in young patients. - Can this finding cause headaches ? : Yes"],

['Doctor',"ID :15506031.0 : Pas de lésion ischémique ou hémorragique récente. Prise de contraste intense isolée du nerf optique droit compatible avec une névrite optique droite. "],
['Chatbot',"ID Patient : 15506031.0 - Positive findings : Prise de contraste intense isolée du nerf optique droit compatible avec une névrite optique droite - Brief explanation : Enhancement of the optical nerve may be secondary to optic neuritis. Patients affected by optic neuritis usually present with decreased visual acuity and peri-orbital pain or headaches. - Can this finding cause headaches ? : Yes"],
]

few_shots_simple_questions =[
['Doctor', '''I will present you radiology reports. I want to you to extract 3 features : the exam ID, the patient age and their sex. Ignore all other information, they are insignificant. One crucial rule is that respond only according to this template : 
- ID : [Patient's ID] 
- Sex : [Male if "/ Mr", Female if "/ Mme"] 
- Age : [Age in years]'''], 
['Chatbot', "Understood, I will respond according to the template."],

['Doctor', "ID :15462157.0 PROTOCOLE : Séquences axiales diffusion, FLAIR, T1 et T2*. Séquence 3D TOF. Après injection séquence ARM veineuse puis T1 axiale SE 144564487 / Mme Né(e) le : [JJ]/[MM]/1970 (Age : 41 ans)"],
['Chatbot', """- ID Patient : 15462157.0 
- Sex : Female 
- Age : 51"""],

['Doctor', "ID :19442154.0 PROTOCOLE : Séquences axiales T1, T2, T2 étoile, flair. 3D fat sat sur les vaisseaux du cou. ARM veineuse et artérielle après injection / Mr Né(e) le : [JJ]/[MM]/1996 (Age : 25 ans)"],
['Chatbot', """- ID Patient : 19442154.0 
- Sex : Male 
- Age : 25"""],

['Doctor', "ID :11236554.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. Séquence 3D TOF sur les artères du polygone de Willis. Séquence axiale 3D-SET1 BV sur les artères carotides et vertébrales cervicales. Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium. Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1967 (Age : 55 ans)"],
['Chatbot', """-ID Patient : 11236554.0 
- Sex : Female 
- Age : 55"""],

['Doctor', "ID :10021549.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. 3D TOF sur les artères du polygone de Willis. / Mme Né(e) le : [JJ]/[MM]/1939 (Age : 83 ans)"],
['Chatbot', """- ID Patient : 10021549.0 
- Sex : Female 
- Age : 83"""],

['Doctor', """ID :16654956.0 PROTOCOLE Séquences axiales T1, T2, CORO T2, 3D SWI, et ARM veineuse et 3D T1 SE après injection reconstruite dans les trois plans Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1943 (Age : 79 ans)"],
['Chatbot', "- ID Patient : 16654956.0 
- Sex : Female 
- Age : 79 """]]

zero_shot_contrast =[
['Doctor', '''I will present you protocoles from radiology reports. I want to you to tell me if contrast media injection was performed (often with gadolinum of gado). Ignore all other information, they are insignificant. One crucial rule is that respond only according to this template : 
- Mention of injection : [Retrieve the Mention of injection or gadolinum or gado in the text] 
- Injected ? : [Choose Yes or No]'''], 
['Chatbot', "Understood, I will respond according to the template."]]

few_shots_contrast =[
['Doctor', '''I will present you protocoles from radiology reports. I want to you to tell me if contrast media injection was performed (often with gadolinum of gado). Ignore all other information, they are insignificant. One crucial rule is that respond only according to this template : 
- Mention of injection : [retrieve the Mention of injection or gadolinum or gado in the text] 
- Injected ? : [Yes/No]'''], 
['Chatbot', "Understood, I will respond according to the template."],

['Doctor', "PROTOCOLE : Séquences axiales diffusion, FLAIR, T1 et T2*. Séquence 3D TOF. Après injection séquence ARM veineuse puis T1 axiale SE 144564487"],
['Chatbot', """- Mention of injection : Après injection séquence ARM veineuse puis T1 axiale SE 
- Injected ? : Yes"""], 

['Doctor', "ID :19442154.0 PROTOCOLE : Séquences axiales T1, T2, T2 étoile, flair. 3D fat sat sur les vaisseaux du cou. ARM veineuse et artérielle après injection"],
['Chatbot', """- Mention of injection : ARM veineuse et artérielle après injection
- Injected ? : Yes"""], 

['Doctor', "ID :11236554.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. Séquence 3D TOF sur les artères du polygone de Willis. Séquence axiale 3D-SET1 BV sur les artères carotides et vertébrales cervicales. Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium. Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1967 (Age : 55 ans)"],
['Chatbot', """- Mention of injection : Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium 
- Injected ? : Yes"""], 

['Doctor', "ID :10021549.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. 3D TOF sur les artères du polygone de Willis. / Mme Né(e) le : [JJ]/[MM]/1939 (Age : 83 ans)"],
['Chatbot', """- Mention of injection : None
- Injected ? : No"""], 

['Doctor', """ID :16654956.0 PROTOCOLE Séquences axiales T1, T2, CORO T2, 3D SWI, et ARM veineuse et 3D T1 SE après injection reconstruite dans les trois plans Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1943 (Age : 79 ans)"],
['Chatbot', "- Mention of injection : ARM veineuse et 3D T1 SE après injection
- Injected ? : Yes"""]]

few_shots_contrast =[
['Doctor', '''I will present you protocoles from radiology reports. I want to you to tell me if contrast media injection was performed (often with gadolinum of gado). Ignore all other information, they are insignificant. One crucial rule is that respond only according to this template : 
- Mention of injection : [retrieve the Mention of injection or gadolinum or gado in the text] 
- Injected ? : [Yes/No]'''], 
['Chatbot', "Understood, I will respond according to the template."],

['Doctor', "PROTOCOLE : Séquences axiales diffusion, FLAIR, T1 et T2*. Séquence 3D TOF. Après injection séquence ARM veineuse puis T1 axiale SE 144564487"],
['Chatbot', """- Mention of injection : Après injection séquence ARM veineuse puis T1 axiale SE 
- Injected ? : Yes"""], 

['Doctor', "ID :19442154.0 PROTOCOLE : Séquences axiales T1, T2, T2 étoile, flair. 3D fat sat sur les vaisseaux du cou. ARM veineuse et artérielle après injection"],
['Chatbot', """- Mention of injection : ARM veineuse et artérielle après injection
- Injected ? : Yes"""], 

['Doctor', "ID :11236554.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. Séquence 3D TOF sur les artères du polygone de Willis. Séquence axiale 3D-SET1 BV sur les artères carotides et vertébrales cervicales. Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium. Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1967 (Age : 55 ans)"],
['Chatbot', """- Mention of injection : Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium 
- Injected ? : Yes"""], 

['Doctor', "ID :10021549.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. 3D TOF sur les artères du polygone de Willis. / Mme Né(e) le : [JJ]/[MM]/1939 (Age : 83 ans)"],
['Chatbot', """- Mention of injection : None
- Injected ? : No"""], 

['Doctor', """ID :16654956.0 PROTOCOLE Séquences axiales T1, T2, CORO T2, 3D SWI, et ARM veineuse et 3D T1 SE après injection reconstruite dans les trois plans Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1943 (Age : 79 ans)"],
['Chatbot', "- Mention of injection : ARM veineuse et 3D T1 SE après injection
- Injected ? : Yes"""]]

few_shots_contrast =[
['Doctor', '''I will present you protocoles from radiology reports. I want to you to tell me if contrast media injection was performed (often with gadolinum of gado). Ignore all other information, they are insignificant. One crucial rule is that respond only according to this template : 
- Mention of injection : [retrieve the Mention of injection or gadolinum or gado in the text] 
- Injected ? : [Yes/No]'''], 
['Chatbot', "Understood, I will respond according to the template."],

['Doctor', "PROTOCOLE : Séquences axiales diffusion, FLAIR, T1 et T2*. Séquence 3D TOF. Après injection séquence ARM veineuse puis T1 axiale SE 144564487"],
['Chatbot', """- Mention of injection : Après injection séquence ARM veineuse puis T1 axiale SE 
- Injected ? : Yes"""], 

['Doctor', "ID :19442154.0 PROTOCOLE : Séquences axiales T1, T2, T2 étoile, flair. 3D fat sat sur les vaisseaux du cou. ARM veineuse et artérielle après injection"],
['Chatbot', """- Mention of injection : ARM veineuse et artérielle après injection
- Injected ? : Yes"""], 

['Doctor', "ID :11236554.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. Séquence 3D TOF sur les artères du polygone de Willis. Séquence axiale 3D-SET1 BV sur les artères carotides et vertébrales cervicales. Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium. Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1967 (Age : 55 ans)"],
['Chatbot', """- Mention of injection : Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium 
- Injected ? : Yes"""], 

['Doctor', "ID :10021549.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. 3D TOF sur les artères du polygone de Willis. / Mme Né(e) le : [JJ]/[MM]/1939 (Age : 83 ans)"],
['Chatbot', """- Mention of injection : None
- Injected ? : No"""], 

['Doctor', """ID :16654956.0 PROTOCOLE Séquences axiales T1, T2, CORO T2, 3D SWI, et ARM veineuse et 3D T1 SE après injection reconstruite dans les trois plans Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1943 (Age : 79 ans)"],
['Chatbot', "- Mention of injection : ARM veineuse et 3D T1 SE après injection
- Injected ? : Yes"""]]

six_shots_indic = [['Doctor', '''I will present you indications for radiology exams. Your task is to list all the symptoms from them and tell me if they correspond to headache or to another symptom. If you find no symptom ("Suspicion de ..."), you can answer "No symptom". You will answer only according to this template: 
Symptoms from Indication : 
- List all the symptoms /is it headache or another symptom ? 
'Robot', "I am ready to respond according to the template. I understand that these are real patients and I will not create any new data.

Here are six examples:
'Doctor', Indication Suspicion de NORB gauche.
'Robot', Symptoms from Indication : 
No symptom

'Doctor', Indication Suspicion d'AVC. Déficit neurologique à 16 heures avec dysarthrie et déficit du membre supérieur droit. Notion de troubles de l'équilibre brutaux. Vertiges.
'Robot', Symptoms from Indication : 
- Déficit neurologique /other
- Dysarthrie /other
- Déficit du membre supérieur droit /other
- Notion de troubles de l'équilibre brutaux /other
- Vertiges /other,

'Doctor', Indication : Bilan d'une masse découverte sur un scanner réalisé pour vertige, diplopie et 3e épisode de céphalées.
'Robot', Symptoms from Indication : 
- Vertige /other
- Diplopie /other
- Céphalées /headache,

'Doctor', Indication : Patient de 14 ans ayant été pris en charge début août 2021 pour une régression psychomotrice d'origine indeterminée. Depuis une semaine, vertiges, flou visuel associés à des migraines ne cédant pas aux antalgiques.
'Robot', Symptoms from Indication : 
- Vertiges /other
- Flou visuel /other
- Migraines ne cédant pas aux antalgiques /headache,

'Doctor', Indication : Bilan complémentaire d'un épisode de vomissements en jet avec hémiparésie senstive modérée depuis 48h. Depuis quatre heures, apparition de céphalées intenses pulsatiles sans signe de focalisation ni déficit focal. Suspicion de thrombose veineuse cérébrale.
'Robot', Symptoms from Indication : 
- Vomissements en jet /other 
- Hémiparésie senstive modérée depuis 48h /other
- Céphalées intenses pulsatiles /headache,

'Doctor', Indication : Patiente de 38 ans enceinte à 32 SA (grossesse sans complication) présentant une perte du mot dans un contexte de céphalées atypiques associées à une aura visuelle. Notion d'epistaxis intense il y a deux semaines traitée par embolisation. Suspicion de thrombose veineuse cérébrale.
'Robot', Symptoms from Indication : 
- Perte du mot /other
- Céphalées atypiques associées à une aura visuelle /headache
- Epistaxis intense il y a deux semaines traitée par embolisation /other''']]

few_shots_contrast =[
['Doctor', '''I will present you protocoles from radiology reports. I want to you to tell me if contrast media injection was performed (often with gadolinum of gado). Ignore all other information, they are insignificant. One crucial rule is that respond only according to this template : 
- Mention of injection : [retrieve the Mention of injection or gadolinum or gado in the text] 
- Injected ? : [Yes/No]'''], 
['Chatbot', "Understood, I will respond according to the template."],

['Doctor', "PROTOCOLE : Séquences axiales diffusion, FLAIR, T1 et T2*. Séquence 3D TOF. Après injection séquence ARM veineuse puis T1 axiale SE 144564487"],
['Chatbot', """- Mention of injection : Après injection séquence ARM veineuse puis T1 axiale SE 
- Injected ? : Yes"""], 

['Doctor', "ID :19442154.0 PROTOCOLE : Séquences axiales T1, T2, T2 étoile, flair. 3D fat sat sur les vaisseaux du cou. ARM veineuse et artérielle après injection"],
['Chatbot', """- Mention of injection : ARM veineuse et artérielle après injection
- Injected ? : Yes"""], 

['Doctor', "ID :11236554.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. Séquence 3D TOF sur les artères du polygone de Willis. Séquence axiale 3D-SET1 BV sur les artères carotides et vertébrales cervicales. Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium. Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1967 (Age : 55 ans)"],
['Chatbot', """- Mention of injection : Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium 
- Injected ? : Yes"""], 

['Doctor', "ID :10021549.0 PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. 3D TOF sur les artères du polygone de Willis. / Mme Né(e) le : [JJ]/[MM]/1939 (Age : 83 ans)"],
['Chatbot', """- Mention of injection : None
- Injected ? : No"""], 

['Doctor', """ID :16654956.0 PROTOCOLE Séquences axiales T1, T2, CORO T2, 3D SWI, et ARM veineuse et 3D T1 SE après injection reconstruite dans les trois plans Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1943 (Age : 79 ans)"],
['Chatbot', "- Mention of injection : ARM veineuse et 3D T1 SE après injection
- Injected ? : Yes"""]]

few_shots_contrast_fr =[
['Docteur', '''Je vais te présenter un protocole d'IRM cérébral. Je vaux que tu me dises s'il y a eu injection de produit de contraste (gadolinum ou gado). Les séquences TOF et PC (contraste de phase) ne sont pas injectées. Tu dois répondre sous cette forme : 
- Mention d'injection : [retrouve moi la mention d'injection ou de gado, si elle existe (rappel : les séquences PC ne sont pas injectées)] 
- Injecté ? : [Oui/Non]
 
Voici quelques exemples de ce que j'attends de toi (tu es le Robot, je suis le Docteur) :

Docteur : PROTOCOLE : Séquences axiales diffusion, FLAIR, T1 et T2*. Séquence 3D TOF. Après injection séquence ARM veineuse puis T1 axiale SE 144564487
Robot:
- Mention d'injection : Après injection séquence ARM veineuse puis T1 axiale SE 
- Injecté ? : Oui

Docteur : PROTOCOLE : Séquences axiales T1, T2, T2 étoile, flair. 3D fat sat sur les vaisseaux du cou. ARM veineuse et artérielle après injection
Robot :
- Mention d'injection : ARM veineuse et artérielle après injection
- Injecté ? : Oui

Docteur : PROTOCOLE : Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. Séquence 3D TOF sur les artères du polygone de Willis. Séquence axiale 3D-SET1 BV sur les artères carotides et vertébrales cervicales. Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium. Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1967 (Age : 55 ans)
- Mention d'injection : Séquence ARM Gado avec reconstructions 3D. Séquences axiales T1 sans et avec injection de Gadolinium 
- Injecté ? : Oui

Docteur : PROTOCOLE Séquences axiales FLAIR, T2* et Diffusion; cartographie ADC. 3D TOF sur les artères du polygone de Willis. Séquence 3D PC veineux. / Mme Né(e) le : [JJ]/[MM]/1939 (Age : 83 ans)
Robot : 
- Mention d'injection : None
- Injecté ? : Non 

Docteur : PROTOCOLE Séquences axiales T1, T2, CORO T2, 3D SWI, et ARM veineuse et 3D T1 SE après injection reconstruite dans les trois plans Produits : 1 - CLARISCAN 20 MLCLA20 - 15899963 / Mme Né(e) le : [JJ]/[MM]/1943 (Age : 79 ans)
Robot :
- Mention of injection : ARM veineuse et 3D T1 SE après injection
- Injecté ? : Oui

Tu es prêt pour le procotole ?
Rappelle toi, la séquence "pc" ou "pc-ASL" ou "PC" n'est PAS injectée !'''], 
['Robot', "Oui, je suis prêt à dire s'il y a eu une injection."]
]

few_shots_ventricule =[
['Doctor', '''I am going to show you results from a brain MRI. I want you to tell me if the ventricles are [normal] or [abnormal] or [not mentioned]. I am only interested in the ventricles, do not mention the brain parenchyma. Ventricles are liquid spaces inside the brain. Hydrocepahlus is the enlargement of the ventricles. You must answer ONLY acording to this template :
Ventricles : [sentence mentioning the ventricles (write "none" if no mention of the ventricles in the text)] [normal or abnormal or not mentioned]
 
Here are a few examples of what I expect from you:

Doctor : Absence de lésion ischémique ou hémorragique récente. Absence d'anomalie de signal significative de la substance blanche supra ou infratentorielle. Structures médianes en place. Respect des espaces liquidiens intra et péricérébraux. Pas de prise de contraste pathologique intra ou extra-axiale. Au niveau vasculaire : Perméabilité des artères proximales du polygone de Willis. Perméabilité des sinus veineux intracrâniens et des veines cérébrales internes. 
Robot: Ventricles : Respect des espaces liquidiens intra et péricérébraux. [normal] 

Doctor : Absence d'hypersignal en diffusion évocateur de lésion ischémique récente. Absence d'anomalies de signal en T2* ou d'hétérogénéité du signal évocatrice d'hémorragie. En fosse postérieure, absence d'anomalies morphologiques ou de signal du tronc cérébral, du vermis ou des hémisphères cérébelleux. Le V4 est en place et est de taille normale. Respect des citernes de la base. En sus-tentoriel, Quelques hypersignaux flair de la substance blanche sous corticale et péri-ventriculaire d’allure non spécifique. Intégrité morphologique du système ventriculaire. Ligne médiane en place Prise de contraste du nerf optique droit dans sa portion intra orbitaire, étendue sur 7 mm au niveau de son tiers moyen. Aspect hypertrophique du nerf à ce niveau. Au niveau des vaisseaux cervicaux, Visualisation satisfaisante des carotides et vertébrales. Pas de signe de dissection évident sous réserve des importants artéfacts liés au matériel sur la séquence 3DT1BV 
Robot : Ventricles : Intégrité morphologique du système ventriculaire. [normal]

Doctor : Lésion ischémique récente, bien visible en flair, occipitale interne gauche, dans le territoire supposé de l'artère cérébrale postérieure gauche. Perméabilité du polygone de Willis et de ses branches, on met seulement en évidence quelques rares flux lents corticaux occipitaux gauches. Pas de lésion hémorragique récente. Intégrité morphologique sans anomalie de signal du parenchyme cérébral par ailleurs. Perméabilité des troncs supra-aortiques sans irrégularité de calibre. 
Robot : Ventricles : [none] [not mentioned]

Doctor : Lésion cérébelleuse vermienne de 12x10mm, située au niveau du foramen de Lushka. Cett lésion exerce un effet de masse sur le V4, collabé. Dilatation passive du système ventriculaire à 12mm. Pas de lésion ischémique ou hémorragique récente.
Robot : Ventricles : Dilatation passive du système ventriculaire à 12mm. [abnormal]

Doctor : Pas de lésion ischémique ou hémorragique récente. Intégrité du signal du parenchyme cérébral. Pas de priste de contraste pathologique encéphalique ou méningée. Absence de lésion.
Robot : Ventricles : [none] [not mentioned]
 
Doctor : Pas de lésion ischémique ou hémorragique récente. Structures médianes en place. Quelques hypersignaux FLAIR de la substance blanche périventriculaire peu spécifique. Ligne médiane en place. Stabilité de la discrète dilatation du système ventriculaire. Pas de prise de contraste pathologique méningée ou encéphalique. Pas de thrombose veineuse cérébrale.
Robot : Ventricles : Stabilité de la discrète dilatation du système ventriculaire. [abnormal]

Are you ready to assess the ventricles in the report ?'''], 
['Robot', "Yes, I am ready."]
]


few_shots_test=[
['Doctor', '''I will present successive texts about different patients. I am interested in 5 findings : Cushing disease, Pituitary adenoma, Adrenal adenoma, Adrenal cortical carcinoma and Neurofibromatosis. For each patient, I want you to tell me which of the 5 findings are /present or /absent, and the related passage of the text if present. It is crucial that you respond only according to this template : 
- Cushing disease /present or absent
- Pituitary adenoma /present or absent
- Adrenal adenoma /present or absent
- Adrenal cortical carcinoma /present or absent
- Neurofibromatosis /present or absent'''], 
['Chatbot', "Understood, I will respond only according to the template."],

['Doctor', """Réévaluation d'un incidentalome surrénalien gauche non sécrétant de 17mm. Nodule surrénalien de 17 mm, avec une densité à 8,50 UH en lien avec un adénome"""],
['Chatbot', """
- Cushing disease /absent
- Pituitary adenoma /absent
- Adrenal adenoma /present "incidentalome surrénalien gauche" ; "Nodule surrénalien"
- Adrenal cortical carcinoma /absent
- Neurofibromatosis /absent"""],

['Doctor', """Réévaluation d'un cortico-surrénalome droit, score de Weiss à 5, opéré en 2011. Il n'a pas été retrouvé de syndrome génétique familial. Le recherche de nodule surrénalien de type adénomateux s'est révélée négative"""],
['Chatbot', """
- Cushing disease /absent
- Pituitary adenoma /absent
- Adrenal adenoma /absent
- Adrenal cortical carcinoma /present "cortico-surrénalome droit"
- Neurofibromatosis /absent"""],

['Doctor', "Bilan de dépistage endocronologique dans le cadre d'une neurofibromatose de type I. Absence de nodule pancréatique ou surrénalien, multiples nodules pulmonaires probablement incidentels sans facteurs d'atypie."],
['Chatbot', """
- Cushing disease /absent
- Pituitary adenoma /absent
- Adrenal adenoma /absent
- Adrenal cortical carcinoma /absent
- Neurofibromatosis /present "neurofibromatose de type I" """],

['Doctor', "Il s'agissait donc d'une réévaluation d'un corticosurrénalome opéré en 2005 dans le cadre d'un dépistage de neurofibromatose de type 1. Voici les résultats endocrinologiques de la patiente suivie pour un corticosurrénalome droit opéré en 2005"],
['Chatbot', """
- Cushing disease /absent
- Pituitary adenoma /absent
- Adrenal adenoma /present
- Adrenal cortical carcinoma /present "corticosurrénalome" ; "corticosurrénalome droit"
- Neurofibromatosis /present "neurofibromatose de type 1" """]
]


shots_self_validation_contrast=[
['Doctor', '''I will present you answers from a medical student about radiology reports. He was asked to find out if exams were performed with ou without contrast media injection. I want you to review his answer and correct him if needed. One crucial rule is that respond only according to this template : "Student answer : [] - Pertinent words : [Find the pertinent words in the report to answer the question abour injection] - My answer : [Based on the report, was there injection ? You can disagree with the student]"'''], 
['Chatbot', "Understood, I will respond according to the template."],

['Doctor', "Report: 'ID :13996799.0 PROTOCOLE Séquences axiales Diffusion, EG T2, T1. Séquences 3D FLAIR et 3D TOF. Séquence 3D T1 BV sur les vaisseaux du cou. Après injection de Gadolinium, angio-IRM artérielle des TSA et séquence 3D T1 EG. / Mr Né(e) le : [JJ]/[MM]/1969 (Age : 52 ans)'. Student answer : injection ? : yes"],
['Chatbot', "Student answer : Yes - Pertinent words :'Après injection de Gadolinium' - My answer : Yes"],

['Doctor', "Report: 'ID :14465465.0 PROTOCOLE Séquences axiales diffusion, flair, T2 étoile. Séquence ARM veineuse et 3D T1 EG sans et après injection. / Mr Né(e) le : [JJ]/[MM]/1969 (Age : 52 ans)'. Student answer : injection ? : no"],
['Chatbot', "Student answer : No - Pertinent words : 'sans et avec injection' - My answer : Yes"],

['Doctor', "Report: 'ID :11492665.0 PROTOCOLE Séquences axiales diffusion, 3D SWI, T2 coro dixon, ASL et ARM PC veineux/ Mr Né(e) le : [JJ]/[MM]/1967 (Age : 50 ans)'. Student answer : injection ? : no"],
['Chatbot', "Student answer : No - Pertinent words : None - My answer : No"]
]

shots_self_validation_inference=[
['Doctor', '''I asked a medical student to tell if certain findings from radiology reports could cause headache. I will present you the report ("Positive findings") and then his answer ("Student answer", at the end of the text). I want you to review them and correct him if needed. The question is "Can the finding cause headache ?". Do not forget, fortuitous and stable findings do not cause headache. One crucial rule is that respond only according to this template : "Student answer : [Extract the student answer from the text] - Do you agree with his answer ? : [Review briefly his answer the to question and explain why you agree or not ?] - Can cause hedacahe ? : [Do you think the main finding cause headache ?]"'''], 
['Chatbot', "Understood, I will respond according to the template."],

['Doctor', "Report : Positive findings : Lésion ischémique pariétale droite - Brief explanation : Ischemic lesions of the brain are infarcted brain lesions. - Student answer : Yes"],
['Chatbot', "Student answer : Yes - What do you think about his answer ? : I agree, strokes can cause headache - Can cause headache ? : Yes"],

['Doctor', "Report : Positive findings : Multiples lésions rehaussées sus et sous-tentorielles d'allure secondaire, avec effet de masse sur les structures adjacentes, sans engagement. - Brief explanation : Multipble brain lesions, both supra and infratentorial, suspicious of besoing of secondary origin. There is mass effect on the brain. - Student answer : No"],
['Chatbot', "Student answer : No - What do you think about his answer ? : I disagree, I think that multiple brain lesions usually cause headache - Can cause headache ? : Yes"],

['Doctor', "Report : Positive findings : Leucopathie vasculaire ; Kyste sous-arachnoïdien sans effet de masse sur les structures adjacentes - Brief explanation : leukoaraiosis and sub-arachnoid cysts are fortouitous lesions.- Student answer : Yes"],
['Chatbot', "Student answer : Yes - What do you think about his answer ? : I disagree, leukoaraiosis and sub-arachnoid cyst are fortuitous findings that usually do not cause headache - Can cause hedache ? : No"],

['Doctor', "Report : Positive findings : Stabilité du cavernome frontal droit de 2cm de diamètre ; leucopathie vasculaire - Brief explanation : Cavernoma are capillary malformations. When stable, they usually do not cause hedache. Leucoaraiosis uusually does not cause headache. - Student answer : No"],
['Chatbot', "Student answer : No - What do you think about his answer ? : I agree, stable lesions usually do not cause headache - Can cause headache ? : No"]
]

shots_oxford=[
['Doctor', '''I will present you a pathology report. I want you to extract a list of information from it. You will answer only according to this template: 
- Number of glomerules: [number]
- Number of fibrous/sclerous glomerules: [number] 
- Number of glomerules with mesangial proliferation/hypercellularity: [number or % or proportion] (diffuse means all of them)
- Presence of lésion fibreuse segmentaire, hyalinose segmentaire et focale, or synéchie floculocapsulaire: [yes or no] 
- Endo-capillar proliferation: [yes or no] 
- Tubular atrophy: [number% or adjective if no %]
- Number of crescents: [number] (can only appear if extracapillar proliferation is present)
 
Here are a few examples :

Doctor: 
ID :425736 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 13 mm. Cortico-médullaire. Glomérules : 8 glomérules dont 2 sont globalement scléreux. Tous les glomérules perméables présentent une hypercellularité mésangiale. Prolifération cellulaire endo-capillaire segmentaire et focale (macrophages et polynucléaires). Hyalinose segmentaire et focale (HSF) sur cinq glomérules caractérisée par une synéchie floculo-capsulaire, une sclérose segmentaire du floculus et une hypertrophie podocytaire. Sur un de ces glomérules présentant une HSF on observe une lésion de type « tip lesion ». Absence de prolifération extra capillaire. Tubes : nécrose tubulaire aiguë modérée et diffuse. Tubulite lymphocytaire sur sections tubulaires atrophiques. Atrophie tubulaire sur 40% de la corticale. Vaisseaux : hyalinose artériolaire modérée. Les sections artérielles sont libres. Tissu interstitiel : fibrose interstitielle sur 40% de la corticale occupée par une inflammation marquée essentiellement lymphocytaire et plasmocytaire mêlée à quelques polynucléaires neutrophiles. Le rouge congo ne révèle pas de dépôt amyloïde.
Robot:
- Number of glomerules: 8
- Number of fibrous/sclerous glomerules: 2
- Number of glomerules with mesangial proliferation/hypercellularity: all of them
- Presence of segmental lésion fibreuse/scléreuse segmentaire, hyalinose segmentaire et focale, or synéchie floculocapsulaire: yes
- Endo-capillar proliferation: yes
- Tubular atrophy: 40%
- Number of crescents: 0
 
Doctor: 
ID :11787311 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 13 mm, cortico-médullaire. Glomérules : 29 glomérules analysables dont 21 globalement scléreux, transformés en pain à cacheter. Quatre glomérules sont le siège d’une synéchie floculo-capsulaire en regard de lésions segmentaires du floculus. Epaississement mésangial léger, segmentaire avec hypercellularité mésangiale sur trois glomérules. Lésion fibreuse segmentaire sur 3 glomérules. Prolifération endo-capillaire sur 1 glomérule. Augmentation de la cellularité mésangiale avec plus de 4 cellules partie mésangiale sur 3 glomérules. Deux glomérules présentent une lésion proliférative extra capillaire sous la forme d’un croissant hémi circonférentiel et d’un croissant de type fibro-cellulaire avec rupture des membranes basales glomérulaires. Absence de nécrose fibrinoïde. Tubules : atrophie tubulaire modérée à sévère. Lésion de tubulite sur les tubes atrophiques. Absence de cristaux. Vaisseaux : 2 artères analysables présentant une fibrose intimale modérée. Hyalinose artériolaire légère à modérée. Tissu interstitiel : fibrose interstitielle modérée à sévère intéressant environ 50% de la surface corticale associée à un infiltrat inflammatoire à cellules mononuclées (majoritairement lymphocytaire) modérée à sévère. Coloration du rouge congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 29
- Number of fibrous/sclerous glomerules: 21
- Number of glomerules with mesangial proliferation/hypercellularity: 3
- Presence of lésion fibreuse/scléreuse segmentaire, hyalinose segmentaire et focale, or synéchie floculocapsulaire: yes
- Endo-capillar proliferation: yes
- Tubular atrophy: modérée à sévère
- Number of crescents: 2 
 
Doctor: 
ID :1028146 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 15 mm, cortico-médullaire. Glomérules : 16 glomérules analysables dont 3 globalement scléreux. Epaississement mésangial globale et diffuse sans hypercellularité mésangiale. Un glomérule présente une lésion fibreuse segmentaire. Prolifération endocapillaire sur 6 glomérules. Prolifération extra-capillaire sur deux glomérules en faveur de croissants cellulaires caractérisés par une prolifération de cellules étirées et arrondies avec figures mitotiques comblant pour moitié la chambre urinaire. Absence de nécrose fibrinoïde. Tubules : atrophie tubulaire modérée à sévère. Lésion de tubulite sur les tubes atrophiques. Absence de cristaux. Vaisseaux : 2 artères analysables présentant une fibrose intimale modérée. Hyalinose artériolaire légère à modérée. Tissu interstitiel : fibrose interstitielle modérée à sévère intéressant environ 10% de la surface corticale Coloration du rouge congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 16
- Number of fibrous/sclerous glomerules: 3
- Number of glomerules with mesangial proliferation/hypercellularity: none
- Presence of lésion fibreuse/scléreuse segmentaire, hyalinose segmentaire et focale, or synéchie floculocapsulaire: yes
- Endo-capillar proliferation: yes
- Tubular atrophy: 10
- Number of crescents: 2 
'''],
['Chatbot', "Understood, I will respond according to the template."]
]


shots_GN=[
['Doctor', '''Extract information from the following pathology report. If you do not find the information, write "NA". You will answer only according to this template: 
- Creatinin: [value]
- Hematuria: [present or absent]
- Nephrotic syndrom: [present or absent]
- Proteinuria: [value] 
- Sample size: [size in mm]
- Number of glomerules: [number]
- Number of fibrous/sclerous glomerules: [number]
- Mesangial thickening (if present, nodular or non nodular?): [present or absent]
- Mesangial proliferation/hypercellularity: [present, with adjective, or absent]
- Double contour: [present or absent]
- Endocapillar proliferation/hypercellularity: [number or proportion] (diffuse means all of them)
- Number of (fibro)cellular crescents: [number]
- Number of fibrous crescents: [number]
- Tubular atrophy: [number, percentage or adjective]
- Interstitial fibrosis: [number, percentage or adjective]
- Number of arteries: [number]
- Intimal sclerosis or fibrosis: [adjective]
- Arteriolar hyalinosis or arteriolosclerosis: [adjective]
- MAT: [present or absent]
- Glomerules on the frozen sample: [present or absent]

Here are three examples of what I expect:

Example 1:
Doctor: Renseigenements cliniques: Greffe de cornée en 2013. Créatinine à 36mg/l; hématurie à 103; protéinurie à 2g/l. PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 13 mm. Cortico-médullaire. Glomérules : 8 glomérules dont 2 sont globalement scléreux. Tous les glomérules perméables présentent une hypercellularité mésangiale, associée à un épaississement mésengial global. Prolifération cellulaire endo-capillaire segmentaire et focale (macrophages et polynucléaires). Hyalinose segmentaire et focale (HSF) sur cinq glomérules caractérisée par une synéchie floculo-capsulaire, une sclérose segmentaire du floculus et une hypertrophie podocytaire. Sur un de ces glomérules présentant une HSF on observe une lésion de type « tip lesion ». Absence de prolifération extra capillaire. Tubes : nécrose tubulaire aiguë modérée et diffuse. Tubulite lymphocytaire sur sections tubulaires atrophiques. Atrophie tubulaire sur 40% de la corticale. Vaisseaux : hyalinose artériolaire modérée. Deux artères sur le fragment, sans anomalie. Tissu interstitiel : fibrose interstitielle sur 40% de la corticale occupée par une inflammation marquée essentiellement lymphocytaire et plasmocytaire mêlée à quelques polynucléaires neutrophiles. Le rouge congo ne révèle pas de dépôt amyloïde. Tissu congelé – Prélèvement : 11 mm, 7 glomérules analysables dont 2 globalement scléreux transformés en « pain à cacheter ». Etude en immunofluorescence : - Anticorps anti-IgA, anti-IgG, anti-IgM, anti-kappa, anti-lambda, anti-C1q, anti-C3 : absence de dépôt significatif. CONCLUSION : Néphrite tubulo-interstitielle aiguë. 
Robot:
- Creatinin: 36
- Hematuria: present
- Nephrotic syndrom: NA
- Proteinuria: 2g/l
- Sample size: 13mm
- Number of glomerules: 8
- Number of fibrous/sclerous glomerules: 2
- Mesangial thickening (if present, nodular or non nodular?): present, non nodular
- Mesangial proliferation/hypercellularity: present, globale
- Double contour: NA
- Endocapillar proliferation/hypercellularity: diffuse
- Number of (fibro)cellular crescents: NA
- Number of fibrous crescents: NA
- Tubular atrophy: 40%
- Interstitial fibrosis: 40%
- Number of arteries: 2
- Intimal sclerosis or fibrosis: absent
- Arteriolar hyalinosis or arteriolosclerosis: modérée
- MAT: NA
- Glomerules on the frozen sample: present
 
Example 2:
Doctor:Renseignements cliniques : bilan d’insuffisance rénale aiguë avec protéinurie à 0,12 g/ 24 H et hématurie. Bilan immunologique négatif. Pas de signes extrarénaux. Prise d’antibiotiques il y a un mois pour une pyélonéphrite. Hypothèse diagnostique : néphrite immuno-allergique postantibiothérapie ?  PONCTION BIOPSIE RENALE : REIN NATIF Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 1.2 cm de long. Corticale. Glomérules : 27 dont un fibreux entièrement hyalinisé. Glomérules normaux en dehors d’une légère rétraction ischémique du floculus avec pseudo-élargissement de la chambre urinaire. Epaississement mésangial modéré nodulaire chromarophile. Pas d'hypercellularité mésangiale. Parois capillaires fines. Pas de doubles contours. Absence de prolifération endo ou extracapillaire. Tubes : lésions tubulaires modérées sous forme d’un aplatissement épithélial, d’une vacuolisation de l’épithélium des tubes contournés proximaux et d’une desquamation fréquente du pôle apical de la cellule tubulaire. Vaisseaux : normaux Tissu interstitiel : absence de fibrose. Absence d’infiltrat inflammatoire. Coloration du Rouge Congo : absence de dépôt amyloïde. Tissu congelé – Prélèvement : 0,6 cm de long. 11 glomérules. Etude en immunofluorescence : La technique a été réalisée à deux reprises étant donné d’importants artéfacts de rétraction tissulaire sur la première série technique. La deuxième série technique permet de mettre en évidence l’absence de dépôts d’immunoglobulines ou de complément au niveau glomérulaire à l’aide des anticorps anti-IgA, anti-IgM, anti-IgM, anti-C1q, anti-C3, anti-fibrinogène et anti-Kappa et anti-Lambda.     CONCLUSION : lésions modérées de nécrose tubulaire aiguë. Absence d’atteinte glomérulaire. Absence d’inflammation interstitielle.
- Creatinin: NA
- Hematuria: present
- Nephrotic syndrom: NA
- Proteinuria: 0.12g/24h
- Sample size: 12mm
- Number of glomerules: 27
- Number of fibrous/sclerous glomerules: 1
- Mesangial thickening (if present, nodular or non nodular?): present, nodular
- Mesangial proliferation/hypercellularity: 0
- Double contour: present
- Endocapillar proliferation/hypercellularity: 0
- Number of (fibro)cellular crescents: 0
- Number of fibrous crescents: 0
- Tubular atrophy: 0
- Interstitial fibrosis: 0
- Number of arteries: NA
- Intimal sclerosis or fibrosis: absent
- Arteriolar hyalinosis or arteriolosclerosis: absent
- MAT: absent
- Glomerules on the frozen sample: NA
 
Example 3:
Doctor:Renseignements cliniques : Insuffisance rénal aigue. Créat. 35. ALbuminurie à 233g/g. Hypothèses: MAT? Néphrite lupique ?  PONCTION BIOPSIE RENALE : Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 0.7 cm . Cortico-médullaire. 35 glomérules perméables. Epaississement mésangial par endroits nodulaires avec augmentation de la cellularité endocapillaire. Un croissant fibrocellulaire et deux croissants fibreux. Tubes : atrophie tubulaire modérée. Vaisseaux : Fibrose intimale sévère. Pas de hyalinose artériolaire. Tissu interstitiel : infiltration fribtrotique estimée à 25%. Absence d’infiltrat inflammatoire. Coloration du Rouge Congo : absence de dépôt amyloïde. Tissu congelé – Prélèvement : 0,8 cm de long. Pas de glomérule perméable.
- Creatinin: 35
- Hematuria: NA
- Nephrotic syndrom: NA
- Proteinuria: 233g/g
- Sample size: 7mm
- Number of glomerules: 35
- Number of fibrous/sclerous glomerules: 0
- Mesangial thickening (if present, nodular or non nodular?): present, nodular
- Mesangial proliferation/hypercellularity: 0
- Double contour: present
- Endocapillar proliferation/hypercellularity: present
- Number of (fibro)cellular crescents: 1
- Number of fibrous crescents: 2
- Tubular atrophy: modérée
- Interstitial fibrosis: 25%
- Number of arteries: NA
- Intimal sclerosis or fibrosis: sévère
- Arteriolar hyalinosis or arteriolosclerosis: absent
- MAT: NA
- Glomerules on the frozen sample: absent
 '''],
['Chatbot', "Understood, I will respond according to the template."]
]

list=["""
Doctor: ID :11787311 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 13 mm, cortico-médullaire. Glomérules : 29 glomérules analysables dont 21 globalement scléreux, transformés en pain à cacheter. Quatre glomérules sont le siège d’une synéchie floculo-capsulaire en regard de lésions segmentaires du floculus. Epaississement mésangial léger, segmentaire avec hypercellularité mésangiale sur trois glomérules. Lésion fibreuse segmentaire sur 3 glomérules. Prolifération endo-capillaire sur 1 glomérule. Augmentation de la cellularité mésangiale avec plus de 4 cellules partie mésangiale sur 3 glomérules. Deux glomérules présentent une lésion proliférative extra capillaire sous la forme d’un croissant cellulaire hémi circonférentiel et d’un croissant de type fibro-cellulaire avec rupture des membranes basales glomérulaires. Absence de nécrose fibrinoïde. Tubules : atrophie tubulaire modérée à sévère. Lésion de tubulite sur les tubes atrophiques. Absence de cristaux. Vaisseaux : 2 artères analysables présentant une fibrose intimale modérée. Hyalinose artériolaire légère à modérée. Tissu interstitiel : fibrose interstitielle modérée à sévère intéressant environ 50% de la surface corticale associée à un infiltrat inflammatoire à cellules mononuclées (majoritairement lymphocytaire) modérée à sévère. Coloration du rouge congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 29
- Number of fibrous/sclerous glomerules: 21
- Mesangial proliferation/hypercellularity: 3
- Endocapillar proliferation/hypercellularity: 0
- Presence of "lésion fibreuse or scléreuse segmentaire", "hyalinose segmentaire et focale", or "synéchie floculocapsulaire": yes
- Tubular atrophy: modérée à sévère
- Number of cellular crescents: 1
- Number of fibrous crescents: 0
- Number of fibrocellular crescents: 1
 
Doctor: ID :1028146 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 15 mm, cortico-médullaire. Glomérules : 16 glomérules analysables dont 3 globalement scléreux. Epaississement mésangial globale et diffuse sans hypercellularité mésangiale. Un glomérule présente une lésion fibreuse segmentaire. Prolifération endocapillaire sur 6 glomérules. Prolifération extra-capillaire sur deux glomérules en faveur de croissants cellulaires caractérisés par une prolifération de cellules étirées et arrondies avec figures mitotiques comblant pour moitié la chambre urinaire. Absence de nécrose fibrinoïde. Tubules : atrophie tubulaire modérée à sévère. Lésion de tubulite sur les tubes atrophiques. Absence de cristaux. Vaisseaux : 2 artères analysables présentant une fibrose intimale modérée. Hyalinose artériolaire légère à modérée. Tissu interstitiel : fibrose interstitielle modérée à sévère intéressant environ 10% de la surface corticale Coloration du rouge congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 16
- Number of fibrous/sclerous glomerules: 13
- Mesangial proliferation/hypercellularity: 0
- Endocapillar proliferation/hypercellularity: 6
- Presence of "lésion fibreuse or scléreuse segmentaire", "hyalinose segmentaire et focale", or "synéchie floculocapsulaire": yes
- Tubular atrophy: modérée à sévère
- Number of cellular crescents: 2
- Number of fibrous crescents: 0
- Number of fibrocellular crescents: 0

Doctor: ID :4756760 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 12 mm. Cortico-médullaire. Glomérules : 15 glomérules dont 1 est globalement scléreux. Epaississement mésangial segmentaire et focal avec hypercellularité mésangiale sur moins de 50% des glomérules. Prolifération endo capillaire segmentaire et focale (1 anse capillaire au sein d’un glomérule). Présence d’une synéchie floculo-capsulaire. Absence de prolifération extra capillaire, absence d’anomalie de la membrane basale. Tubes : sans particularité. Vaisseaux : sans particularité. Tissu interstitiel : sans particularité. La coloration par le rouge congo ne révèle pas de dépôt amyloïde. Tissu 
- Number of glomerules: 15
- Number of fibrous/sclerous glomerules: 1
- Mesangial proliferation/hypercellularity: <50%
- Endocapillar proliferation/hypercellularity: 1
- Presence of "lésion fibreuse or scléreuse segmentaire", "hyalinose segmentaire et focale", or "synéchie floculocapsulaire": yes
- Tubular atrophy: 0
- Number of cellular crescents: 0
- Number of fibrous crescents: 0
- Number of fibrocellular crescents: 0
"""]

shots_Mayo=[
['Doctor', '''I will present you a pathology report. I want you to extract a list of information from it. If the information is not present or cannot be implied, write "NA". You will answer only according to this template: 
- Number of glomerules: [number]
- Number of fibrous/sclerous glomerules: [number] 
- Tubular atrophy: [description or percentage]
- Interstitial fibrosis: [description or percentage]
- Intimal arteriosclerosis/fibrosis: [absent or present (severity)] (write "NA" if no artery is present in the sample)
 
Here are a few examples (I annotated the text with brackets):

Doctor: PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 13 mm. Cortico-médullaire. Glomérules : 8 glomérules [Number of glomerules] dont 2 sont globalement scléreux [Number of fibrous/sclerous glomerules]. Tous les glomérules perméables présentent une hypercellularité mésangiale. Prolifération cellulaire endo-capillaire segmentaire et focale (macrophages et polynucléaires). Hyalinose segmentaire et focale (HSF) sur cinq glomérules caractérisée par une synéchie floculo-capsulaire, une sclérose segmentaire du floculus et une hypertrophie podocytaire. Sur un de ces glomérules présentant une HSF on observe une lésion de type « tip lesion ». Absence de prolifération extra capillaire. Tubes : nécrose tubulaire aiguë modérée et diffuse. Tubulite lymphocytaire sur sections tubulaires atrophiques. Atrophie tubulaire sur 40% de la corticale [Tubular atrophy]. Vaisseaux : hyalinose artériolaire modérée. Les sections artérielles sont libres [Intimal arteriosclerosis]. Tissu interstitiel : fibrose interstitielle sur 40% de la corticale [Interstitial fibrosis] occupée par une inflammation marquée essentiellement lymphocytaire et plasmocytaire mêlée à quelques polynucléaires neutrophiles. Le rouge congo ne révèle pas de dépôt amyloïde.
Robot:
- Number of glomerules: 8
- Number of fibrous/sclerous glomerules: 2 
- Tubular atrophy: 40%
- Interstitial fibrosis: 40%
- Intimal arteriosclerosis/fibrosis: absent
 
Doctor: PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 13 mm, cortico-médullaire. Glomérules : 29 glomérules analysables [Number of glomerules] dont 21 globalement scléreux [Number of fibrous/sclerous glomerules], transformés en pain à cacheter. Quatre glomérules sont le siège d’une synéchie floculo-capsulaire en regard de lésions segmentaires du floculus. Epaississement mésangial léger, segmentaire avec hypercellularité mésangiale sur trois glomérules. Lésion fibreuse segmentaire sur 3 glomérules. Prolifération endo-capillaire sur 1 glomérule. Augmentation de la cellularité mésangiale avec plus de 4 cellules partie mésangiale sur 3 glomérules. Deux glomérules présentent une lésion proliférative extra capillaire sous la forme d’un croissant cellulaire hémi circonférentiel et d’un croissant de type fibro-cellulaire avec rupture des membranes basales glomérulaires. Absence de nécrose fibrinoïde. Tubules : atrophie tubulaire modérée à sévère [Tubular atrophy]. Lésion de tubulite sur les tubes atrophiques. Absence de cristaux. Vaisseaux : 2 artères analysables présentant une fibrose intimale modérée [Intimal arteriosclerosis]. Hyalinose artériolaire légère à modérée. Tissu interstitiel : fibrose interstitielle modérée à sévère intéressant environ 50% de la surface corticale [Interstitial fibrosis] associée à un infiltrat inflammatoire à cellules mononuclées (majoritairement lymphocytaire) modérée à sévère. Coloration du rouge congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 29
- Number of fibrous/sclerous glomerules: 21 
- Tubular atrophy: modérée à sévère
- Interstitial fibrosis: 50%
- Intimal arteriosclerosis/fibrosis: present (modérée)
 
Doctor: PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 15 mm, cortico-médullaire. Glomérules : 16 glomérules analysables [Number of glomerules] dont 3 globalement scléreux [Number of fibrous/sclerous glomerules]. Epaississement mésangial globale et diffuse sans hypercellularité mésangiale. Un glomérule présente une lésion fibreuse segmentaire. Prolifération endocapillaire sur 6 glomérules. Prolifération extra-capillaire sur deux glomérules en faveur de croissants cellulaires caractérisés par une prolifération de cellules étirées et arrondies avec figures mitotiques comblant pour moitié la chambre urinaire. Absence de nécrose fibrinoïde. Tubules : atrophie tubulaire modérée à sévère [Tubular atrophy]. Lésion de tubulite sur les tubes atrophiques. Absence de cristaux. Vaisseaux : 2 artères analysables présentant une fibrose intimale modérée [Intimal arteriosclerosis]. Hyalinose artériolaire légère à modérée. Tissu interstitiel : fibrose interstitielle modérée à sévère intéressant environ 10% de la surface corticale [Interstitial fibrosis] Coloration du rouge congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 16
- Number of fibrous/sclerous glomerules: 13 
- Tubular atrophy: modérée à sévère
- Interstitial fibrosis: 10%
- Intimal arteriosclerosis/fibrosis: present (modérée à sévère)

Doctor: PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 12 mm. Cortico-médullaire. Glomérules : 15 glomérules [Number of glomerules] dont 1 est globalement scléreux [Number of fibrous/sclerous glomerules]. Epaississement mésangial segmentaire et focal avec hypercellularité mésangiale sur moins de 50% des glomérules. Prolifération endo capillaire segmentaire et focale (1 anse capillaire au sein d’un glomérule). Présence d’une synéchie floculo-capsulaire. Absence de prolifération extra capillaire, absence d’anomalie de la membrane basale. Tubes : sans particularité [Tubular atrophy]. Vaisseaux : sans particularité [Intimal arteriosclerosis]. Tissu interstitiel : sans particularité [Interstitial fibrosis]. La coloration par le rouge congo ne révèle pas de dépôt amyloïde. Tissu 
- Number of glomerules: 15
- Number of fibrous/sclerous glomerules: 1
- Tubular atrophy: 0%
- Interstitial fibrosis: 0%
- Intimal arteriosclerosis/fibrosis: absent

Understood?
 '''],
['Chatbot', "Understood, I will respond according to the template."]
]

shots_Mayo_de=[
['Doctor', '''I will present you a pathology report. I want you to extract a list of information from it. If the information is not present or cannot be implied, write "NA". You will answer only according to this template: 
- Number of glomerules: [number]
- Number of fibrous/sclerous glomerules: [number] 
- Tubular atrophy: [description or percentage]
- Interstitial fibrosis: [description or percentage]
- Arterial intimal Fibrosis: [absent or present (severity)] (write "NA" if no artery is present in the sample)
 
Here are a few examples (I annotated the text with brackets):

Doctor: Mikroskopisch:1. Nierengewebe ausschließlich aus der Rinde mit 25 Glomeruli [Number of glomerules]. Davon 15 komplett erhöhtem solide Mesangialzellgehalt, weitere zwei mit segmental verschlossenen Kapillarlichtungen. Keine Kapselproliferate, keine Nekrosen. Glomeruläre Basalmembranen regulär konturiert. Kortikale Tubuli insgesamt zu ca. 20 % atrophisch [Tubular atrophy], umgeben von mäßig fibrotisch verbreitertem Interstitium [Interstitial fibrosis] mit mäßig dichtem Rundzellinfiltrat. Übrige kortikale Tubuli mit fokal gering vergrößerten Außendurchmessern, Epithel anisometrisch vakuolisiert und mit Bürstensaumdefekten. Angrenzendes Interstitium mit zahlreichen Schaumzellnestern. Keine weiteren eindeutigen leukozytären Infiltrate. Präglomeruläre Gefäßäste mit freien Lichtungen, Arterien nicht erfasst [Arterial intimal Fibrosis], Arteriolen mit keine regulärem Wandaufbau. 
Robot:
- Number of glomerules: 25
- Number of fibrous/sclerous glomerules: 0
- Tubular atrophy: 20%
- Interstitial fibrosis: mäßig
- Arterial intimal Fibrosis: NA
 
Doctor: Mikroskopisch: Rinde mit 20 Glomeruli [Number of glomerules], davon 7 global bis komplett kollaptisch sklerosiert [Number of fibrous/sclerous glomerules], 2 dieser Glomeruli mit zellreichen Kapselproliferaten und tubulärer Metaplasie. Ein weiterer Glomerulus mit segmentalem Kapillarlichtungsverschluss bei stark fibrotisch verbreitertem Mesangium ohne eindeutig erhöhten Zellgehalt. Fokal und segmental erhöhter Mesangialzellgehalt. In den Kapillarlichtungen kein eindeutig erhöhter Zellgehalt. Kortikale Tubuli insgesamt zu ca. 20 % atrophisch [Tubular atrophy], umgeben von ödematös und fibrotisch verbreitertem Interstitium [Interstitial fibrosis] mit dichtem Infiltrat aus Lymphozyten, Histiozyten und auch vielen Plasmazellen. Kein storiformes Fibrosemuster. Übrige kortikale Tubuli mit frische Erythrozytenzylinder. Tubuläre Basalmembranen zart, angrenzendes Interstitium zu ca. 20 % mäßig ödematös aufgelockert, dabei von dem oben beschriebenen Infiltrat mäßig dicht infiltriert. Geringe intraepitheliale Komponente des Infiltrates. Präglomeruläre Gefäßäste mit freien Lichtungen, Arterien mit regulärem Wandaufbau [Arterial intimal Fibrosis], Arteriolen ebenfalls. Keine leukozytären Gefäßwandinfiltrate.  
Robot:
- Number of glomerules: 20
- Number of fibrous/sclerous glomerules: 7 
- Tubular atrophy: 20%
- Interstitial fibrosis: 20%
- Arterial intimal Fibrosis: absent
 
Doctor: Mikroskopisch:1. Das Gewebe enthält Rindenanteile, in denen auf mehreren Schnittstufen bis zu 19 Glomerula getroffen sind [Number of glomerules], davon 3 alt verödet [Number of fibrous/sclerous glomerules]. Die restlichen zeigen überwiegend zarte Basalmembranen. Teilweise eine segmental betonte Matrixvermehrung ohne eindeutige Hyperzellularität. Die Schlingen füllen stellenweise den Kapselraum aus. In 6 Glomerula zeigen sich Halbmonde, 4 x proliferativ, 2 x sklerosierend-proliferativ. Dazu einmal eine Synechie zur Bowman-Kapsel sowie eine Kapselfibrose. Das Tubulo-Interstitium ist regelrecht ausgebildet und zeigt keine Entzündungsinfiltrate [Tubular atrophy] [Interstitial fibrosis]. Es liegen Zeichen der akuten tubulären Schädigung vor in Form von Shedding des Bürstensaums, Epithelabflachung und Kernverlust. Ein Tubulus weist Tamm-Horsfall-Proteinauf. Arterien mit schwer fibrotisch verbreiterer [Arterial intimal Fibrosis]. Arteriolen unauffällig. 
Robot:
- Number of glomerules: 19
- Number of fibrous/sclerous glomerules: 3
- Tubular atrophy: 0%
- Interstitial fibrosis: 0%
- Arterial intimal Fibrosis: present (schwer)

Doctor: Mikroskopisch: 1. Nierengewebe aus Mark und Rinde mit 17 Glomeruli [Number of glomerules]. Davon 6 überwiegend kollaptisch sklerosiert [Number of fibrous/sclerous glomerules]. 1 Glomerulus mit zellreichem Kapselproliferat. Dieser Glomerulus und mindestens 5 weitere mit segmentalem Verschluss von Kapillarlichtungen, teils unter Anheftung an die Kapsel. Diffus, gering erhöhter Mesangialzellgehalt. Glomeruläre Basalmembranen ohne Aufsplitterungen. In der Kapillarlichtungen nur fokal erhöhter Gehalt an mononukleären Zellen. Keine Thromben. Kortikale Tubuli insgesamt zu ca. 30 % atrophisch [Tubular atrophy], umgeben von ödematös und fibrotisch verbreitertem Interstitium [Interstitial fibrosis] mit mäßig dichtem lymphohistiozytären Infiltrat unter Beteiligung von einigen Schaumzellen. Übrige kortikale Tubuli fokal mit stark vergrößertem Außendurchmesser. Epithel mit Resorptionsvakuolen. Tubuläre Basalmembranen und angrenzendes Interstitium zart. Keine leukozytären Infiltrate. Präglomeruläre Gefäßäste mit freien Lichtungen, Arterien mit überwiegend gering fibrotisch verbreiterer Intima [Arterial intimal Fibrosis]. Eine kleine Arteriole mit stark ödematös aufgelockerter Media und Wandaufbau sonst regulär. In den  
- Number of glomerules: 17
- Number of fibrous/sclerous glomerules: 6
- Tubular atrophy: 30%
- Interstitial fibrosis: 30%
- Arterial intimal Fibrosis: present (überwiegend gering)
 
Doctor: Mikroskopisch: 1. Kortikomedullärer Übergang, Rinde, Kapselanteile, perirenales Fettgewebe und 24 Glomeruli [Number of glomerules]. Einige davon aus dem Verband gelöst. Schlingenkonvolute fokal gering eingefallen, mit aufgefalteten Basalmembranen. Keine Aufsplitterungen, keine Nekrosen, keine Kapselproliferate, kein erhöhter Zell- oder Matrixgehalt. Kortikale Tubuli insgesamt zu ca. 10% atrophisch [Tubular atrophy], angrenzendes Interstitium fibrotisch verbreitert [Interstitial fibrosis], dicht von zahlreichen Plasmazellen, ödematös und Lymphozyten, Histiozyten und wenigen eosinophilen Granulozyten durchsetzt. Keine Granulome, keine Riesenzellen. Übrige kortikale Tubuli mit abgeflachtem Epithel und Bürstensaumdefekten. Angrenzendes Interstitium nahezu komplett gering bis mäßig ödematös verbreitert und von dem oben beschriebenen Infiltrat durchsetzt. Fokal dichtes intraepitheliales Infiltrat. Außerdem plasmazelluläres Infiltrat im perirenalen Fettgewebe. Kein eindeutig storiformes Fibrosemuster. Präglomeruläre Gefäßäste mit freien Lichtungen, als Arterien und Arteriolen mit regulärem Wandaufbau [Arterial intimal Fibrosis]. Die wenigen erfassten Venenäste unauffällig. 
- Number of glomerules: 24
- Number of fibrous/sclerous glomerules: 0
- Tubular atrophy: 10%
- Interstitial fibrosis: 10%
- Arterial intimal Fibrosis: absent


Understood?
 '''],
['Chatbot', "Understood, I will respond according to the template."]
]


shots_GN_de=[
['Doctor', '''I will present you a german pathology report. I want you to extract a list of information from it. If the information is the present, write "NA". You will answer only according to this template: 
- Number of glomerules: [number]
- Number of fibrous/sclerous glomerules: [number] 
- Mesangial proliferation/hypercellularity: [number or adjective from the text]
- Presence of cells in the capillary light: [number or adjective from the text or yes or no]
- Capsular proliferation/increase in cellularity: [number or adjective from the text or yes or no]
- Presence of closure lesion: [yes or no] 
- Tubular atrophy: [number or percentage or adjective]
- Number of crescents: [number]
 
Here are a few examples (I have annotated them with brackets to tell you where to look for information) :

Doctor: Mikroskopisch:1. Nierengewebe ausschließlich aus der Rinde mit 25 Glomeruli [Number of glomerules]. Davon 15 komplett erhöhtem solide Mesangialzellgehalt [Mesangial proliferation/hypercellularity], weitere zwei mit segmental verschlossenen Kapillarlichtungen [Presence of closure lesion]. Keine Kapselproliferate [Capsular proliferation/increase in cellularity], keine Nekrosen. Glomeruläre Basalmembranen regulär konturiert. Kortikale Tubuli insgesamt zu ca. 20 % atrophisch [Tubular atrophy], umgeben von mäßig fibrotisch verbreitertem Interstitium mit mäßig dichtem Rundzellinfiltrat. Übrige kortikale Tubuli mit fokal gering vergrößerten Außendurchmessern, Epithel anisometrisch vakuolisiert und mit Bürstensaumdefekten. Angrenzendes Interstitium mit zahlreichen Schaumzellnestern. Keine weiteren eindeutigen leukozytären Infiltrate. Präglomeruläre Gefäßäste mit freien Lichtungen, Wandaufbau regulär. bis mäßig - b.w. -In dem immunhistochemischen Präparaten mesangial und auch zum Teil entlang der Basalmembranen kräftige Positivität für IgA, mäßige für C1q und geringe für C3c. Glomerulär keine eindeutige Positivität für IgG oder IgM.Beurteilung:Fortgeschrittene diffuse mesangioproliferative IgA-Glomerulonephritis mit sekundärer fokaler und segmentaler Glomerulosklerose vom nicht näher spezifizierten Typ sowie sekundärer diffDoctor glomerulärer Globalsklerose. Ca 20 %ige Tubulusatrophie und interstitielle Fibrose der Nierenrinde sowie mäßiger akuter Tubulusschaden in den übrigen Kortexanteilen.
Robot:
- Number of glomerules: 25
- Number of fibrous/sclerous glomerules: 0
- Mesangial proliferation/hypercellularity: 15
- Presence of cells in the capillary light: 0
- Capsular proliferation/increase in cellularity: 0
- Presence of closure lesion: yes
- Tubular atrophy: 20%
- Number of crescents: 0
 
Doctor: Mikroskopisch: Rinde mit 20 Glomeruli [Number of glomerules], davon 7 global bis komplett kollaptisch sklerosiert [Number of fibrous/sclerous glomerules], 2 dieser Glomeruli mit zellreichen Kapselproliferaten [Capsular proliferation/increase in cellularity] und tubulärer Metaplasie. Ein weiterer Glomerulus mit segmentalem Kapillarlichtungsverschluss [Presence of closure lesion] bei stark fibrotisch verbreitertem Mesangium ohne eindeutig erhöhten Zellgehalt. Fokal und segmental erhöhter Mesangialzellgehalt [Mesangial proliferation/hypercellularity]. In den Kapillarlichtungen kein eindeutig erhöhter Zellgehalt [Presence of cells in the capillary light]. Kortikale Tubuli insgesamt zu ca. 20 % atrophisch [Tubular atrophy], umgeben von ödematös und fibrotisch verbreitertem Interstitium mit dichtem Infiltrat aus Lymphozyten, Histiozyten und auch vielen Plasmazellen. Kein storiformes Fibrosemuster. Übrige kortikale Tubuli mit frische Erythrozytenzylinder. Tubuläre Basalmembranen zart, angrenzendes Interstitium zu ca. 20 % mäßig ödematös aufgelockert, dabei von dem oben beschriebenen Infiltrat mäßig dicht infiltriert. Geringe intraepitheliale Komponente des Infiltrates. Präglomeruläre Gefäßäste mit freien Lichtungen, Arterien mit regulärem Wandaufbau, Arteriolen ebenfalls. Keine leukozytären Gefäßwandinfiltrate. /beck/zanfBürstensaumdefekten. Resorptionsvakuolen Einige und  In den immunhistochemischen Präparaten mesangial kräftige Positivität für IgA, mäßige für IgM, mäßige bis kräftige für C1q, spurenhafte für C3c. Glomerulär keine eindeutige Positivität für IgG. Beurteilung: Fokal mesangioproliferative und IgA-Glomerulonephritis mit sekundärer fokaler und segmentaler Glomerulosklerose vom tubulointerstitieller florider (NOS), mäßiger nicht näher spezifizierten Typ Begleitnephritis, ca. 20%iger Tubulusatrophie und interstitieller Fibrose der Nierenrinde. floride extrakapillär proliferative fokal 
Robot:
- Number of glomerules: 20
- Number of fibrous/sclerous glomerules: 7
- Mesangial proliferation/hypercellularity: Fokal und segmental
- Presence of cells in the capillary light: 0
- Capsular proliferation/increase in cellularity: 2
- Presence of closure lesion: yes
- Tubular atrophy: 20%
- Number of crescents: 0
 
Doctor: Mikroskopisch:1. Das Gewebe enthält Rindenanteile, in denen auf mehreren Schnittstufen bis zu 19 Glomerula getroffen sind [Number of glomerules], davon 3 alt verödet [Number of fibrous/sclerous glomerules]. Die restlichen zeigen überwiegend zarte Basalmembranen. Teilweise eine segmental betonte Matrixvermehrung ohne eindeutige Hyperzellularität [Mesangial proliferation/hypercellularity]. Die Schlingen füllen stellenweise den Kapselraum aus. In 6 Glomerula zeigen sich Halbmonde [Number of crescents], 4 x proliferativ, 2 x sklerosierend-proliferativ. Dazu einmal eine Synechie zur Bowman-Kapsel sowie eine Kapselfibrose [Presence of focal sclerous of fibrous lesion]. Das Tubulo-Interstitium ist regelrecht ausgebildet und zeigt keine Entzündungsinfiltrate. Es liegen Zeichen der akuten tubulären Schädigung vor in Form von Shedding des Bürstensaums, Epithelabflachung und Kernverlust. Ein Tubulus weist Tamm-Horsfall-Proteinauf. Größere Arterienäste sind nicht getroffen. Arteriolen unauffällig.Immunhistochemie:Es finden sich körnige mesangiale Ablagerungen für IgA bei negativem Befund für C3, IgM, IgG, Fibrinogen und Komplementkomplex C5b-9.2. Leider kein Glomerulum getroffen, daher keine Immunfluoreszenzuntersuchung. Beurteilung:1. und 2. Eigennierenbiopsie mit extrakapillär-proliferativer IgA-Nephropathie mit segmentalen Sklerosen und tubulären Schädigung bei ansonsten unauffälligem Tubulo-Interstitium sowie unauffälligen Arteriolen. IgA-Ablagerungen, Zeichen der akuten Oxford-Klassifikation: M0 S1 E0 T0.Wir werden noch elektronenmikroskopische Untersuchungen (3.) durchführen und nachberichten.Der Befund wurde der Klinik (Fr. Dr. Kisner) am 01.10.2014 vorab mitgeteilt. /göbe/dahm(Prof. Dr. Büttner) (Frau Dr. Göbel)
Robot:
- Number of glomerules: 19
- Number of fibrous/sclerous glomerules: 3
- Mesangial proliferation/hypercellularity: 0
- Presence of cells in the capillary light: 0
- Capsular proliferation/increase in cellularity: yes
- Presence of closure lesion: yes
- Tubular atrophy: NA
- Number of crescents: 6

Doctor: Mikroskopisch: 1. Nierengewebe aus Mark und Rinde mit 17 Glomeruli [Number of glomerules]. Davon 6 überwiegend kollaptisch sklerosiert [Number of fibrous/sclerous glomerules]. 1 Glomerulus mit zellreichem Kapselproliferat [Capsular proliferation/increase in cellularity]. Dieser Glomerulus und mindestens 5 weitere mit segmentalem Verschluss von Kapillarlichtungen [Presence of closure lesion], teils unter Anheftung an die Kapsel. Diffus, gering erhöhter Mesangialzellgehalt [Mesangial proliferation/hypercellularity]. Glomeruläre Basalmembranen ohne Aufsplitterungen. In der Kapillarlichtungen nur fokal erhöhter Gehalt an mononukleären Zellen [Presence of cells in the capillary light]. Keine Thromben. Kortikale Tubuli insgesamt zu ca. 30 % atrophisch, umgeben von ödematös und fibrotisch verbreitertem Interstitium mit mäßig dichtem lymphohistiozytären Infiltrat unter Beteiligung von einigen Schaumzellen. Übrige kortikale Tubuli fokal mit stark vergrößertem Außendurchmesser. Epithel mit Resorptionsvakuolen. Tubuläre Basalmembranen und angrenzendes Interstitium zart. Keine leukozytären Infiltrate. Präglomeruläre Gefäßäste mit freien Lichtungen, Arterien mit regulärem Wandaufbau, Intima, eine kleine Arteriole mit stark ödematös aufgelockerter Media und Wandaufbau sonst regulär. In den immunhistochemischen Präparaten mesangial kräftige Positivität für IgA, mäßige Positivität für IgG, GN, C1q, C3c. Beurteilung: Diffus mesangioproliferative sowie IgA-Glomerulonephritis mit sekundärer fokaler und segmentaler Glomerolussklerose vom nicht näher spezifiziertem Typ (NOS), ca. 30 %iger Tubulusatrophie [Tubular atrophy] und interstitieller Fibrose der Nierenrinde. Schwere kompensatorische Hypertrophie der übrigen kortikalen Tubuli. Geringe Arteriolopathie, grenzwertig zu einer primären malignen Nephrosklerose (siehe 
Robot :
- Number of glomerules: 17
- Number of fibrous/sclerous glomerules: 6
- Mesangial proliferation/hypercellularity: Diffus, gering erhöhter
- Presence of cells in the capillary light: fokal erhöhter
- Capsular proliferation/increase in cellularity: 1
- Presence of closure lesion: yes
- Tubular atrophy: 30%
- Number of crescents: 0


Understood ?
 '''],
['Chatbot', "Understood, I will respond according to the template."]
]

shots_lupus=[
['Doctor', '''I will present you a pathology report. I want you to extract a list of information from it. You will answer only according to this template: 
- Number of glomerules: [number]
- Number of fibrous/sclerous glomerules: [number] 
- Number of glomerules with leukocytes, polynuclear neutrophils or caryorrhexis: [number or percentage: 0%, <25%, 25-50% or >50%]
- Number of glomerules with endo-capillar proliferation: [number or percentage: 0%, <25%, 25-50% or >50%]
- Number of glomerules with fibrinoid necrosis: [number or percentage: 0%, <25%, 25-50% or >50%]
- Number of glomerules with wire-loops or double contours: [number or percentage: 0%, <25%, 25-50% or >50%]
- Tubular atrophy: [percentage: 0%, <25%, 25-50% or >50%]
- Number of cellular or fibrocellular crescents: [number]
- Number of fibrous crescents: [number]
- Interstitial inflammation: [percentage: 0%, <25%, 25-50% or >50%]
- Interstitial fibrosis: [percentage: 0%, <25%, 25-50% or >50%]
 
Here are a few examples :

Doctor: 
ID :C2114235 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 3 fragments adressés fixés de 1 mm chacun correspondant à du tissu musculaire. A la réception des prélèvements en Anatomie Cytologique, on décide de fixer un des deux fragments adressés à l’état frais, devant l’absence de tissu rénal adressé initialement fixé. Glomérules : 12 glomérules analysables, tous perméables. Epaississement mésangial diffus et global associé à une augmentation significative de la cellularité mésangiale diffuse et globale. Hypercellularité endocapillaire segmentaire et focale (sur 2 glomérules). Les podocytes sont dans leur ensemble turgescents. Il n’est pas observé de véritable lésion de hyalinose segmentaire et focale. Sur un glomérule, possible prolifération extracellulaire (croissant cellulaire). Un glomérule présente un aspect de wire loop. A l’imprégnation argentique de Jones, on met en évidence la présence de spicule de façon diffuse. Tubes : absence d’atrophie tubulaire. Absence de tubulite. Vaisseaux : une artère analysable dépourvue de fibrose intimale. Absence de hyalinose artériolaire. Tissu interstitiel : absence de fibrose interstitielle. Absence d’infiltrat inflammatoire interstitiel. Coloration du rouge Congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 12
- Number of fibrous/sclerous glomerules: 0 
- Number of glomerules with leukocytes, polynuclear neutrophils or caryorrhexis: 0
- Number of glomerules with endo-capillar proliferation: 2
- Number of glomerules with fibrinoid necrosis: 0
- Number of glomerules with wire-loops or double contours: 1
- Tubular atrophy: 0%
- Number of cellular or fibrocellular crescents: 1
- Number of fibrous crescents: 0
- Interstitial inflammation: 0%
- Interstitial fibrosis: 0%
 
Doctor: 
ID :C1919702 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 10 mm, corticale. Glomérules : 16 glomérules perméables. Prolifération cellulaire extra-capillaire caractérisée par 9 glomérules présentant un croissant de type cellulaire associés à une rupture de la membrane basale sur l’imprégnation argentique de Jones ainsi qu’à une nécrose fibrinoïde. Hypercellularité endocapillaire globale et diffuse composée d’éléments mononucléés et de polynucléaires neutrophiles. Hypercellularité mésangiale segmentaire et focale. Deux glomérules présentent une adhésion floculocapsulaire. On note focalement un dépôt endo-membraneux. Présence de quelques débris nucléaires au sein des floculus. Absence de double contour ou de thrombose. Tubes : sans particularité. Vaisseaux : sans particularité. Tissu interstitiel : minime infiltrat inflammatoire lymphocytaire interstitiel. Absence de fibrose. Le Rouge Congo ne révèle pas de dépôt amyloïde.
Robot:
- Number of glomerules: 16
- Number of fibrous/sclerous glomerules: 0 
- Number of glomerules with leukocytes, polynuclear neutrophils or caryorrhexis: >50%
- Number of glomerules with endo-capillar proliferation: >50%
- Number of glomerules with fibrinoid necrosis: >50%
- Number of glomerules with wire-loops or double contours: 0
- Tubular atrophy: 0%
- Number of cellular or fibrocellular crescents: 9
- Number of fibrous crescents: 0
- Interstitial inflammation: <25%
- Interstitial fibrosis: 0%
 
Doctor: 
ID :C1926041 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA - colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 9 mm, cortical. Glomérules : 18 glomérules dont 2 glomérules entièrement scléreux. Aucun glomérule normal. Deux lésions de prolifération extracapillaire : 1 glomérule avec avec rupture de la paroi capillaire et présence d’un croissant cellulaire et 1 autre glomérule avec croissant fibro-cellulaire. Absence de nécrose fibrinoïde. 4 glomérules avec prolifération endocapillaire dont 2 avec karryorhexis. Epaississement mésangial diffus et global avec nombreuses images d’hypercellularité mésangiale. Accentuation de la lobulation du flocculus dans plusieurs glomérules. Présence de nombreuses lésions de synéchie flocculo-capsulaires (4 glomérules en moyenne par plan de ocupe). Une lésion de hyalinose segmentaire et focale Nombreux doubles contuores Tubes : Atrophie tubulaire sur 10% de la surface corticale. Absence de cylindres hématiques. Absence de cylindre leucocytaire. Absence de nécrose tubulaire aiguë. Absence de cristaux. Absence de tubulite. Vaisseaux : Absence de lésion vasculaire. Tissu interstitiel : Fibrose interstitielle sur 10% de la surface corticale. Absence d’infiltrat inflammatoire interstitiel significatif. Coloration du rouge Congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 18
- Number of fibrous/sclerous glomerules: 2 
- Number of glomerules with leukocytes, polynuclear neutrophils or caryorrhexis: 2
- Number of glomerules with endo-capillar proliferation: 4
- Number of glomerules with fibrinoid necrosis: 0
- Number of glomerules with wire-loops or double contours: >50%
- Tubular atrophy: <25%
- Number of cellular or fibrocellular crescents: 2
- Number of fibrous crescents: 0
- Interstitial inflammation: 0%
- Interstitial fibrosis: <25%
'''],
['Chatbot', "Understood, I will respond according to the template."]
]

shots_lupus_class=[
['Doctor', '''I will present you a pathology report. I want you to extract a list of information from it. Then, you will decide the class of the report based on a set of rules. Follow this template: 
1- Number of glomerules: [number]
2- Number of fibrous/sclerous glomerules: [number] 
3- Endo-capillar proliferation: [number]
4- Cellular or fibrocellular crescents: [number]
5- Total percentage of glomeruli with endocapillar proliferation or cellular crescents: [0%/<50%/>50%]
6- Mesengial proliferation: [yes or no]
7- Membranous immune deposits: [yes or no - I am only interested in deposits that are "extra-membraneux", not "sous-endothéliaux" or "mésangiaux", and not focal or segmental]
- Class: [I if no mesengial or endocapillar proliferation or crescents (5- must be 0%, 6- must be no), II if mesengial proliferation and NO endocapillar proliferation or crescents (6- must be yes, 5 must be 0%), III if endocapillar proliferation and crescents <50% (5- must be <50%), IV if endocapillar proliferation and crescents >50% (5- must be >50%), V if extra-membranous immune deposits (7- must be yes), VI if >90% of sclerous glomeruli]
[If multiple classes are possible, choose the highest one. Class V is the only one that can be associated to another class]
 
Here are a few examples :

Doctor: 
ID :C1415360 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 13 mm, cortico-médullaire. Glomérules : 29 glomérules analysables dont 2 globalement scléreux. 18 glomérules présentent des signes d'activité ou de chronicité. On note la présence de croissants cellulaires sur 6 glomérules. 11 glomérules sont le siège de wire loop. 3 glomérules sont le siège d'une fibrose mésangiale segmentaire. Il existe une prolifération endocapillaire sur 13 glomérules. On met en évidence la présence de spicules de façon diffuse. Tubes : atrophie tubulaire légère. Présence de cylindres hématiques. Vaisseaux : une artère analysable dépourvue de fibrose intimale. Absence d'anomalie artériolaire. Tissu interstitiel : fibrose interstitielle légère. Présence d'un infiltrat lymphocytaire focal. Coloration du Rouge Congo : absence de dépôt amyloïde. Tissu congelé – Prélèvement : 12 glomérules ; 14 glomérules analysables dont 2 globalement scléreux. Etude en immunofluorescence : - Anticorps anti-IgG, anti-C1q, anti-C3 : dépôts abondants mésangiaux sous endothéliaux et extra-membraneux (3+). - Anticorps anti-IgA, anti-IgM, anti-Kappa et anti-Lambda : dépôts mésangiaux, sous endothéliaux et extra-membraneux (2+). - Anticorps anti-fibrinogène : fixation segmentaire sur 2 glomérules.
Robot:
1- Number of glomerules: 29
2- Number of fibrous/sclerous glomerules: 2 
3- Endo-capillar proliferation: 13
4- Cellular or fibrocellular crescents: 6
5- Total percentage of glomeruli with endocapillar proliferation or cellular crescents: >50%
6- Mesengial proliferation: yes
7- Extra-membraneux immune deposits: yes
- Class: IV and V
 
Doctor: 
ID :C1429952 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : deux fragments biopsiques, cortico-médullaire, 24 mm au total. Glomérules : 46 glomérules analysables dont deux globalement scléreux transformés en pains à cacheter. Lésions glomérulaires segmentaires et focales sous la forme d’une prolifération endocapillaire sur 7 glomérules, d’une prolifération extracapillaire avec croissants fibro-cellulaires sur 3 glomérules. Présence de lésions fibreuses segmentaires sur 5 glomérules. Trois glomérules sont le siège de lésions de hyalinose segmentaire et focale. Absence de caryorrhexis. Absence de wire loop. Absence de spicules ou de chaînettes. Tubes : présence de quelques cylindriques hématiques. Absence d’atrophie tubulaire significative. Vaisseaux : deux artères dépourvues de fibrose intimale. Absence d’infiltrat inflammatoire. Absence de hyalinose artériolaire. Tissu interstitiel : absence de fibrose interstitielle. Absence d’infiltrat inflammatoire. Coloration du Rouge Congo : absence de dépôts amyloïdes. Tissu congelé – Prélèvement : 7 mm ; 7 glomérules analysables dont un globalement scléreux transformé en pain à cacheter. Etude en immunofluorescence : - Anticorps anti-IgG : fixation glomérulaire exclusive : dépôts granuleux mésangiaux, sous endothéliaux diffus et dépôts extramembraneux segmentaires sur 3 glomérules (3+) - Anticorps anti-IgA : dépôts granuleux mésangiaux et sous endothéliaux (2-3+) 
Robot:
1- Number of glomerules: 46
2- Number of fibrous/sclerous glomerules: 2 
3- Endo-capillar proliferation: 7
4- Cellular or fibrocellular crescents: 3
5- Total percentage of glomeruli with endocapillar proliferation or cellular crescents: <50%
6- Mesengial proliferation: no
7- Extra-membraneux immune deposits: no
- Class: III
 
Doctor: 
ID :C1412466 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 15 mm, cortico-médullaire. Glomérules : 33 glomérules analysables dont 12 globalement scléreux, transformés en pain à cacheter. Sur 12 glomérules, présence de croissants cellulaires associés à des ruptures de la membrane basale glomérulaire et à la présence de dépôts de fibrine sur un glomérule. Il s’y associe également des lésions prolifératives globales endocapillaires ainsi qu’une prolifération mésangiale. Sur deux glomérules, on met en évidence la présence de Wire loop et dans les zones prolifératives extra capillaires, on observe fréquemment une caryorrhexie. Absence de thrombus. Absence de remaniement des membranes basales à type de spicule ou chaînette. Tubes : atrophie tubulaire très légère. Vaisseaux : deux artères analysables dépourvues de fibrose. Absence d’artérite. Tissu interstitiel : fibrose interstitielle très légère. Infiltrat inflammatoire lymphocytaire dense et compact sur une surface représentant environ 5 % de la surface corticale. Coloration du Rouge congo : absence de dépôt amyloïde. Tissu congelé – Prélèvement : 13 mm ; 11 glomérules analysables dont 4 globalement scléreux. Etude en immunofluorescence : - Anticorps anti IgG et anti C1q : dépôts mésangiaux et sous endothéliaux intenses (3+). - Anticorps anti C3 : dépôts granuleux mésangiaux et sous endothéliaux (2+/3+). - Anticorps anti IgA : dépôts granuleux mésangiaux et sous endothéliaux (1+/2+). - Anticorps anti IgM : dépôts granuleux mésangiaux et sous endothéliaux (1+). 
Robot:
1- Number of glomerules: 33
2- Number of fibrous/sclerous glomerules: 12 
3- Endo-capillar proliferation: 100%
4- Cellular or fibrocellular crescents: 12
5- Total percentage of glomeruli with endocapillar proliferation or cellular crescents: >50%
6- Mesengial proliferation: yes
7- Extra-membraneux immune deposits: no
- Class: IV
'''],
['Chatbot', "Understood, I will respond according to the template."]
]


shots_berden=[
['Doctor', '''I will present you a pathology report. I want you to extract a list of information from it. You will answer only according to this template: 
- Number of glomerules: [number]
- Number of fibrous/sclerous glomerules: [number] 
- Number of cellular crescents: [number] 
 
Here are a few examples :

Doctor: 
ID :1743743 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 11 mm, corticale et médullaire Glomérules : 27 glomérules analysables dont 3 globalement scléreux transformés en pain à cacheter. Lésions glomérulaires d’âges différents associant prolifération extracapillaire majoritairement cellulaire, rupture des membranes basales glomérulaires, abondants dépôts de fibrine en flammèche. On dénombre 10 croissants cellulaires, 3 croissants fibrocellulaires et un croissant fibreux. Présence d’éléments inflammatoires au sein des lumières des capillaires. Absence d’amylose. Tubes : lésions modérément abondantes de nécrose tubulaire aiguë. Quelques cylindres hématiques. Atrophie tubulaire légère. Vaisseaux : une artère analysable dépourvue de fibrose. Absence d’infiltrat inflammatoire des parois vasculaires. Absence de thrombose. Absence d’anomalie artériolaire. Tissu interstitiel : fibrose interstitielle légère. Infiltrat inflammatoire modérément abondant composé de lymphocytes, de plasmocytes.
Robot:
- Number of glomerules: 27
- Number of fibrous/sclerous glomerules: 3 
- Number of cellular crescents: 10
 
Doctor: 
ID :9581843 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 1 cm de long, corticale. Glomérules : 19 dont 3 entièrement fibreux en pain à cacheter. Cinq glomérules sont de morphologie normale. Onze glomérules sont le siège d’une prolifération extracapillaire sous forme d’un croissant hémi-circonférentiel, cellulaire. Deux d’entre eux comportent une zone extracapillaire en voie de fibrose avec synéchie flocculo-capsulaire. Le secteur endo-capillaire est de morphologie normale. Tubes : atrophie tubulaire légère. Vaisseaux : discrète lésion d’artériosclérose. Tissu interstitiel : infiltrat inflammatoire très abondant diffus à prédominance de lymphocytes et de plasmocytes. Fibrose interstitielle légère. Coloration du rouge congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 19
- Number of fibrous/sclerous glomerules: 3 
- Number of cellular crescents: 11
 
Doctor: 
ID :101212 PONCTION BIOPSIE RÉNALE Tissu fixé : AFA – colorations : HES, Trichrome de Masson, Jones, Rouge Congo, PAS. Prélèvement : 0,7 cm de long. Corticale. Glomérules : 8 dont 2 fibreux, entièrement hyalinisés. Sur les 6 glomérules non fibreux, 2 comportent une prolifération extracapillaire sous forme d’un croissant hémicirconférentiel en voie de fibrose, mixte, cellulaire et fibreux. Le secteur endocapillaire est de morphologie normale. Présence d’un épaississement mésangial modéré. Sur l’un des glomérules, lésion de hyalinose segmentaire avec hypertrophie podocytaire en regard. Tubes : atrophie tubulaire sévère. Sur les tubes restants, lésions tubulaires aiguës sous forme d’une vacuolisation cytoplasmique et d’une desquamation du pôle apical de la cellule tubulaire voire d’une mise à nu de la membrane basale. Vaisseaux : une section artérielle siège d’une fibrose intimale modérée. Absence d’anomalie artériolaire. Tissu interstitiel : fibrose interstitielle sévère diffuse, parsemée de quelques éléments inflammatoires lymphocytaires et plasmocytaires peu nombreux. Coloration du rouge Congo : absence de dépôt amyloïde.
Robot:
- Number of glomerules: 8
- Number of fibrous/sclerous glomerules: 2 
- Number of cellular crescents: 0
'''],
['Chatbot', "Understood, I will respond according to the template."]
]

shots_gliome=[
['Doctor', '''I will present you a radiology report from a patient with a glioma. I want you to extract the size from the target lesions in the report and in the previous exam. It is crucial that you report only taget ("cible") lesions. There may be no target lesion. You must respond according to this template: 
- Target X today: [size in the report]
- Target X previously: [previous size]
(Create a new line for every new measurement. Include ONLY lesions that are explicit TARGET)
 
Here are a few examples :

Doctor: 
Lésion cible : Discrète majoration de la prise de contraste nodulaire en regard de l’infiltration lésionnelle flair située au contact de la corne frontale du ventricule latéral gauche mesurée à 22 x 10 mm dans le plan coronal contre 20 x 10 mm précédemment. Absence d'hyperperfusion en regard de cette prise de contraste. Meilleure visibilité d’une prise de contraste nodulaire apparue sur l’IRM précédente au sein de la substance blanche frontale gauche à proximité de l’infiltration lésionnelle décrite ci-dessus, mesurée à 7 mm de grand axe contre 5 mm précédemment. Absence d’hyperperfusion au sein de cette prise de contraste. Stabilité de l'infiltration lésionnelle en hypersignal flair fronto-cingulaire gauche. Cavité opératoire fronto-cingulaire droite ouverte au sein de la corne frontale avec stabilité de la prise de contraste linéaire de sa berge postérieure et des hypersignaux flair au pourtour. Multiples microsaignements anciens de la substance blanche sous corticale du vertex en rapport avec des cavernomes post radiques. 
Robot:
- Target 1 today: 22x10
- Target 1 previously: 20x10
 
Doctor: 
Résultats comparaison à l'IRM du 10 février 2022. IRM 10/02/22 IRM19/05/22 Lésions cibles et non cibles : Nette majoration de l'infiltration lésionnelle cortico-sous-corticale en hypersignal flair avec prise de contraste et hyperperfusion (rCBV max: 3,8) en son sein intéressant la région frontale droite, s'étendant au corps calleux (genou et tiers antérieur), au striatum et à la région insulaire antérieure : - Lésion fronto-calleuse droite (cible 1) : 56 x 41 mm contre 36 x 45 mm précédemment ; - Lésion striatum droite (cible 2) : 36 x 33 mm contre 12 x 16 mm ; - Lésion frontale supérieure droite (cible 3) : 30 x 35 mm (non mesurable précédemment, lésion non cible). Apparition d'un effet de masse sur les cornes frontales des ventricules latéraux. Stabilité de la déviation millimétrique de la ligne médiane vers la droite. Remaniements post thérapeutique : Stabilité des plages de leucopathie de la substance blanche profonde et périventriculaire fronto-pariétale bilatérale. Stabilité de la fine prise de contraste pachyméningée frontotemporale droite en regard de la voie d'abord. 
Robot:
- Target 1 today: 56x41
- Target 1 previously: 36x45
- Target 2 today: 36x33
- Target 2 previsouly: 12x16
- Target 3 today: 30x35
- Target 3 previously: non mesurable
 
Doctor: 
Résultats comparaison à l'IRM du 16 mai 2022. Lésion cible : -Majoration en taille de la lésion temporale supérieure gauche mesurée à 14 x 11 mm dans le plan coronal contre 11 mm de grand axe précédemment ainsi que de la plage d'œdème aux pourtours. Lésions non cible : Stabilité en taille de la lésion non cible à centre nécrotique temporo-insulaire gauche mesurée à 29 x 30 mm dans le plan axial avec diminution de l'épaisseur des prises de contraste pariétales et de la plage d'œdème aux pourtours. Stabilité en taille du système ventriculaire. Lésion pariétale supérieure gauche rehaussée : mesurant 8,9 x 6,8 mm et présentant une hyperperfusion avec rCBV corr de 2,7. Éléments médians en place. Aspect inchangé des discrets hypersignaux flair focaux non spécifiques de la substance périventriculaire et sous corticale bifrontale. Stabilité en taille de la lésion rehaussée extra-axiale frontale moyenne droite de 4 mm de grand axe compatible avec un méningiome. 
Robot:
- Target 1 today: 14x11
- Target 1 previously: 11
 
Doctor:
Résultats comparaison aux IRM antérieures de 2022. Lésions cibles : absence Lésions non cible : Stabilité de l'infiltration lésionnelle en hypersignal flair intéressant les régions insulaire antérieure, frontale interne et fronto-orbitaire postérieure gauches ainsi que le genou du corps calleux. Meilleure visibilité d'une prise de contraste linéaire apparue sur l'IRM du 21 novembre 2022, non spécifique, située en profondeur de l'infiltration lésionnelle insulaire, au contact d'une formation kystique (cf. Images clés sur le PACS). Remaniements post thérapeutiques : Stabilité de la plage en hypersignal flair frontale supérieure gauche, sans hyperperfusion ni restriction de diffusion en son sein et n’atteignant pas le cortex, compatible avec un remaniement post-thérapeutique. Stabilité des fine prises de contraste des parois de la cavité opératoire frontale gauche d'allure réactionnelle. Stabilité en taille du système ventriculaire. 
Robot:
- No target lesion
'''],
['Chatbot', "Understood, I will respond according to the template."]
]

shots_gliome_indic=[
['Doctor', '''I will present you a radiology report from a patient with a glioma. I want you to extract information from the indication section. You must respond according to this template: 
- Surgery ?: [yes or no (biopsy is not surgery)]
- Radiation therapy ?: [yes or no]
- Chemotherapy ?: [yes or no ?]
- Mutation ?: [none mentioned/negative ones/positive ones (when MGMT is hypermethylated it is positive)]
 
Here are a few examples :

Doctor: 
Indication:
Surveillance d'un glioblastome diagnostiqué sur des biopsies stéréotaxiques puis pris en
charge par radiochimiothérapie concomitante de juin à juillet 2020 suivie de 6 cycles de
Témodal jusqu'en janvier 2021 associée à de l'Avastin d'octobre 2020 à mars 2021.
Progression sur l'IRM de juin 2021 avec reprise de l’Avastin jusqu'en juillet 2021
(traitement arrêté en raison d'une cytolyse hépatique).
Robot:
- Surgery ?: no
- Radiation therapy ?: yes
- Chemotherapy ?: yes
- Mutation ?: none mentioned
 
Doctor: 
Indication :
Surveillance chez une patiente aux antécédents de glioblastome frontal supérieur gauche non hyperméthylé MGMT, avec amplification du gène EGFR. Prise en charge initiale par
chirurgie en mars 2018 puis radiochimiothérapie concomitante et 6 cycles de Témodal jusqu'en mars 2019.
Progression en janvier 2022 prise en charge par Témodal.
Robot:
- Surgery ?: yes
- Radiation therapy ?: yes
- Chemotherapy ?: yes
- Mutation ?: MGMT negative / EGFR positive
 
Doctor:
Indication:
Surveillance dans le cadre d'un astrocytome IDH1 non muté de grade IV diagnostiqué sur des biopsies.
Radiochimiothérapie selon le protocole STUPP avec fin de la radiothérapie le 21 janvier 2022.
Chimiothérapie adjuvante par Témodal.
Robot:
- Surgery ?: no
- Radiation therapy ?: yes
- Chemotherapy ?: yes
- Mutation ?: IDH1 negative
'''],
['Chatbot', "Understood, I will respond according to the template."]
]

shots_anapath=[
['Doctor', '''I will present you a pathology report. I want you to tell me which antibodies are positives. You must answer with a structured list. You must only include antibodies cited in the text, but you must include all of them. "0/1+" or "0-1+" antibodies are considered POSITIVE. You will answer only according to this template: 
- Positive antibodies : 
-- List of positives antibodies (with number of + if noted)
- Negative antibodies :
-- List of negative antibodies
 
Here are a few examples :

Doctor: Etude en immunofluorescence : - Anticorps anti-IgA : marquage glomérulaire à prédominance mésangiale, granuleux (2+) - Anticorps anti-IgG : marquage glomérulaire à prédominance mésangiale, granuleux (3+) - Anticorps anti-IgM : marquage glomérulaire à prédominance mésangiale, granuleux (2+) - Anticorps anti-C3 : marquage glomérulaire à prédominance mésangiale, granuleux (2+) - Anticorps anti-C1q : marquage glomérulaire à prédominance mésangiale, granuleux (3+) 
Expected Robot response :
- Positive antibodies : 
-- IgA (2+)
-- IgG (3+)
-- IgM (2+)
-- C3 (2+)
-- C1q (3+)
- Negative antibodies
-- None
 
Doctor: Etude en immunofluorescence : - Anticorps anti-IgA, anti-IgG, anti-IgM, anti-kappa, anti-lambda, anti-C1q, anti-C3 : absence de dépôt significatif. Anticorps anti-chaîne alpha 5 et 3 du collagène IV : expression conservée. 
Expected Robot response :
- Positive antibodies : 
-- chaîne alpha 5 et 3 du collagène IV
- Negative antibodies :
-- IgA
-- IgG
-- IgM
-- kappa
-- lambda
-- C1q
-- C3
 
Doctor: Etude en immunofluorescence : - Anticorps anti-IgM, anti-lambda : dépôts granuleux, diffus et globaux, peu abondants, disposés le long des capillaires glomérulaires (2+) - Anticorps anti-IgA, anti-IgG, anti-C1q, anti-C3, anti-kappa : absence de dépôt immun.
Expected Robot response :
- Positive antibodies : 
-- IgM (2+)
-- lambda (2+)
- Negative antibodies :
-- IgA
-- IgG
-- C1q
-- C3
-- kappa
 
Doctor: Etude en immunofluorescence : - Anticorps anti-IgA, anti-IgG, anti-IgM, anti-kappa, anti-lambda, anti-C1q : absence de dépôt significatif. - Anticorps anti-C3 : fins dépôts granuleux mésangiaux sur un glomérule (0-1+).
Expected Robot response : 
- Positive antibodies :
-- C3 (0-1+)
- Negative antibodies :
-- IgA
-- IgG
-- IgM
-- kappa
-- lambda
-- C1q
 
Doctor: Etude en immunofluorescence : Etude en immunofluorescence : - Anticorps anti-IgA, anti-IgG, anti-IgM, anti-kappa, anti-lambda, anti-C1q : absence de dépôt immun. - Anticorps anti-C3 : rares dépôts segmentaires au niveau des zones de nécrose. - Anticorps anti-fibrinogène : fixation intense segmentaire au niveau de 5 glomérules.
Expected Robot response : 
- Positive antibodies :
-- C3
-- fibrinogène
- Negative antibodies :
-- IgA
-- IgG
-- IgM
-- kappa
-- lambda
-- C1q
 
Doctor: Etude en immunofluorescence : - Anticorps anti IgG, anti C1q, anti C3, anti Lambda : fins dépôts granuleux segmentaires sur un glomérule de topographie endomembraneuse possiblement au pôle vasculaire. - Anticorps anti IgA, anti IgM, anti Kappa : absence de dépôt significatif.
- Positive antibodies :
-- IgG
-- C1q
-- C3
-- Lambda
- Negative antibodies :
-- IgA
-- IgM
-- Kappa'''],
['Chatbot', "Understood, I will respond according to the template."],
]


shots_immuno=[
['Doctor', '''I will present you a pathology report. I am interested in anti-IgA, anti-IgG, anti-IgM, anti-C1q, anti-C3, anti-kappa and anti-lambda immunohistochemistry. For each of those, I want to know if they are positive and if so, I want to know if they include any of the caratcteristics of interest: "granuleux", "périphérique", "linéaire", "mésangial", "podocytaire" and/or "sous-endothélial". They are the only caracteristics of interest, I do not care about the rest, do not include any other word in your response. It is crucial that you answer only according to this template: 
- antibody : [+ or -] // granuleux [yes or no] // périphérique [yes or no] // linéaire [yes or no] // mésangial [yes or no] // podocytaire [yes or no] // sous-endothélial [yes or no]
- etc
 
Here are a few examples :

Doctor: Etude en immunofluorescence : - Anticorps anti-IgA : marquage glomérulaire segmentaire et focal à prédominance mésangiale, granuleux (2+) - Anticorps anti-IgG : marquage glomérulaire à prédominance mésangiale, granuleux (3+) - Anticorps anti-IgM et C3 : marquage glomérulaire à prédominance mésangiale, granuleux (2+) - Anticorps anti-C1q : rare marquage sous-endothélial linéaire (0/1+) - Anticorps anti-kappa et anti-lambda : absence de marquage significatif
Expected Robot response :
- anti-IgA : 2+ // granuleux: yes // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-IgG : 3+ // granuleux: yes // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-IgM : 2+ // granuleux: yes // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-C1q : 0/1+ // granuleux: no // périphérique: no // linéaire: yes // mésangial: yes // podocytaire: no // sous-endothélial: yes
- anti-C3 : 2+ // granuleux: yes // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-kappa : - // N/A
- anti-lambda : - // N/A 
 
Doctor: Etude en immunofluorescence : - Anticorps anti-IgA, anti-IgG, anti-IgM, anti-kappa, anti-lambda, anti-C1q, anti-C3 : absence de dépôt significatif. 
Expected Robot response :
- anti-IgA : - // N/A 
- anti-IgG : - // N/A 
- anti-IgM : - // N/A 
- anti-C1q : - // N/A 
- anti-C3 : - // N/A 
- anti-kappa : - // N/A 
- anti-lambda : - // N/A
 
Doctor: Etude en immunofluorescence : - Anticorps anti-IgM, anti-lambda, anti-C1q : dépôts granuleux, segmentaire et globaux, peu abondants, endomembraneux disposés le long des capillaires glomérulaires (2+) - Anticorps anti-IgA, anti-IgG, anti-C3, anti-kappa : absence de dépôt immun.
Expected Robot response :
- anti-IgA : - // N/A
- anti-IgG : - // N/A
- anti-IgM : 2+ // granuleux: yes // périphérique: no // linéaire: no // mésangial: no // podocytaire: no // sous-endothélial: no
- anti-C1q : 2+ // granuleux: yes // périphérique: no // linéaire: no // mésangial: no // podocytaire: no // sous-endothélial: no
- anti-C3 : - // N/A
- anti-kappa : - // N/A
- anti-lambda : 2+ // granuleux: yes // périphérique: no // linéaire: no // mésangial: no // podocytaire: no // sous-endothélial: no
 
Doctor: Etude en immunofluorescence : - Anticorps anti-IgA, anti-IgG, anti-IgM, anti-kappa, anti-lambda, anti-C1q : absence de dépôt significatif. - Anticorps anti-C3 : fins dépôts granuleux mésangiaux sur un glomérule (0-1+).
Expected Robot response : 
- anti-IgA : - // N/A
- anti-IgG : - // N/A
- anti-IgM : - // N/A
- anti-C1q : - // N/A
- anti-C3 : 0-1+ // granuleux: yes // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-kappa : - // N/A
- anti-lambda : - // N/A
 
Doctor: Etude en immunofluorescence : - Anticorps anti-IgA et anti-C3 : dépôts mésangiaux très abondants segmentaire et focal (+++). - Anticorps anti-IgG et anti-M : dépôts mésangiaux peu abondants sur tous les glomérules (+). - Anticorps anti-kappa et anti-Lambda : dépôts mésangiaux très abondants sur tous les glomérules (++ kappa ; +++ lambda). - Anticorps anti-C1q : absence de dépôt glomérulaire.
- anti-IgA : +++ // granuleux: no // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-IgG : + // granuleux: no // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-IgM : + // granuleux: no // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-C1q : - // N/A
- anti-C3 : +++ // granuleux: no // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-kappa : ++ // granuleux: no // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
- anti-lambda : +++ // granuleux: no // périphérique: no // linéaire: no // mésangial: yes // podocytaire: no // sous-endothélial: no
'''],
['Chatbot', "Understood, I will respond according to the template."],
]

shots_indic_fr=[
['Docteur', '''Je vais te montrer un court contexte clinique. Tu dois me dire s'il mentionne EXPLICITEMENT des céphalées ou maux de tête ou migraine. Les patients peuvent avoir d'autres symptômes mais ils ne m'intéressent pas. Tu dois me répondre sous ce format uniquement: 
Céphalées : [oui ou non]
 
Voici quelques exemples de ce que j'attends de toi. Je t'ai mis entre crochets la mention de céphalées dans le texte (tu es le Robot, je suis le Docteur) :

Docteur: Suspicion de NORB gauche.
Robot: Céphalées : non.

Docteur : Suspicion d'AVC. Déficit neurologique à 16 heures avec dysarthrie et déficit du membre supérieur droit. Notion de troubles de l'équilibre brutaux. Vertiges.
Robot: Céphalées : non.

Docteur : Bilan d'une masse découverte sur un scanner réalisé pour vertige, diplopie et 3e épisode de céphalées [céphalées].
Robot: Céphalées : oui.

Docteur : Patient de 14 ans ayant été pris en charge début août 2021 pour une régression psychomotrice d'origine indeterminée. Depuis une semaine, vertiges, flou visuel associés à des migraines [céphalées] ne cédant pas aux antalgiques.
Robot: Céphalées : oui.

Docteur : Bilan complémentaire d'un épisode de vomissements en jet avec hémiparésie senstive modérée depuis 48h. Depuis quatre heures, apparition de céphalées [céphalées] intenses pulsatiles sans signe de focalisation ni déficit focal. Suspicion de thrombose veineuse cérébrale.
Robot: Céphalées : oui.

Docteur : Patiente de 38 ans enceinte à 32 SA (grossesse sans complication) présentant une perte du mot dans un contexte de céphalées [céphalées] atypiques associées à une aura visuelle. Notion d'epistaxis intense il y a deux semaines traitée par embolisation. Suspicion de thrombose veineuse cérébrale.
Robot: Céphalées : oui.
 
Docteur : Alerte thrombolyse avec vertige brutal
Robot : Céphalées : non.
 
Docteur : Suspicion de dissection ou de thrombose veineuse cérébrale
Dobot : Céphalées : non.
 
Tu es prêt pour le texte ?'''],
['Robot', "Je suis prêt à lister les symptômes du texte et à indiquer s'ils mentionnent explicitement des céphalées"]
]

shots_anev=[
['Doctor', '''I will present you a radiology report. I want you to tell me if any aneurysm is described. I am even interested in aneurysms that are stable, or not permable, so please be sure to not miss any. Aneurysms are sometimes described as addition image. Please answer using the following template:
- Mention of aneurysm: [Sentence mentioning aneurysm in the report]
- Interpretation: [Based on the previous sentence, explain why an aneurysm is present]
- Aneurism: [present or not present]"
 
Here are a three examples:
Doctor: 13354580.0 Absence de lésion ischémique ou hémorragique. Absence d’hémosidérose corticale. Les repères en T1 spontané de la ligne médiane sont en place. Intégrité du système ventriculaire. Pas de processus expansif intracrânien. Intégrité de l’ARM veineuse et remplissage normal du système veineux sur la séquence volumique T1 après injection de gadolinium. Pas d'anévrisme. Anomalie veineuse de développement cérébelleuse droite, sans cavernome associé. Absence de prise de contraste pathologique. Kyste sous-muqueux du sinus maxillaire gauche. CONCLUSION : Absence de thrombose veineuse cérébrale. Absence d’anomalie expliquant la symptomatologie.
Robot: - Mention of aneurysm: Pas d'anévrisme.
- Interpretation: No aneurysm is present.
- Aneurysm: not present
 
Doctor: 14076014.0 ence d’IRM cérébrale antérieure pour comparaison. Absence de lésion ischémique ou hémorragique récente. Hyposignaux T2* para-falcoriels prédominant à droite, non visibles en Flair et sans réhaussement après injection, compatible en première hypothèse avec un méningiome complètement calcifié. Rares hypersignaux Flair millimétriques de la substance blanche profonde et sous-corticale, non spécifiques. Au niveau vasculaire, Perméabilité des artères du polygone de Willis sans disparité de calibre. Hypoplasie de la portion A1 de l’artère cérébrale antérieure droite. Sur environ 2cm de hauteur, sténose focale non significative de l’artère carotide interne gauche sous-pétreuse associé à un hypersignal T1 spontané de sa paroi interne compatible avec un hématome de paroi. Dilatation pseudo-anévrismale interne juste en aval de la sténose. Probable flap intimal difficilement visible. Le tout est compatible avec une dissection artérielle avec faux chenal circulant. Perméabilité sans disparité de calibre des autres artères des TSA Perméabilité des sinus veineux et des veines cérébrales internes. , le 
Robot: - Mention of aneurysm: Dilatation pseudo-anévrismale interne juste en aval de la sténose.
- Interpretation: There is an aneurysmal dilatation of the vessel.
- Aneurysm: present
 
Doctor: 12464424.0 Absence de lésion ischémique ou hémorragique récente en particulier, pas d'hémorragie sous-arachnoïdienne. Globale stabilité des hypersignaux punctiformes de la substance blanche périventriculaire et profonde, compatibles avec une leucopathie classée FAZEKAS 1. Deux microsaignements chroniques : fronto-orbitaire droit et occipital droit, déjà présents sur la précédente IRM. Pas de recanalisation de l'anévrisme carotidien traité stable. CONCLUSION: IRM encéphalique normale.
Robot: - Mention of aneurysm: Pas de recanalisation de l'anévrisme carotidien traité stable.
- Interpretation: A treated aneurysm is mentioned, so an aneurysm is present. 
- Aneurysm: present

Doctor: 14139688.0 paraison aux IRM de septembre 2021. Absence de lésion ischémique ou hémorragique récente. Séquelles ischémiques lenticulo-caudée et fronto-insulaire gauches en rapport avec l'événement ischémique de septembre 2021. Séquelle connue cortico-sous-corticale frontale supérieure gauche. Stabilité des 4 microsaignements anciens de topographie pallidale et fronto- insulaire gauche en regard des séquelles ischémiques. Stabilité des hypersignaux flair de la substance blanche périventriculaire prédominant au niveau des carrefours ainsi que de la substance blanche profonde en rapport avec une leucopathie d'origine vasculaire probable. Éléments médians en place. Morphologie normale du système ventriculaire. Absence de prise de contraste pathologique parenchymateuse. Stabilité de la prise de contraste millimétrique du bord libre de l'hémi tente droite du cervelet : méningiome ?  Anomalie veineuse de développement temporale gauche. Hypersignal flair diffus du nerf optique droit, sans prise de contraste associée décelable. Prise de contraste diffuse du nerf optique gauche sans anomalie de signal en FLAIR associée, à confronter à la clinique. Absence d'hématome de paroi des carotides internes et des artères vertébrales. Perméabilité des artères du polygone de Willis avec aspect grêle et irrégulier des branches sylviennes gauches en M2 distale et M3, pouvant être en rapport avec l'événement ischémique de septembre 2021. Perméabilité des carotides internes et des artères vertébrales sans sténose. CONCLUSION Absence de lésion ischémique ou hémorragique récente. Pas d'argument pour une dissection carotidienne ou vertébrale. Pas d'argument pour une thrombose veineuse cérébrale. Hypersignal flair diffus du nerf optique droit, sans prise de contraste associée décelable. Prise de contraste diffuse du nerf optique gauche sans anomalie de signal en FLAIR associée. L’ensemble de ces anomalies sont à confronter à la clinique : névrites optiques ? 
Robot: - Mention of aneurysm: No mention.
- Interpretation: None.
- Aneurysm: not present

 ''']
]

shots_pdl1=[
['Doctor', '''I will present you a pathology report. I want you to tell me the percentage of positive cells for PDL1. I don't care about ROS or ALK, just PDL1. Respond only like this: "- Pathological diagnosis: [report the pathological diagnosis] 
- Percentage of PD-L1 cells: [% or NA if the info is missing]"
 
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
        log_prog=[]
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
            log_prog.append(output_log["token_logprobs"][index+1])
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
