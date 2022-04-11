get_ipython().system('pip install ohmeow-blurr -q')
get_ipython().system('pip install bert-score -q')

import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *

df = pd.read_csv('../input/news-summary/news_summary_more.csv')

df['text'] = df['text'].apply(lambda x: x.replace('/',''))
df['text'] = df['text'].apply(lambda x: x.replace('\xa0',''))
df.head()

articles = df.head(17000)

pretrained_model_name = "facebook/bart-large-cnn"
hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name, 
                                                                  model_cls=BartForConditionalGeneration)

hf_batch_tfm = HF_Seq2SeqBeforeBatchTransform(hf_arch, hf_config, hf_tokenizer, hf_model, task='summarization',
text_gen_kwargs={'max_length': 248,
 'min_length': 56,
 'do_sample': False,
 'early_stopping': True,
 'num_beams': 4,
 'temperature': 1.0,
 'top_k': 50,
 'top_p': 1.0,
 'repetition_penalty': 1.0,
 'bad_words_ids': None,
 'bos_token_id': 0,
 'pad_token_id': 1,
 'eos_token_id': 2,
 'length_penalty': 2.0,
 'no_repeat_ngram_size': 3,
 'encoder_no_repeat_ngram_size': 0,
 'num_return_sequences': 1,
 'decoder_start_token_id': 2,
 'use_cache': True,
 'num_beam_groups': 1,
 'diversity_penalty': 0.0,
 'output_attentions': False,
 'output_hidden_states': False,
 'output_scores': False,
 'return_dict_in_generate': False,
 'forced_bos_token_id': 0,
 'forced_eos_token_id': 2,
 'remove_invalid_values': False
                })

blocks = (HF_Seq2SeqBlock(before_batch_tfm=hf_batch_tfm), noop)

dblock = DataBlock(blocks=blocks, get_x=ColReader('text'), get_y=ColReader('headlines'), splitter=RandomSplitter())

dls = dblock.dataloaders(articles, bs=2)
seq2seq_metrics = {
        'rouge': {
            'compute_kwargs': { 'rouge_types': ["rouge1", "rouge2", "rougeL"], 'use_stemmer': True },
            'returns': ["rouge1", "rouge2", "rougeL"]
        },
        'bertscore': {
            'compute_kwargs': { 'lang': 'fr' },
            'returns': ["precision", "recall", "f1"]
        }
    }
get_ipython().system('pip install GPUtil')

import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()     

model = HF_BaseModelWrapper(hf_model)
learn_cbs = [HF_BaseModelCallback]
fit_cbs = [HF_Seq2SeqMetricsCallback(custom_metrics=seq2seq_metrics)]

learn = Learner(dls, 
                model,
                opt_func=ranger,
                loss_func=CrossEntropyLossFlat(),
                cbs=learn_cbs,
                splitter=partial(seq2seq_splitter, arch=hf_arch)).to_fp16()

learn.create_opt() 
learn.freeze()

learn.fit_one_cycle(6, lr_max=3e-5, cbs=fit_cbs)

outputs = learn.blurr_generate(df['text'][4], early_stopping=False, num_return_sequences=1)

for idx, o in enumerate(outputs):
    print(f'=== Prediction {idx+1} ===\n{o}\n')

learn.metrics = None
learn.export(fname='best.pkl')

inf_learn = load_learner(fname='best.pkl')

test_article_1 = "Rezantsev was the commander of Russia's 49th combined army.A western official said he was the seventh general to die in Ukraine, and the second lieutenant general - the highest rank officer reportedly killed.It is thought that low morale among Russian troops has forced senior officers closer to the front line.In a conversation intercepted by the Ukrainian military, a Russian soldier complained that Rezantsev had claimed the war would be over within hours, just four days after it began.Ukrainian media reported on Friday that the general was killed at the Chornobaivka airbase near Kherson, which Russia is using as a command post and has been attacked by Ukraine's military several times.Another lieutenant general, Andrei Mordvichev, was reportedly killed by a Ukrainian strike on the same base.Kherson was the first Ukrainian city to be occupied by Russian forces, although there are reports that daily protests are held there against the Russian occupation.Although Russia has confirmed the death of only one general, Kyiv and western officials believe up to seven have been killed in fighting since the war began.However the death of Maj Gen Magomed Tushayev of the Chechen national guard has been disputed.It is unusual for such senior Russian officers to be so close to the battlefield, and western officials believe that they have been forced to move towards the front lines to deal with low morale among Russian troops.The unexpectedly strong Ukrainian resistance, poor Russian equipment and a high death toll amongst Russian troops are all thought to be contributing to the low morale.Russian forces are believed to be relying in part on open communication systems, for example mobile phones and analogue radios, which are easy to intercept and could give away the locations of high-ranking officers.A person inside Ukrainian President Volodymyr Zelensky's inner circle told the Wall Street Journal that Ukraine had a military intelligence team dedicated to targeting Russia's officer class.So far, Vladimir Putin has only referred to the death of one general, thought to be Maj Gen Andrey Sukhovetsky, in a speech soon after the start of the war.Russia says 1,351 soldiers have died since the war began in Ukraine, although Kyiv and western officials say the number is much higher."

test_article_2 = "A team of officers of the Central Bureau of Investigation (CBI) on Saturday visited Bogtui village in West Bengal ‘s Birbhum district where eight persons were burnt to death on March 21 following the murder of a local Trinamool Congress leader. The Calcutta High Court on Friday directed the CBI to probe the violence in the “interest of justice”, “fair investigation” and to “instill confidence in society”.During the day, CBI officers led by DIG Akhilesh Singh visited the village and inspected the houses that were set on fire. The CBI team comprising over a dozen personnel went to the house of Sona Sheikh where seven persons were charred to death.The investigating officers went to the room from where the bodies were recovered and also to the terrace of the house. They also visited the other houses that were set on fire and used a digital three– dimensional scanning device to collect evidence.Forensic experts collected samples from the site for the second consecutive day. While the investigators were collecting evidence, TMC Rampurhat Block–I president Anaraul Hossain, who was arrested on the instructions of Chief Minister Mamata Banerjee, said “there has been a conspiracy” against him. The TMC leader, who was being taken to hospital by the police, told the media that he was at his house on the night of March 21 when the violence occurred and the police did not call him. The Chief Minister had blamed the local leader along with the Rampurhat Sub-Divisional Police Office for the violence. The SDPO has been removed and sent on compulsory waiting."

test_article_3 = """The defence ministry said that the initial aims of the war were complete, and that Russia had reduced the combat capacity of Ukraine.Russia's invasion appeared aimed at swiftly capturing major cities and toppling the government.But it has stalled in the face of fierce Ukrainian resistance."The main tasks of the first stage of the operation have been carried out," said Sergei Rudskoy, head of the General Staff's main operations administration."The combat capabilities of the Ukrainian armed forces have been substantially reduced, which allows us to concentrate our main efforts on achieving the main goal: the liberation of Donbas," he added, referring to an area in eastern Ukraine largely in the hands of Russian-backed separatists.Russia's military has been bombarding and trying to encircle key Ukrainian cities such as the capital Kyiv, which Gen Rudskoy characterised as an attempt to tie down Ukraine's forces elsewhere in the country while Russia focuses on the east.Ukraine's President, Volodymyr Zelensky, said his troops had landed "powerful blows" on Russia and called on Moscow to recognise the need for serious peace talks."By restraining Russia's actions, our defenders are leading the Russian leadership to a simple and logical idea: talk is necessary. Meaningful. Urgent. Fair. For the sake of the result, not for the sake of the delay," he said."""

test_article_4 = """ Leaked papers indicate a Chinese military base could potentially be set up on the island to Australia's north.That's sparked concern from Australia, long the chief defence partner and biggest aid donor to the tiny island.Both Australia and New Zealand said it had raised concerns with Honiara.Australia's Foreign Minister Marise Payne said she respected the Pacific island's right to make sovereign decisions but:"We would be particularly concerned by any actions that undermine the stability and security of our region, including the establishment of a permanent presence such as a military base."New Zealand said it was also concerned as the plan threatened to "destabilise the current institutions and arrangements that have long underpinned the Pacific region's security." """

inf_learn.blurr_generate(test_article_1)
inf_learn.blurr_generate(test_article_2)
inf_learn.blurr_generate(test_article_3)
inf_learn.blurr_generate(test_article_4)




