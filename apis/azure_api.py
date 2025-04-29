import torch
import numpy as np
from tqdm import tqdm
import logging
from .api import API
import random
import tiktoken
import copy
import openai
import re
import collections
from .utils import *
from .openai_chat import openai_completions
from .deepseek_chat import deepseek_completions
import time
from dpsda.refinement import *

MODEL_CONFIG = {
    'gpt-4o-mini': 
    {"openai_api_key":  "", # your azure openai key
    "openai_api_base": "",
     "engine": 'gpt-4o-mini',
                      },
}

DeepSeek_CONFIG = {
    'DeepSeek-R1': 
    {"openai_api_key":  "", # your azure key
    "openai_api_base": "",
     "engine": 'DeepSeek-R1',
                      },
}

# for gpt-4
#MAX_REQUESTS_PER_MINUTE = 42
#MAX_TOKENS_PER_MINUTE = 7000

#for gpt-4o-mini
MAX_REQUESTS_PER_MINUTE = 100
MAX_TOKENS_PER_MINUTE = 10000

# for gt-3.5-turbo
# MAX_REQUESTS_PER_MINUTE = 174
# MAX_TOKENS_PER_MINUTE = 29000


class MessageConstructor(object):
    def __init__(self, sys_demo, task_desc):
        self.sys_demo = sys_demo
        self.task_desc = task_desc

    def get_message(self, input):
        messages = [{"role": "system", "content": f"{self.sys_demo}"}]
        messages.append({"role": "user", "content": self.task_desc + input})
        return messages
    
class DSMessageConstructor(object):
    def __init__(self, sys_demo, task_desc):
        self.sys_demo = sys_demo
        self.task_desc = task_desc

    def get_message(self, input):
        combined_content = f"{self.sys_demo}\n\n{self.task_desc}\n\n{input}"
        messages = [{"role": "user", "content": combined_content}]
        return messages



class AzureAPI(API):
    def __init__(self,
                 model_type, variation_type, use_subcategory,
                 output_dir, seed, num_procs,
                 length, temperature, top_p, do_sample, no_cuda,
                 dynamic_len, min_target_word, word_var_scale, max_token_word_scale,
                 sleep_time,
                 max_token_limit, dry_run, r_data,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.engine = ''
        self.model_type = model_type
        print(f"model_type: {self.model_type}")
        print(f"MODEL_CONFIG.keys(): {MODEL_CONFIG.keys()}")
        if self.model_type not in MODEL_CONFIG.keys():
            raise ValueError(f'Unknown model type name {self.model_type}')
        else:
            self.openai_api_key = MODEL_CONFIG[self.model_type]['openai_api_key']
            self.openai_api_base = MODEL_CONFIG[self.model_type]['openai_api_base']
            self.engine = MODEL_CONFIG[self.model_type]['engine']
            openai.api_type = 'azure'  # here we use azure openai service
            #openai.api_version = '2024-08-01-preview' # for gpt-3.5
            openai.api_version = '2024-05-01-preview'

        openai.api_key = self.openai_api_key
        openai.api_base = self.openai_api_base

        self.num_procs = num_procs
        self.use_subcategory = use_subcategory
        available_datasets = ['yelp', 'openreview', 'pubmed', 'sd']

        self.variation_type = variation_type
        self.init_template = None
        self.var_template = None

        for current_dataset in available_datasets:
            if current_dataset in self.variation_type:
                self.init_template = f'init_{current_dataset}'
                self.var_template = f'variant_{current_dataset}'
                break

        if use_subcategory:
            self.subcategory_dict = {}
            for current_dataset in available_datasets:
                if current_dataset in self.variation_type:
                    self.subcategory_dict[current_dataset] = get_subcategories(
                        current_dataset)

        self.output_dir = output_dir
        self.length = length
        self.temperature = temperature
        self.p = top_p
        self.do_sample = do_sample
        self.no_cuda = no_cuda
        self.dynamic_len = dynamic_len
        self.min_target_word = min_target_word
        self.word_var_scale = word_var_scale
        self.max_token_word_scale = max_token_word_scale
        self.max_token_limit = max_token_limit
        self.dry_run = dry_run

        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        set_seed(seed=seed, n_gpu=self.n_gpu)

        self.sleep_time = sleep_time
        if 'sd' in self.variation_type or 'pubmed' in self.variation_type:
            from transformers import BertTokenizer
            self.mask_tokenizer = BertTokenizer.from_pretrained(
                "bert-base-cased",  mask_token="_")
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")      
        self.r_data = r_data

    @staticmethod
    def command_line_parser():
        parser = super(AzureAPI, AzureAPI).command_line_parser()
        parser.add_argument(
            '--model_type',
            type=str,
            default='gpt-4o-mini',
            help='Which model to use')

        parser.add_argument("--use_subcategory",
                            action="store_true", help="use subcategory")

        parser.add_argument(
            '--variation_type',
            type=str,
            default='pubmed_rephrase_tone',
            choices=["yelp_blank_fill_3_shot_word",
                     "openreview_blank_fill_1_shot_word",
                     "pubmed_blank_fill_0_shot_word",
                     "sd_blank_fill_0_shot_word",
                     "sd_rephrase_tone",
                     "pubmed_rephrase_tone",
                     ],
            help='Which image feature extractor to use')

        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
        )
        parser.add_argument("--length", type=int, default=128)
        parser.add_argument("--sleep_time", type=int, default=1)
        parser.add_argument("--max_token_limit", type=int, default=1200)
        parser.add_argument("--min_target_word", type=int, default=25)
        parser.add_argument("--word_var_scale", type=int, default=10)
        parser.add_argument("--max_token_word_scale", type=float, default=1.2)
        parser.add_argument("--temperature", type=float, default=1.0)

        parser.add_argument("--top_p", type=float, default=1)

        parser.add_argument("--num_procs", type=int, default=4)
        parser.add_argument("--do_sample", action="store_true",
                            help="sampling when generation")
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")

        parser.add_argument("--no_cuda", action="store_true",
                            help="Avoid using CUDA when available")
        parser.add_argument(
            "--dynamic_len", action="store_true", help="change max token")
        parser.add_argument("--dry_run", action="store_true", help="dry run")
        parser.add_argument("--r_data", type=int, default=1)
        return parser

    def enforce_rate_limit(self, tokens):
        global tokens_used, requests_made, start_time
        elapsed_time = time.time() - start_time

        if elapsed_time > 60:
            tokens_used = 0
            requests_made = 0
            start_time = time.time()

        if tokens_used + tokens > MAX_TOKENS_PER_MINUTE or requests_made >= MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - elapsed_time  # Wait the remaining time in the minute
            time.sleep(sleep_time)
            tokens_used = 0
            requests_made = 0
            start_time = time.time()

        tokens_used += tokens
        requests_made += 1
    
    def text_random_sampling(self, num_samples, prompt_counter=None, lens_dict=None):
        ratio_generation_training = num_samples / sum(prompt_counter.values())
        syn_samples = []
        additional_info = []
        sync_labels_counter = collections.Counter()

        message_constructor = MessageConstructor(
            PROMPTS_templates[self.init_template]['sys_prompt'], PROMPTS_templates[self.init_template]['task_desc'])
        all_prefix_prompts = []
        for prompt_text in tqdm(prompt_counter):
            prompt = prompt_text

            num_seq_to_generate = round(
                prompt_counter[prompt_text] * ratio_generation_training)
            if num_seq_to_generate > 0:
                max_lens_ls = [
                    self.length]*num_seq_to_generate if lens_dict is None else lens_dict[prompt_text]
                sequences, prefix_prompts = self._generate_text(prompt, num_seq_to_generate, message_constructor,
                                                                max_length=max_lens_ls)

                syn_samples += sequences
                all_prefix_prompts += prefix_prompts

                additional_info += [prompt_text] * len(sequences)

                sync_labels_counter[prompt_text] = len(sequences)

        return syn_samples, additional_info, sync_labels_counter, all_prefix_prompts

    def _generate_text(self, prompt, seq_num, message_constructor, max_length):
        all_data = []
        all_prompts = []
        all_prefix_prompts = []

        for i in tqdm(range(seq_num)):
            if self.use_subcategory:
                if "yelp" in self.variation_type:
                    category_label = prompt.split(
                        "\t")[0].replace('Business Category: ', '')
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['yelp'][category_label]))
                    keyword = self.subcategory_dict['yelp'][category_label][rand_keyword_idx]
                    prefix = f'{prompt} with keyword {keyword}'
                elif "openreview" in self.variation_type:
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['openreview']))
                    keyword = self.subcategory_dict['openreview'][rand_keyword_idx]
                    
                    prefix = f"Suppose that you are a {keyword}, and your answer should contain 1000 words.\n" + prompt
                elif "pubmed" in self.variation_type:
                    prefix = PUBMED_INIT_templates[random.randrange(
                        len(PUBMED_INIT_templates))]  # random select one template
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['pubmed']))
                    keyword = self.subcategory_dict['pubmed'][rand_keyword_idx]
                    prefix = f"Suppose that you are a {keyword}. " + prefix
                elif "sd" in self.variation_type:
                    prefix = SD_INIT_templates[random.randrange(
                        len(SD_INIT_templates))]  # random select one template
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['sd']))
                    keyword = self.subcategory_dict['sd'][rand_keyword_idx]
                    prefix = f"Suppose that you are a {keyword}. " + prefix
            else:
                prefix = prompt
            
            all_prompts.append(
                (message_constructor.get_message(prefix), max_length[i]))
            all_prefix_prompts.append(message_constructor.get_message(prefix))

        if self.dry_run:
            generations = [prompt[0][1]["content"] for prompt in all_prompts]
        else:
            generations = openai_completions(all_prompts, model_name=self.model_type, engine_name=self.engine, openai_api_keys=[self.openai_api_key], openai_api_base=self.openai_api_base,
                                             num_procs=self.num_procs,  top_p=self.p, temperature=self.temperature)['completions']

        for idx in range(len(generations)):
            seq = generations[idx]
            seq = " ".join(seq.split())

            if seq:
                all_data.append(seq)

        return all_data, all_prefix_prompts

    def text_variation(self, sequences, additional_info,
                       num_variations_per_sequence, variation_degree):

        if not (0 <= variation_degree <= 1):
            raise ValueError('variation_degree should be between 0 and 1')
        variations, var_labels,  all_target_words, all_gen_words, all_masked_prompts = self._text_variation_parallel(
            sequences=sequences,
            labels=list(additional_info),
            variation_degree=variation_degree,
            variation_type=self.variation_type,
            lookahead=num_variations_per_sequence)

        return np.stack(variations, axis=1), var_labels,  all_target_words, all_gen_words, all_masked_prompts

    def _create_masked_seq(self, input_sequence, mlm_probability=0.5):
        try:
            input_ids = self.encoding.encode(input_sequence)

            masked_indices = np.random.binomial(
                1, mlm_probability, size=len(input_ids)).astype(bool).tolist()

            new_input_ids = copy.deepcopy(input_ids)
            for (i, v) in enumerate(masked_indices):
                new_input_ids[i] = self.encoding.encode(
                    "_")[0] if v else input_ids[i]

            return self.encoding.decode(new_input_ids)
        except:
            return input_sequence

    def _rephrase(self, label, sequence, variation_type):

        target_word = len(sequence.split(
            " ")) + int(np.random.normal(0, self.mlm_probability*self.word_var_scale, 1)[0])
        target_word = max(target_word, self.min_target_word)

        masked_seq = self._create_masked_seq(sequence, self.mlm_probability)
        if "pubmed_blank_fill_0_shot_word" in variation_type:
            selected_style = ALL_PUBMED_styles[random.randrange(
                len(ALL_PUBMED_styles))]
            instruction = f"You are required to fill in the blanks with more details for the input medical abstract {selected_style}. If there is no blanks, please output the original medical abstract.\n"
            prompt = instruction + \
                f"Please fill in the blanks in the following sentences to write an abstract of a medical research paper: \"{masked_seq}\" \n"

        elif "pubmed_rephrase_tone" in variation_type:
            selected_style = ALL_PUBMED_styles[random.randrange(
                len(ALL_PUBMED_styles))]
            prompt = f"Rephrase the following sentences as an abstract for medical research paper :\"{sequence}\" \n"

        elif "sd_rephrase_tone" in variation_type:
            selected_style = ALL_SD_styles[random.randrange(
                len(ALL_SD_styles))]
            prompt = f"Use different words to rephrase the following sentences {selected_style}: \"{sequence}\"\n"
                
        elif "sd_blank_fill_0_shot_word" in variation_type:
            selected_style = ALL_SD_styles[random.randrange(
                len(ALL_SD_styles))]
            prompt = f"You are required to fill in the blanks with more details {selected_style} for the input passage. If there is no blanks, please output the original passage: \"{masked_seq}\"\n"

        elif "openreview_blank_fill_1_shot_word" in variation_type:
            selected_style = ALL_OPENREVIEW_styles[random.randrange(
                len(ALL_OPENREVIEW_styles))]
            instruction = f"Based on the area and final decision of a research paper, you are required to fill in the blanks for the input sentences **{selected_style}**. If there is no blanks, please output the original input sentences.\n"

            demos = [
                {'label': 'Area: Applications (eg, speech processing, computer vision, NLP)\tRecommendation: 3: reject, not good enough',
                 'input_word': 'This paper proposes an attention generation _ for ROI detection by adversarial counterfactual without attention label. The attention map can _ used _ highlight useful information for disease classification and detection. _ experiments show its _ on different medical imaging tasks. Strengths: --The idea using _ images for _ map _ is interesting. --The _ _ medical imaging taks is significant. Weaknesses: --The novelty is _ and _ --More _ are needed, such as _ counterfactual generation. _ proposed method is interesting, but the novelty is limited',
                 'input': '__ proposes an__ method_ ROI detection__arial_f_ without attention_. The_ map can_ used____ for__ and____ show_ improvements on different medical__._Strength__ \n--The idea using__actual images_ sali__ generation_ interesting.\n\n_The improvement____aks is significant. \n\nWeak____The___ and_____ experiments are needed_ such as__f___the_ method_ interesting_ but_ novelty_ limited',
                 'output': 'This paper proposes an attention generation method for ROI detection by adversarial counterfactual without attention label. The attention map can be used to highlight useful information for disease classification and detection. The experiments show its improvements on different medical imaging tasks.  \nStrengths: \n--The idea using counterfactual images for saliency map generation is interesting.\n\n--The improvement for medical imaging taks is significant. \n\nWeaknesses:\n\n--The novelty is simple and limited. \n\n--More experiments are needed, such as existing counterfactual generation.\nthe proposed method is interesting, but the novelty is limited',
                 'count': 85},
                {'label': label, 'input': masked_seq,
                 'output': '', 'count': target_word}
            ]

            prompt = instruction
            template_demo = "{label}.\nInput: {input}\nFill-in-Blanks and your answer MUST be exactly {count} words: {output}\n"
            template_request = "{label}.\nInput: {input}\nFill-in-Blanks and your answer MUST be exactly {count} words:"
            prompt += "\n\n".join(template_demo.format(**a)
                                  for i, a in enumerate(demos[0:1]))
            prompt += "\n\n" + template_request.format(**demos[-1])

        elif "yelp_blank_fill_3_shot_word" in variation_type:
            lens_prompt = "and your answer MUST be exactly"
            lens_control = lens_prompt + f" {target_word} words"
            selected_style = ALL_styles[random.randrange(len(ALL_styles))]
            instruction = f"Based on the Business Category and Review Stars, you are required to fill in the blanks in the Input sentences {selected_style}. If there are no blanks, you are required to output the original Input sentences.\n"
            demo_1 = f"Business Category: Restaurants\tReview Stars: 2.0\nInput: _ that great , terrible _ rolls and fish _ smelling _ _.\nFill-in-Blanks {lens_prompt} 10 words: Not that great, terrible egg rolls and fishy smelling shrimp.\n"
            demo_2 = f"Business Category: Beauty & Spas\tReview Stars: 5.0\nInput: Very clean! Staff are super friendly!!\nFill-in-Blanks {lens_prompt} 6 words: Very clean! Staff are super friendly!!\n"
            demo_3 = f"Business Category: Shopping\tReview Stars: 3.0\nInput: I _ in _ and stopped in for a _. I was _ surprised. Good _, nice price.\nFill-in-Blanks {lens_prompt} 19 words: I was in a rush and stopped in for a mani-pedi. I was pleasantly surprised. Good service, nice price.\n"
            prompt = instruction + demo_1 + demo_2 + demo_3 + \
                f"{label} \nInput: {masked_seq} \nFill-in-Blanks {lens_control}:"

        return prompt, target_word

    def _text_variation_parallel(self, sequences, labels, variation_degree, variation_type, lookahead=1):
        num_seq = len(sequences)
        all_data = [[] for i in range(lookahead)]
        all_labels = []

        all_target_words = []
        all_gen_words = []

        message_constructor = MessageConstructor(
            PROMPTS_templates[self.var_template]['sys_prompt'], PROMPTS_templates[self.var_template]['task_desc'])

        self.mlm_probability = variation_degree
        all_prompts = []
        all_masked_prompts = []

        for idx in tqdm(range(num_seq)):
            for _ in range(lookahead):
                prompt, target_word = self._rephrase(
                    labels[idx], sequences[idx], variation_type)

                if self.dynamic_len and target_word is not None:
                    variation_max_length = int(
                        target_word * self.max_token_word_scale)
                else:
                    variation_max_length = self.length
                variation_max_length = min(
                    variation_max_length, self.max_token_limit)  # hard max_token limit

                all_prompts.append(
                    (message_constructor.get_message(prompt), variation_max_length))
                
                all_masked_prompts.append(
                    message_constructor.get_message(prompt))
                all_target_words.append(target_word)

        
        if self.dry_run:
            generations = [prompt[0][1]["content"] for prompt in all_prompts]
        else:
            generations = openai_completions(all_prompts, model_name=self.model_type, engine_name=self.engine, openai_api_keys=[self.openai_api_key], openai_api_base=self.openai_api_base,
                                             num_procs=self.num_procs,  top_p=0.95, temperature=self.temperature, sleep_time=self.sleep_time)['completions']

        gen_idx = -1
        for idx in tqdm(range(num_seq)):
            for j in range(lookahead):
                gen_idx += 1
                try:
                    seq = generations[gen_idx]

                    seq = " ".join(seq.split())
                    if seq:
                        all_data[j].append(seq)
                    else:
                        all_data[j].append(sequences[idx])
                except Exception as e:
                    logging.info(e)
                    if j > 1:  # already have variaitions
                        all_data[j].append(all_data[j-1][-1])
                    else:
                        all_data[j].append(sequences[idx])
                all_gen_words.append(len(all_data[j][-1].split(" ")))

            all_labels.append(labels[idx])

        all_lens = [len(one_data) for one_data in all_data]

        return all_data, all_labels,   all_target_words, all_gen_words, all_masked_prompts
    

class AzureDeepSeek(API):
    def __init__(self,
                 model_type, variation_type, use_subcategory,
                 output_dir, seed, num_procs,
                 length, temperature, top_p, do_sample, no_cuda,
                 dynamic_len, min_target_word, word_var_scale, max_token_word_scale,
                 sleep_time,
                 max_token_limit, dry_run, r_data,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.engine = ''
        self.model_type = model_type
        print(f"model_type: {model_type}")
        print(f"MODEL_CONFIG.keys(): {DeepSeek_CONFIG.keys()}")
        if model_type not in DeepSeek_CONFIG.keys():
            raise ValueError(f'Unknown model type name {model_type}')
        else:
            self.openai_api_key = DeepSeek_CONFIG[model_type]['openai_api_key']
            self.openai_api_base = DeepSeek_CONFIG[model_type]['openai_api_base']
            self.engine = DeepSeek_CONFIG[model_type]['engine']


        self.num_procs = num_procs
        self.use_subcategory = use_subcategory
        available_datasets = ['yelp', 'openreview', 'pubmed', 'sd']

        self.variation_type = variation_type
        self.init_template = None
        self.var_template = None

        for current_dataset in available_datasets:
            if current_dataset in self.variation_type:
                self.init_template = f'init_{current_dataset}'
                self.var_template = f'variant_{current_dataset}'
                break

        if use_subcategory:
            self.subcategory_dict = {}
            for current_dataset in available_datasets:
                if current_dataset in self.variation_type:
                    self.subcategory_dict[current_dataset] = get_subcategories(
                        current_dataset)

        self.output_dir = output_dir
        self.length = length
        self.temperature = temperature
        self.p = top_p
        self.do_sample = do_sample
        self.no_cuda = no_cuda
        self.dynamic_len = dynamic_len
        self.min_target_word = min_target_word
        self.word_var_scale = word_var_scale
        self.max_token_word_scale = max_token_word_scale
        self.max_token_limit = max_token_limit
        self.dry_run = dry_run

        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        set_seed(seed=seed, n_gpu=self.n_gpu)

        self.sleep_time = sleep_time
        #if 'blank_fill' in self.variation_type:
        if 'sd' in self.variation_type or 'pubmed' in self.variation_type:
            from transformers import BertTokenizer
            self.mask_tokenizer = BertTokenizer.from_pretrained(
                "bert-base-cased",  mask_token="_")
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        self.r_data = r_data

    @staticmethod
    def command_line_parser():
        parser = super(AzureDeepSeek, AzureDeepSeek).command_line_parser()
        parser.add_argument(
            '--model_type',
            type=str,
            default='DeepSeek-R1',
            help='Which model to use')

        parser.add_argument("--use_subcategory",
                            action="store_true", help="use subcategory")

        parser.add_argument(
            '--variation_type',
            type=str,
            default='pubmed_rephrase_tone',
            choices=["yelp_blank_fill_3_shot_word",
                     "openreview_blank_fill_1_shot_word",
                     "pubmed_blank_fill_0_shot_word",
                     "sd_blank_fill_0_shot_word",
                     "sd_rephrase_tone",
                     "pubmed_rephrase_tone",
                     ],
            help='Which image feature extractor to use')

        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
        )
        parser.add_argument("--length", type=int, default=128)
        parser.add_argument("--sleep_time", type=int, default=1)
        parser.add_argument("--max_token_limit", type=int, default=1200)
        parser.add_argument("--min_target_word", type=int, default=25)
        parser.add_argument("--word_var_scale", type=int, default=10)
        parser.add_argument("--max_token_word_scale", type=float, default=1.2)
        parser.add_argument("--temperature", type=float, default=1.0)

        parser.add_argument("--top_p", type=float, default=1)

        parser.add_argument("--num_procs", type=int, default=4)
        parser.add_argument("--do_sample", action="store_true",
                            help="sampling when generation")
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")

        parser.add_argument("--no_cuda", action="store_true",
                            help="Avoid using CUDA when available")
        parser.add_argument(
            "--dynamic_len", action="store_true", help="change max token")
        parser.add_argument("--dry_run", action="store_true", help="dry run")
        parser.add_argument("--r_data", type=int, default=1)
        return parser

    def enforce_rate_limit(self, tokens):
        global tokens_used, requests_made, start_time
        elapsed_time = time.time() - start_time

        if elapsed_time > 60:
            tokens_used = 0
            requests_made = 0
            start_time = time.time()

        if tokens_used + tokens > MAX_TOKENS_PER_MINUTE or requests_made >= MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - elapsed_time  # Wait the remaining time in the minute
            time.sleep(sleep_time)
            tokens_used = 0
            requests_made = 0
            start_time = time.time()

        # Update counters
        tokens_used += tokens
        requests_made += 1
    
    def text_random_sampling(self, num_samples, prompt_counter=None, lens_dict=None):
        ratio_generation_training = num_samples / sum(prompt_counter.values())
        syn_samples = []
        additional_info = []
        sync_labels_counter = collections.Counter()

        message_constructor = MessageConstructor(
            PROMPTS_templates[self.init_template]['sys_prompt'], PROMPTS_templates[self.init_template]['task_desc'])
        all_prefix_prompts = []
        for prompt_text in tqdm(prompt_counter):
            prompt = prompt_text
            # generation is proportional to the label distributions
            num_seq_to_generate = round(
                prompt_counter[prompt_text] * ratio_generation_training)
            if num_seq_to_generate > 0:
                max_lens_ls = [
                    self.length]*num_seq_to_generate if lens_dict is None else lens_dict[prompt_text]
                sequences, prefix_prompts = self._generate_text(prompt, num_seq_to_generate, message_constructor,
                                                                max_length=max_lens_ls)

                syn_samples += sequences
                all_prefix_prompts += prefix_prompts

                additional_info += [prompt_text] * len(sequences)
                sync_labels_counter[prompt_text] = len(sequences)

        return syn_samples, additional_info, sync_labels_counter, all_prefix_prompts

    def _generate_text(self, prompt, seq_num, message_constructor, max_length):
        all_data = []
        all_prompts = []
        all_prefix_prompts = []

        for i in tqdm(range(seq_num)):
            if self.use_subcategory:
                if "yelp" in self.variation_type:
                    category_label = prompt.split(
                        "\t")[0].replace('Business Category: ', '')
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['yelp'][category_label]))
                    keyword = self.subcategory_dict['yelp'][category_label][rand_keyword_idx]
                    prefix = f'{prompt} with keyword {keyword}'
                elif "openreview" in self.variation_type:
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['openreview']))
                    keyword = self.subcategory_dict['openreview'][rand_keyword_idx]
                    
                    prefix = f"Suppose that you are a {keyword}, and your answer should contain 1000 words.\n" + prompt
                elif "pubmed" in self.variation_type:
                    prefix = PUBMED_INIT_templates[random.randrange(
                        len(PUBMED_INIT_templates))]  # random select one template
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['pubmed']))
                    keyword = self.subcategory_dict['pubmed'][rand_keyword_idx]
                    prefix = f"Suppose that you are a {keyword}. " + prefix
                elif "sd" in self.variation_type:
                    prefix = SD_INIT_templates[random.randrange(
                        len(SD_INIT_templates))]  # random select one template
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['sd']))
                    keyword = self.subcategory_dict['sd'][rand_keyword_idx]
                    prefix = f"Suppose that you are a {keyword}. " + prefix
            else:
                prefix = prompt
            
            all_prompts.append(
                (message_constructor.get_message(prefix), max_length[i]))
            all_prefix_prompts.append(message_constructor.get_message(prefix))

        if self.dry_run:
            generations = [prompt[0][1]["content"] for prompt in all_prompts]
        else:
            #self.enforce_rate_limit(10)
            generations = deepseek_completions(all_prompts, model_name=self.model_type, engine_name=self.engine, openai_api_keys=[self.openai_api_key], openai_api_base=self.openai_api_base,
                                             num_procs=self.num_procs,  top_p=self.p, temperature=self.temperature)['completions']

        logging.info(f"len generations:  {len(generations)}")
        for idx in range(len(generations)):
            seq = generations[idx]
            seq = " ".join(seq.split())

            if seq:
                all_data.append(seq)

        logging.info(f"len all_data:  {len(all_data)}")
        logging.info(f"seq_num:  {seq_num}")
        return all_data, all_prefix_prompts

    def text_variation(self, sequences, additional_info,
                       num_variations_per_sequence, variation_degree):

        if not (0 <= variation_degree <= 1):
            raise ValueError('variation_degree should be between 0 and 1')
        variations, var_labels,  all_target_words, all_gen_words, all_masked_prompts = self._text_variation_parallel(
            sequences=sequences,
            labels=list(additional_info),
            variation_degree=variation_degree,
            variation_type=self.variation_type,
            lookahead=num_variations_per_sequence)

        return np.stack(variations, axis=1), var_labels,  all_target_words, all_gen_words, all_masked_prompts

    def _create_masked_seq(self, input_sequence, mlm_probability=0.5):
        try:
            input_ids = self.encoding.encode(input_sequence)

            masked_indices = np.random.binomial(
                1, mlm_probability, size=len(input_ids)).astype(bool).tolist()

            new_input_ids = copy.deepcopy(input_ids)
            for (i, v) in enumerate(masked_indices):
                new_input_ids[i] = self.encoding.encode(
                    "_")[0] if v else input_ids[i]

            return self.encoding.decode(new_input_ids)
        except:
            return input_sequence

    def _rephrase(self, label, sequence, variation_type):

        target_word = len(sequence.split(
            " ")) + int(np.random.normal(0, self.mlm_probability*self.word_var_scale, 1)[0])
        target_word = max(target_word, self.min_target_word)

        masked_seq = self._create_masked_seq(sequence, self.mlm_probability)
        if "pubmed_blank_fill_0_shot_word" in variation_type:
            selected_style = ALL_PUBMED_styles[random.randrange(
                len(ALL_PUBMED_styles))]
            instruction = f"You are required to fill in the blanks with more details for the input medical abstract {selected_style}. If there is no blanks, please output the original medical abstract.\n"
            prompt = instruction + \
                f"Please fill in the blanks in the following sentences to write an abstract of a medical research paper: \"{masked_seq}\" \n"

        elif "pubmed_rephrase_tone" in variation_type:
            selected_style = ALL_PUBMED_styles[random.randrange(
                len(ALL_PUBMED_styles))]
            prompt = f"Use different words to rephrase the following sentences {selected_style}:\n{sequence} \n"

        elif "sd_rephrase_tone" in variation_type:
            selected_style = ALL_SD_styles[random.randrange(
                len(ALL_SD_styles))]
            prompt = f"Rrephrase the following sentences: \"{sequence}\"\n"
                
        elif "sd_blank_fill_0_shot_word" in variation_type:
            selected_style = ALL_SD_styles[random.randrange(
                len(ALL_SD_styles))]
            prompt = f"You are required to fill in the blanks with more details {selected_style} for the input passage. If there is no blanks, please output the original passage: \"{masked_seq}\"\n"
                
        
        elif "openreview_blank_fill_1_shot_word" in variation_type:
            selected_style = ALL_OPENREVIEW_styles[random.randrange(
                len(ALL_OPENREVIEW_styles))]
            instruction = f"Based on the area and final decision of a research paper, you are required to fill in the blanks for the input sentences **{selected_style}**. If there is no blanks, please output the original input sentences.\n"

            demos = [
                {'label': 'Area: Applications (eg, speech processing, computer vision, NLP)\tRecommendation: 3: reject, not good enough',
                 'input_word': 'This paper proposes an attention generation _ for ROI detection by adversarial counterfactual without attention label. The attention map can _ used _ highlight useful information for disease classification and detection. _ experiments show its _ on different medical imaging tasks. Strengths: --The idea using _ images for _ map _ is interesting. --The _ _ medical imaging taks is significant. Weaknesses: --The novelty is _ and _ --More _ are needed, such as _ counterfactual generation. _ proposed method is interesting, but the novelty is limited',
                 'input': '__ proposes an__ method_ ROI detection__arial_f_ without attention_. The_ map can_ used____ for__ and____ show_ improvements on different medical__._Strength__ \n--The idea using__actual images_ sali__ generation_ interesting.\n\n_The improvement____aks is significant. \n\nWeak____The___ and_____ experiments are needed_ such as__f___the_ method_ interesting_ but_ novelty_ limited',
                 'output': 'This paper proposes an attention generation method for ROI detection by adversarial counterfactual without attention label. The attention map can be used to highlight useful information for disease classification and detection. The experiments show its improvements on different medical imaging tasks.  \nStrengths: \n--The idea using counterfactual images for saliency map generation is interesting.\n\n--The improvement for medical imaging taks is significant. \n\nWeaknesses:\n\n--The novelty is simple and limited. \n\n--More experiments are needed, such as existing counterfactual generation.\nthe proposed method is interesting, but the novelty is limited',
                 'count': 85},
                {'label': label, 'input': masked_seq,
                 'output': '', 'count': target_word}
            ]

            prompt = instruction
            template_demo = "{label}.\nInput: {input}\nFill-in-Blanks and your answer MUST be exactly {count} words: {output}\n"
            template_request = "{label}.\nInput: {input}\nFill-in-Blanks and your answer MUST be exactly {count} words:"
            prompt += "\n\n".join(template_demo.format(**a)
                                  for i, a in enumerate(demos[0:1]))
            prompt += "\n\n" + template_request.format(**demos[-1])

        elif "yelp_blank_fill_3_shot_word" in variation_type:
            lens_prompt = "and your answer MUST be exactly"
            lens_control = lens_prompt + f" {target_word} words"
            selected_style = ALL_styles[random.randrange(len(ALL_styles))]
            instruction = f"Based on the Business Category and Review Stars, you are required to fill in the blanks in the Input sentences {selected_style}. If there are no blanks, you are required to output the original Input sentences.\n"
            demo_1 = f"Business Category: Restaurants\tReview Stars: 2.0\nInput: _ that great , terrible _ rolls and fish _ smelling _ _.\nFill-in-Blanks {lens_prompt} 10 words: Not that great, terrible egg rolls and fishy smelling shrimp.\n"
            demo_2 = f"Business Category: Beauty & Spas\tReview Stars: 5.0\nInput: Very clean! Staff are super friendly!!\nFill-in-Blanks {lens_prompt} 6 words: Very clean! Staff are super friendly!!\n"
            demo_3 = f"Business Category: Shopping\tReview Stars: 3.0\nInput: I _ in _ and stopped in for a _. I was _ surprised. Good _, nice price.\nFill-in-Blanks {lens_prompt} 19 words: I was in a rush and stopped in for a mani-pedi. I was pleasantly surprised. Good service, nice price.\n"
            prompt = instruction + demo_1 + demo_2 + demo_3 + \
                f"{label} \nInput: {masked_seq} \nFill-in-Blanks {lens_control}:"

        return prompt, target_word

    def _text_variation_parallel(self, sequences, labels, variation_degree, variation_type, lookahead=1):
        num_seq = len(sequences)
        all_data = [[] for i in range(lookahead)]
        all_labels = []

        all_target_words = []
        all_gen_words = []

        message_constructor = DSMessageConstructor(
            PROMPTS_templates[self.var_template]['sys_prompt'], PROMPTS_templates[self.var_template]['task_desc'])

        self.mlm_probability = variation_degree
        all_prompts = []
        all_masked_prompts = []

        for idx in tqdm(range(num_seq)):
            for _ in range(lookahead):
                prompt, target_word = self._rephrase(
                    labels[idx], sequences[idx], variation_type)

                if self.dynamic_len and target_word is not None:
                    variation_max_length = int(
                        target_word * self.max_token_word_scale)
                else:
                    variation_max_length = self.length
                variation_max_length = min(
                    variation_max_length, self.max_token_limit)  # hard max_token limit

                all_prompts.append(
                    (message_constructor.get_message(prompt), variation_max_length))
                
                all_masked_prompts.append(
                    message_constructor.get_message(prompt))
                all_target_words.append(target_word)

        
        if self.dry_run:
            generations = [prompt[0][1]["content"] for prompt in all_prompts]
        else:
            generations = deepseek_completions(all_prompts, model_name=self.model_type, engine_name=self.engine, openai_api_keys=[self.openai_api_key], openai_api_base=self.openai_api_base, num_procs=self.num_procs, top_p=0.95, temperature=self.temperature, sleep_time=self.sleep_time)['completions']

        gen_idx = -1
        for idx in tqdm(range(num_seq)):
            for j in range(lookahead):
                gen_idx += 1
                try:
                    seq = generations[gen_idx]
                    seq = " ".join(seq.split())
                    if seq:
                        all_data[j].append(seq)
                    else:
                        all_data[j].append(sequences[idx])
                except Exception as e:
                    logging.info(e)
                    if j > 1:  # already have variaitions
                        all_data[j].append(all_data[j-1][-1])
                    else:
                        all_data[j].append(sequences[idx])
                all_gen_words.append(len(all_data[j][-1].split(" ")))

            all_labels.append(labels[idx])

        all_lens = [len(one_data) for one_data in all_data]

        return all_data, all_labels,   all_target_words, all_gen_words, all_masked_prompts
