import torch
import numpy as np
from tqdm import tqdm
import logging
from .api import API
import transformers
import random
from .utils import *
import re
import collections
from dpsda.refinement import *
import copy


class HFAPI(API):
    def __init__(self,
                 model_type, variation_type, use_subcategory,
                 output_dir, seed, mlm_probability,
                 length, temperature, top_k, top_p, repetition_penalty, do_sample, fp16, no_cuda,
                 random_sampling_batch_size, num_beams, dry_run,
                 variation_batch_size,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_type = model_type
        self.variation_type = variation_type
        self.output_dir = output_dir
        self.length = length
        self.temperature = temperature
        self.k = top_k
        self.p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.fp16 = fp16
        self.no_cuda = no_cuda
        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        set_seed(seed=seed, n_gpu=self.n_gpu)
        self.dry_run = dry_run

        self.use_subcategory = use_subcategory
        if use_subcategory:
            self.subcategory_dict = {}
            #self.subcategory_dict['yelp'] = get_subcategories("yelp")
            self.subcategory_dict['sd'] = get_subcategories("sd")
            self.subcategory_dict['pubmed'] = get_subcategories("pubmed")
            #self.subcategory_dict['openreview'] = get_subcategories("openreview")

        model_name_or_path = self.model_type

        if self.model_type == "microsoft/phi-4":
            model_name = "microsoft/phi-4"
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,local_files_only=True)
            self.tokenizer.padding_side = "left"
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",  
                torch_dtype="auto",  
                local_files_only=True
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name_or_path, device_map="auto")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map="auto", pad_token_id=pad_token_id)
            if self.fp16:
                self.model.half()

        self.random_sampling_batch_size = random_sampling_batch_size
        self.variation_batch_size = variation_batch_size

    @staticmethod
    def command_line_parser():
        parser = super(HFAPI, HFAPI).command_line_parser()
        parser.add_argument(
            '--model_type',
            type=str,
            default='microsoft/phi-4',
            help='Which image feature extractor to use')
        parser.add_argument("--use_subcategory",
                            action="store_true", help="use subcategory")
        parser.add_argument(
            '--variation_type',
            type=str,
            default='sd_rephrase_tone',
            help='Which image feature extractor to use')
        parser.add_argument("--mlm_probability", type=float, default=0.5)

        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
        )
        parser.add_argument("--length", type=int, default=128)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--repetition_penalty", type=float, default=1.0,
                            help="primarily useful for CTRL model; in that case, use 1.2")
        parser.add_argument("--top_k", type=int, default=50)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--num_beams", type=int, default=5)
        parser.add_argument("--do_sample", action="store_true",
                            help="sampling when generation")
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument("--dry_run", action="store_true", help="dry run")
        parser.add_argument(
            '--random_sampling_batch_size',
            type=int,
            default=64,
            help='The batch size for random sampling API')
        parser.add_argument(
            '--variation_batch_size',
            type=int,
            default=256,
            help='The batch size for variation API')

        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser.add_argument("--no_cuda", action="store_true",
                            help="Avoid using CUDA when available")

        return parser

    def text_random_sampling(self, num_samples, prompt_counter=None, lens_dict=None):
        ratio_generation_training = num_samples / sum(prompt_counter.values())
        all_sequences = []
        ppls_cur = []
        additional_info = []
        sync_labels_counter = collections.Counter()

        self.model.eval()

        simulate_num = 0
        for prompt in tqdm(prompt_counter):
            simulate_num_seq_to_generate = round(
                prompt_counter[prompt] * ratio_generation_training)
            simulate_num += simulate_num_seq_to_generate

        logging.info(
            f"should -- simulated generated sequences: %d", simulate_num)
        all_prefix_prompts = []
        for prompt in tqdm(prompt_counter):
            num_seq_to_generate = round(
                prompt_counter[prompt] * ratio_generation_training)
            if self.use_subcategory:                    
                if "yelp" in self.variation_type:
                    category_label = prompt.split(
                        "\t")[0].replace('Business Category: ', '')
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['yelp'][category_label]))
                    keyword = self.subcategory_dict['yelp'][category_label][rand_keyword_idx]
                    full_prompt_text = f'{prompt} with keyword {keyword}'
                    
                elif "openreview" in self.variation_type:
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['openreview']))
                    keyword = self.subcategory_dict['openreview'][rand_keyword_idx]
                    full_prompt_text = f"Suppose that you are a {keyword}. Write a paper review based on " + prompt

                elif "pubmed" in self.variation_type and "phi4" not in self.variation_type:
                    full_prompt_text = "Using a variety of sentence structures, write an abstract for a medical research paper: "
                    
                elif "sd" in self.variation_type and "phi4" not in self.variation_type:
                    full_prompt_text = "Using a variety of sentence structures, write a passage in the tone of a person who is {poor or broke or homeless or unemployment or not being able to afford basic necessities}: "
                elif "phi4" in self.variation_type:
                    if self.variation_type == "sd_phi4_rephrase_tone":
                        selected_style = ALL_SD_styles[random.randrange(
                            len(ALL_SD_styles))]
                        system_prompt = f"Using a variety of sentence structures, write a passage {selected_style} of a person who is poor or broke or homeless or unemployment or not being able to afford basic necessities: "
                    else:
                        selected_style = ALL_PUBMED_styles[random.randrange(
                            len(ALL_PUBMED_styles))]
                        system_prompt = f"Using a variety of sentence structures, write an abstract for a medical research paper in {selected_style} :"            
                    full_prompt_text = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": ""}
                    ]
            else:
                full_prompt_text = prompt
            
            if isinstance(full_prompt_text, list):

                formatted_prompt = self.tokenizer.apply_chat_template(
                    full_prompt_text, return_tensors="pt", padding=True, truncation=False
                ).to(self.device)
            else:
                formatted_prompt = self.tokenizer(
                full_prompt_text, return_tensors="pt", padding=True, truncation=True
            )['input_ids'].to(self.device)
                
            prompt_input_ids = formatted_prompt    
            before_gen_length = len(full_prompt_text)

            if num_seq_to_generate > 0:
                # condition on the prompt
                sequences = self._generate_text(prompt_input_ids, num_seq_to_generate,
                                                max_length=self.length, batch_size=self.random_sampling_batch_size,
                                                before_gen_length=before_gen_length)
                all_sequences += sequences
            all_prefix_prompts += [copy.deepcopy(full_prompt_text)] * num_seq_to_generate

            
            additional_info += [prompt] * num_seq_to_generate
            sync_labels_counter[prompt] = num_seq_to_generate

        logging.info(f"Total generated sequences: %d", len(all_sequences))
        torch.cuda.empty_cache()
        return all_sequences,  additional_info, sync_labels_counter, all_prefix_prompts
    
    def _generate_text(self, prompt, seq_num, max_length, batch_size, before_gen_length):

        all_data = []

        if "phi4" in self.variation_type:
            for _ in range(seq_num):
                raw_generated = self.generate_long_text(prompt, step_size=200)

                cleaned_text = raw_generated.strip()

                cleaned_text = re.sub(r'^[^{\[]*?[\[{]?"?text"?\s*:\s*"?', '', cleaned_text)
                cleaned_text = cleaned_text.strip('"{}[]')

                segments = re.split(r'(?i)(system|user|assistant)', cleaned_text)
                if 'assistant' in segments:
                    last_idx = len(segments) - 1 - segments[::-1].index('assistant')
                    cleaned_text = ''.join(segments[last_idx+1:]).strip()

                cleaned_text = re.sub(r'(?:\*\*Solution \d+:\*\*.*?)?(?:\d\)\s.*?)+(?=[A-Z])', '', cleaned_text, flags=re.DOTALL)

                cleaned_text = cleaned_text.strip('"').strip()
                if cleaned_text:
                    all_data.append(cleaned_text)
            return all_data

        if seq_num < batch_size:
            batch_size = seq_num + 1  # TODO: improve

        num_return_sequences = 2 if batch_size > 1 else 1
        for i in tqdm(range(seq_num // batch_size + 1)):
            if self.dry_run:
                generated_sequences = ["s" * max_length] * batch_size
            else:
                input_ids = torch.as_tensor(prompt, device=self.device).repeat(batch_size, 1)

                with torch.no_grad():
                    output_sequences = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_length,
                        temperature=self.temperature,
                        top_k=self.k,
                        top_p=self.p,
                        early_stopping=True,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=self.do_sample,
                        num_return_sequences=num_return_sequences,
                        no_repeat_ngram_size=2,
                    )
                    generated_sequences = self.tokenizer.batch_decode(
                        output_sequences[:, input_ids.shape[1]:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
            
            for g in generated_sequences:
                seq = " ".join(g.split())
                seq = re.sub(r"([.?!,]){2,}", r"\1", seq)
                if seq:
                    all_data.append(seq)

        if len(all_data) > seq_num:
            all_data = random.sample(all_data, seq_num)

        return all_data

    def text_variation(self, sequences, additional_info,
                       num_variations_per_sequence, variation_degree):
        self.model.eval()
        variations = []
        for i in tqdm(range(num_variations_per_sequence)):
            sub_variations, var_labels = self._text_variation(
                sequences=sequences,
                labels=list(additional_info),
                variation_degree=variation_degree,
                variation_type=self.variation_type,
                batch_size=self.variation_batch_size)  
            variations.append(sub_variations)

            if i % 5 == 0:  
                torch.cuda.empty_cache()
        
        return np.stack(variations, axis=1), var_labels, [], [], []

    def _rephrase(self, label, sequence, variation_type):

        if variation_type == "yelp_rephrase_tone":
            selected_style = ALL_styles[random.randrange(len(ALL_styles))]
            prompt = "Based on {}, please rephrase the following sentences {}:\n{} \n".format(
                label, selected_style, sequence)
        elif variation_type == "openreview_rephrase_tone":
            selected_style = ALL_OPENREVIEW_styles[random.randrange(
                len(ALL_OPENREVIEW_styles))]
            prompt = "Based on {}, please rephrase the following sentences {} as a paper review:\n{} \n".format(
                label, selected_style, sequence)
        elif variation_type == "pubmed_rephrase_tone":
            selected_style = ALL_PUBMED_styles[random.randrange(
                len(ALL_PUBMED_styles))]
            prompt = "Please rephrase the following sentences as an abstract for medical research paper in clear and natural English while preserving meaning:\n{} \n".format(sequence)
        elif variation_type == "sd_rephrase_tone":
            selected_style = ALL_SD_styles[random.randrange(
                len(ALL_SD_styles))]
            prompt = "Rephrase the following passage {}:\n{} \n".format(selected_style, sequence)
        elif "phi4" in self.variation_type:
            # Parse fields
            _parts = str(label).strip().split("\t")
            _sent = _parts[1].strip().lower() if len(_parts) >= 2 else None
            try:
                _tgt_words = int(_parts[2]) if len(_parts) >= 3 else None
            except:
                _tgt_words = None

            system_prompt = (
                "You are rewriting an abstracted self-disclosure into a realistic first-person passage. "
                "Preserve the core facts implied by the abstracted text, avoid adding new events or details you cannot infer, "
                "and keep the tone coherent and natural. "
            )
            if _sent in {"positive", "negative"}:
                system_prompt += f"Ensure the rewritten text clearly expresses a {_sent} sentiment. "

            if _tgt_words and _tgt_words > 0:
                lo = max(1, int(round(0.85 * _tgt_words)))
                hi = int(round(1.15 * _tgt_words))
                system_prompt += (
                    f"Write between {lo} and {hi} words (aim ~{_tgt_words}); do not be concise. "
                    "Write 4–6 sentences covering: (a) brief context, (b) what happened, "
                    f"(c) how I felt (matching {_sent}), (d) what I did or plan to do. "
                    "Avoid names, identifiers, or specific places. "
                    "Output only the passage; no meta commentary."
                )
                if _tgt_words >= 240:
                    system_prompt += " If it reads more naturally, use two short paragraphs."
            else:
                system_prompt += "Write 4–6 sentences. Output only the passage; no meta commentary."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sequence}
            ]
            return messages


        '''
        elif "phi4" in self.variation_type:
            # Parse label fields: expected "sd\t<sentiment>\t<target_words>"
            # Backward-compatible: if fields are missing, we just skip length hint.
            _parts = str(label).strip().split("\t")
            _tag = _parts[0].lower() if len(_parts) >= 1 else ""
            _sent = _parts[1].strip().lower() if len(_parts) >= 2 else None
            try:
                _tgt_words = int(_parts[2]) if len(_parts) >= 3 else None
            except:
                _tgt_words = None

            # Build the system prompt with sentiment + OPTIONAL length hint
            if self.variation_type == "sd_phi4_rephrase_tone":
                system_prompt = (
                    "You are rewriting an abstracted self-disclosure into a realistic first-person passage. "
                    "Preserve the core facts implied by the abstracted text, avoid adding new events, and keep the tone coherent. "
                )
                if _sent in {"positive", "negative"}:
                    system_prompt += f"Ensure the rewritten text clearly expresses a {_sent} sentiment. "
                # Soft length hint if available
                if _tgt_words and _tgt_words > 0:
                    # ±15% window without being too rigid
                    lo = max(1, int(round(0.85 * _tgt_words)))
                    hi = int(round(1.15 * _tgt_words))
                    system_prompt += (
                        f"Aim for about {_tgt_words} words (acceptable range {lo}–{hi}). "
                        "Do not add meta commentary or instructions; write only the passage."
                    )
                else:
                    system_prompt += "Write only the passage; no meta commentary."
            else:
                # e.g., pubmed variants
                system_prompt = (
                    "Rewrite the abstracted research summary into a clear, neutral abstract. "
                    "Preserve meaning, avoid vivid metaphors or emotional language."
                )
                if _tgt_words and _tgt_words > 0:
                    lo = max(1, int(round(0.85 * _tgt_words)))
                    hi = int(round(1.15 * _tgt_words))``
                    system_prompt += f" Aim for ~{_tgt_words} words (range {lo}–{hi})."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sequence}
            ]
            return messages
            '''
        return prompt
    

    def generate_long_text(self, input_ids, step_size=200):
        """Generate long synthetic text in smaller steps to avoid CUDA OOM issues."""
        generated_text = []
    
        with torch.no_grad():
            for _ in range(self.length // step_size):
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=step_size,
                    do_sample=True,
                    temperature=self.temperature,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    num_return_sequences=1,
                    use_cache=True,
                )
                new_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                if new_text.strip():
                    generated_text.append(new_text)

        torch.cuda.empty_cache()
        return " ".join(generated_text)

    def extract_first_assistant_response(self,text):
        """
        Extracts only the first 'assistant' response and removes system/user content.
        Captures everything after the first 'assistant' and stops at the next 'system'.
        """
        match = re.search(r"assistant\s*([\s\S]+?)(?=\bsystem\b|\bsystem\B|$)", text, re.IGNORECASE)
        if match:
            response = match.group(1).strip()
            return response
        return text.strip()  



    def _text_variation(self, sequences, labels, variation_degree, variation_type, batch_size):
        """Handles batch-wise text variation generation with chunked sequence output."""
        device = self.device
        num_seq = len(sequences)
        all_data, all_labels = [], []

        for i in tqdm(range(num_seq // batch_size + 1)):
            start_idx = i * batch_size
            if start_idx >= num_seq:
                break
            end_idx = min(num_seq, (i + 1) * batch_size)

            batch_messages, batch_labels = [], []
            for idx in range(start_idx, end_idx):
                messages = self._rephrase(labels[idx], sequences[idx], variation_type)

                if messages is None or not isinstance(messages, list):
                    logging.error(f"Invalid messages structure at index {idx}: {messages}")
                    continue

                batch_messages.append(messages)
                batch_labels.append(labels[idx])

            with torch.no_grad():
                if not batch_messages:
                    logging.warning("Skipping batch because batch_messages is empty.")
                    continue

                for idx in range(len(batch_messages)):
                    target = str(batch_labels[idx]).strip().upper()
                    generated_text, _pred = self.generate_with_sentiment_guard(
                        messages=batch_messages[idx],
                        target_label=target,
                        max_attempts=4,      
                        threshold=None       
                    )
                    cleaned_response = generated_text if generated_text else None

                    lab = str(batch_labels[idx]).strip().split("\t")
                    if cleaned_response:
                        all_data.append(cleaned_response)
                    else:
                        all_data.append(batch_messages[idx][-1]['content'])  

                    all_labels.append(lab)  


            torch.cuda.empty_cache()

        logging.info(f"_text_variation output length: {len(all_data)}")
        return all_data, all_labels

    
    from collections import Counter
    import pandas as pd

    def _ensure_sentiment_clf(self):
        """Create the HF pipeline once and cache on self."""
        if not hasattr(self, "sentiment_clf") or self.sentiment_clf is None:
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.sentiment_clf = pipeline(
                "sentiment-analysis",
                model="siebert/sentiment-roberta-large-english",
                device=device
            )

    def _normalize_sentiment_label(self, raw_label):
        up = str(raw_label).strip().upper()
        if up.startswith("LABEL_"):
            idx = up.split("_")[-1]
            id2label = getattr(self.sentiment_clf.model.config, "id2label", {}) or {}
            mapped = None
            if idx in id2label:
                mapped = id2label[idx]
            else:
                try:
                    mapped = id2label[int(idx)]
                except Exception:
                    mapped = None
            if mapped:
                return str(mapped).strip().upper()
            fallback = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE", "LABEL_2": "NEUTRAL"}
            return fallback.get(up, up)
        return up

    def _classify_chunked_with_conf(self, text, max_chars=512, stride=400):
        from collections import Counter
        import pandas as pd

        if text is None or (isinstance(text, float) and pd.isna(text)): 
            return "ERROR", 0.0
        t = str(text).strip()
        if not t:
            return "ERROR", 0.0

        chunks, i = [], 0
        while i < len(t):
            chunk = t[i:i+max_chars]
            if not chunk: break
            chunks.append(chunk)
            if i + max_chars >= len(t): break
            i += stride

        try:
            results = self.sentiment_clf(chunks)  
        except Exception:
            return "ERROR", 0.0

        by_label = {}
        for r in results:
            try:
                lab = self._normalize_sentiment_label(r["label"])
                sc  = float(r.get("score", 0.0))
                if lab != "ERROR":
                    by_label.setdefault(lab, []).append(sc)
            except Exception:
                pass

        if not by_label:
            return "ERROR", 0.0

        def key_fn(item):
            lab, scores = item
            return (len(scores), sum(scores)/len(scores))

        best_lab, best_scores = max(by_label.items(), key=key_fn)
        avg_conf = sum(best_scores) / len(best_scores)
        return best_lab, avg_conf


    def generate_with_sentiment_guard(self, messages, target_label, max_attempts=4, threshold=None):

        self._ensure_sentiment_clf()
        target = str(target_label).strip().upper()

        best_match_text, best_match_conf = None, -1.0
        best_any_text, best_any_conf, best_any_pred = "", -1.0, "ERROR"

        for _ in range(max_attempts):
            input_ids = self.tokenizer.apply_chat_template(
                [messages], return_tensors="pt", padding=True, truncation=False
            ).to(self.device)[0].unsqueeze(0)

            gen = self.generate_long_text(input_ids, step_size=200)
            text = getattr(self, "extract_first_assistant_response", lambda x: x)(gen) or ""

            pred, conf = self._classify_chunked_with_conf(text)  

            if conf > best_any_conf:
                best_any_text, best_any_conf, best_any_pred = text, conf, pred

            if pred == target:
                if conf > best_match_conf:
                    best_match_text, best_match_conf = text, conf
                if threshold is not None and conf >= float(threshold):
                    return best_match_text, pred

        if best_match_text is not None:
            return best_match_text, target
        return best_any_text, best_any_pred








