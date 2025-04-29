mlm_prob=0.6
var_type="pubmed_rephrase_tone" 
feat_ext="sentence-t5-base"
temperature=1.0
length=800
num_seed_samples=100
lookahead_degree=0
k=1 
L=$((k+1))
init_L=${L}
num_samples=$((L*num_seed_samples))
echo generating $num_samples samples
g_epochs=1
word_var_scale=60
max_token_word_scale=5
select_syn_mode=rank
model_type="DeepSeek-R1" 
noise=0 
feature_extractor_batch_size=1024
api="AzureDeepSeek"
save_syn_mode=all
r_data=1

export CUDA_VISIBLE_DEVICES=0

args="--dynamic_len --word_var_scale ${word_var_scale} --max_token_word_scale ${max_token_word_scale}"

result_folder="result/pubmed/${model_type}_${feat_ext}/${num_samples}_n${noise}_L${L}_initL${init_L}_var${lookahead_degree}_${var_type}_${select_syn_mode}_len${length}var${word_var_scale}wo2to${max_token_word_scale}_t${temperature}"


## load datacheckpoint 
data_checkpoint_args=""
for  (( iter=0; iter<=epochs; iter++ ))
do
train_file=${result_folder}/${iter}/samples.csv
if [ -e "$train_file" ]; then
    echo "$train_file does exist."
    # load from  data checkpoint
    data_checkpoint_args="--data_checkpoint_step ${iter} --data_checkpoint_path ${result_folder}/${iter}/samples.csv"
else
    echo "$train_file does not exist."
fi
done
echo load data from ${data_checkpoint_args} ${args}

python main.py ${args} ${data_checkpoint_args} \
--train_data_file "data/pubmed/pubmed_ds.csv" \
--dataset "pubmed" \
--api ${api} \
--noise ${noise} \
--model_type ${model_type} \
--do_sample  \
--length ${length} \
--temperature ${temperature} \
--select_syn_mode ${select_syn_mode} \
--num_samples_schedule ${num_samples} \
--combine_divide_L ${L} \
--init_combine_divide_L ${init_L} \
--variation_degree_schedule ${mlm_prob} \
--lookahead_degree ${lookahead_degree} \
--feature_extractor_batch_size ${feature_extractor_batch_size} \
--epochs ${g_epochs} \
--use_subcategory \
--feature_extractor ${feat_ext} \
--variation_type ${var_type} \
--result_folder ${result_folder} \
--train_data_embeddings_file "result/embeddings/${feat_ext}/pubmed_train_ds.embeddings.npz" \
--r_data ${r_data} \
--save_syn_mode ${save_syn_mode} 


f_batch_size=32
min_token_threshold=50
f_lr=2e-3
f_wd=0.01
item=${result_folder}
f_epochs=-1

for model in 'bert-small' 
do
    num_train_epochs=5
    for  (( iter=f_epochs; iter>=-1; iter-- ))
    do
        train_file="${item}/${iter}"
        if [ -d "$train_file" ]; then
            echo "$train_file does exist."

            train_output_dir=${train_file}/train_${model}/
            eval_output_dir=${train_file}/eval_${model}/


            if [ -e "$eval_output_dir/eval_results.json" ]; then
                echo "$eval_output_dir/eval_results.json does exist. -- SKIP running classification"
            else
                echo "Training directory: $train_output_dir"
                echo "Evaluation directory: $eval_output_dir"

                python utility_eval/run_clm.py \
                    --model_name_or_path prajjwal1/${model} \
                    --clean_dataset  \
                    --min_token_threshold ${min_token_threshold} \
                    --output_dir ${train_output_dir} \
                    --train_file ${train_file}/samples.csv \
                    --validation_file data/pubmed/dev.csv \
                    --per_device_train_batch_size ${f_batch_size} \
                    --per_device_eval_batch_size ${f_batch_size} \
                    --learning_rate ${f_lr} \
                    --do_train \
                    --do_eval \
                    --weight_decay ${f_wd} \
                    --num_train_epochs ${num_train_epochs} \
                    --save_total_limit 2 \
                    --overwrite_cache \
                    --gradient_accumulation_steps 2

                python utility_eval/run_clm.py \
                    --model_name_or_path ${train_output_dir} \
                    --output_dir ${eval_output_dir} \
                    --validation_file data/pubmed/test.csv \
                    --per_device_eval_batch_size ${f_batch_size} \
                    --do_eval \
                    --overwrite_cache
            fi
        fi
    done
done

python ner.py \
    --input_csv ${result_folder}/1_all/samples.csv \
    --output_csv ${result_folder}/1_all/sanitized_filter_sample.csv \
    --output_dir ${result_folder}/1_all \

python filtering_wp.py \
    --train_data_file "data/pubmed/pubmed_seeds_10k.csv" \
    --model_dir ${train_output_dir} \
    --input_csv ${result_folder}/1_all/sanitized_filter_sample.csv \
    --output_csv ${result_folder}/1_all/filtered_samples.csv \


d_batch_size=32
d_lr=2e-3 
d_wd=0.01
d_epochs=1

for model in 'bert-small' 
do
    num_train_epochs=5
    for (( iter=d_epochs; iter>=1; iter-- ))
    do
        train_file="${item}/${iter}_all"
        if [ -d "$train_file" ]; then
            echo "$train_file does exist."

            train_output_dir=${train_file}/fd_train_${model}/
            eval_output_dir=${train_file}/fd_eval_${model}/

            if [ -e "$eval_output_dir/eval_results.json" ]; then
                echo "$eval_output_dir/eval_results.json does exist. -- SKIP running classification"
            else
                echo "Training directory: $train_output_dir"
                echo "Evaluation directory: $eval_output_dir"

                python utility_eval/run_clm.py \
                    --model_name_or_path prajjwal1/${model} \
                    --clean_dataset \
                    --min_token_threshold ${min_token_threshold} \
                    --output_dir ${train_output_dir} \
                    --train_file ${train_file}/filtered_samples.csv \
                    --validation_file data/pubmed/dev.csv \
                    --per_device_train_batch_size ${d_batch_size} \
                    --per_device_eval_batch_size ${d_batch_size} \
                    --learning_rate ${d_lr} \
                    --do_train \
                    --do_eval \
                    --weight_decay ${d_wd} \
                    --num_train_epochs ${num_train_epochs} \
                    --save_total_limit 2 \
                    --overwrite_cache \
                    --gradient_accumulation_steps 2

                python utility_eval/run_clm.py \
                    --model_name_or_path ${train_output_dir} \
                    --output_dir ${eval_output_dir} \
                    --validation_file data/pubmed/test.csv \
                    --per_device_eval_batch_size ${d_batch_size} \
                    --do_eval \
                    --overwrite_cache
            fi
        fi
    done
done


m_epochs=1
synthetic_start_iter=1

python metric.py \
    --private_data_size 10000 \
    --synthetic_folder ${result_folder} \
    --run 5  \
    --min_token_threshold ${min_token_threshold} \
    --synthetic_iteration ${m_epochs} \
    --synthetic_start_iter ${synthetic_start_iter} \
    --original_file "data/pubmed/pubmed_seeds_10k.csv"  \
    --train_data_embeddings_file result/embeddings/${feat_ext}/pubmed_train_all.embeddings.npz \
    --model_name_or_path ${feat_ext} \
    --dataset pubmed \


python diversity.py \
    --synthetic_folder ${result_folder} \
    --synthetic_iteration ${m_epochs} \
    --synthetic_start_iter ${synthetic_start_iter} \
    --dataset pubmed \
    --min_token_threshold ${min_token_threshold} \



