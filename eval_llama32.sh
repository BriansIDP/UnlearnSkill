. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

expdir=exp/llama32_3B_instruct_origmodel
mkdir -p $expdir

python inference.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --testfile ./data/testnames.json \
    --outfile $expdir/testresults.json \