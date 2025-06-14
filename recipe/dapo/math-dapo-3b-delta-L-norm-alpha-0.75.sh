#!/usr/bin/env bash
set -xeuo pipefail

export N_GPUS=4


project_name='LatestVerlExperiment'
exp_name='math-dapo-3b-delta-L-norm-alpha-0.75'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.3

max_prompt_length=$((1024))
max_response_length=$((3072))
enable_overlong_buffer=False
overlong_buffer_len=$((2048))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=seq_final_reward
max_num_gen_batches=10
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=8
train_prompt_mini_bsz=32

# Ray
NNODES=1
# Paths
MODEL_PATH=Qwen/Qwen2.5-3B
export DATA_DIR=~/data/open_reasoner_zero_nochat


# Algorithm
temperature=1.0
val_top_p=1.0

# Performance Related Parameter
sp_size=1
offload=True
gen_tp=1


ray stop
env WANDB_API_KEY=${WANDB_API_KEY} VLLM_ATTENTION_BACKEND=XFORMERS RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 ray start --head --dashboard-host=0.0.0.0
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${MODEL_PATH}');"

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=~/verl_latest
RUNTIME_ENV=~/verl_latest/verl/trainer/runtime_env.yaml

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.apply_chat_template=False \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files="[\"${DATA_DIR}/test1.parquet\",\"${DATA_DIR}/test2.parquet\",\"${DATA_DIR}/test_amc.parquet\",\"${DATA_DIR}/test_minerva.parquet\"]" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=50 \
    trainer.save_freq=1000 \
    trainer.total_epochs=15 \
    algorithm.use_grpopp=True \
    algorithm.grpopp_config.alpha=0.75 \
    actor_rollout_ref.actor.use_grpopp=True \
    actor_rollout_ref.actor.grpopp_config.alpha=0.75