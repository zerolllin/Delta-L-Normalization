pushd $(dirname "$0") > /dev/null
    set -x

    export N_GPUS=4
    export BASE_MODEL=Qwen/Qwen2.5-3B
    export DATA_DIR=~/data/open_reasoner_zero_nochat
    export ROLLOUT_TP_SIZE=1
    export EXPERIMENT_NAME=math-3b-dapo-norm
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export PYTHONUNBUFFERED=1

    ray stop
    env WANDB_API_KEY=${WANDB_API_KEY} VLLM_ATTENTION_BACKEND=XFORMERS RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 ray start --head --dashboard-host=0.0.0.0


    python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${BASE_MODEL}');"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=${DATA_DIR}/train.parquet \
        data.val_files="[\"${DATA_DIR}/test1.parquet\",\"${DATA_DIR}/test2.parquet\",\"${DATA_DIR}/test_amc.parquet\",\"${DATA_DIR}/test_minerva.parquet\"]" \
        data.train_batch_size=128 \
        data.filter_overlong_prompts=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
        data.apply_chat_template=False \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        critic.optim.lr=1e-6 \
        algorithm.kl_ctrl.kl_coef=0.0 \
        trainer.logger="['console','wandb']" \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.val_before_train=True \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=2000 \
        trainer.test_freq=50 \
        trainer.project_name=LatestVerlExperiment \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.total_epochs=15 \
        actor_rollout_ref.rollout.n=10 \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.3

popd > /dev/null