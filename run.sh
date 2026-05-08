# 1. 环境变量设置
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/  # NPU 版 PyTorch 路径
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"  # PyTorch 安装路径
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"  # LibTorch 路径
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH  # 添加 NPU 库路径

# 2. 加载环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh

# export ASDOPS_LOG_TO_STDOUT=1
# export ASDOPS_LOG_LEVEL=DEBUG
# export ASDOPS_LOG_TO_FILE=1
# export MINDIE_LOG_LEVEL=DEBUG

# export ASCEND_RT_VISIBLE_DEVICES=9,10
# export ASDOPS_LOG_TO_STDOUT=1
# export ASDOPS_LOG_LEVEL=INFO
# export SPDLOG_LEVEL=debug
# export ASCEND_MODULE_LOG_LEVEL=ATB=0
# export ASDOPS_LOG_TO_FILE=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_RT_VISIBLE_DEVICES=12,13
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
#export HCCL_OP_EXPANSION_MODE=AIV
export INF_NAN_MODE_FORCE_DISABLE=1
export ASCEND_LAUNCH_BLOCKING=0
# export LCCL_DETERMINISTIC=1 
# export HCCL_DETERMINISTIC=true
# export ATB_MATMUL_SHUFFLE_K_ENABLE=0
# 3. 清理旧日志
\rm -rf core.*

# 4. 启动分布式服务
MODEL_PATH="/export/home/shanchenfeng/xllm_build/xllm_qwen_embed/qwen3_omni_moe_thinker"
MODEL_PATH="/export/home/shanchenfeng/xllm_build/xllm_qwen_embed/qwen3_omni_moe_instruct/dir/"
#MODEL_PATH="/export/home/wangyunlong.115/workspace/Qwen3-VL-8B"
#MODEL_PATH="/export/home/wangyunlong.115/workspace/oxygen-vlm-7b"


#MODEL_PATH="/export/home/wangyunlong.115/workspace/Qwen3-4B"
MODEL_PATH="/export/home/models/Qwen3-8B"
MODEL_PATH="/export/home/wangyunlong.115/workspace/oxygen-vlm-7b"
MODEL_PATH="/export/home/wangyunlong.115/workspace/Qwen2.5-VL-7B"
#MODEL_PATH="/export/home/wangyunlong.115/workspace/Qwen3-Embedding-0.6B"
#MODEL_PATH="/export/home/wangyunlong.115/workspace/Qwen3-VL-32B"
#MODEL_PATH="/export/home/wangziyue.28/test/models/Qwen3-VL-A3B"
#MODEL_PATH="/export/home/wangyunlong.115/workspace/GLM-46V-Moe"
#MODEL_PATH="/export/home/wangyunlong.115/workspace/GLM-4.6V-Flash"
#MODEL_PATH="/export/home/wangyunlong.115/workspace/GLM-46V"
#MODEL_PATH="/export/home/wangyunlong.115/workspace/Qwen3-VL-A3B"
MASTER_NODE_ADDR="127.0.0.1:9741"                  # Master 节点地址（需全局一致）
START_PORT=12346                                   # 服务起始端口
START_DEVICE=0                           # 起始 NPU 逻辑设备号
LOG_DIR="log"                                      # 日志目录
NNODES=1                                      # 节点数（当前脚本启动 2 个进程）
export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口
#msit llm dump --exec "./bin/mindieservice_daemon" --type model tensor -o /home/ea-9n-das-admin-2/msit_dump/ -child False -er 0,6
for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
    ./build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --devices="npu:$DEVICE" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.80 \
    --model_id=Qwen3-VL-32B-Instruct \
    --max_tokens_per_batch=81920 \
    --max_tokens_per_chunk_for_prefill=8192 \
    --max_seqs_per_batch=256 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --node_rank=$i  \
    --communication_backend="lccl" \
    --limit_image_per_prompt=100 \
    --enable_graph=false \
    --enable_encoder_cache=true \
    --enable_shm=true \
    --task="generate" \
    --backend vlm  > $LOG_FILE 2>&1 &
done
#     --mm_process_config='{"image":{"min_pixels":5024}}' \
# for (( i=0; i<$NNODES; i++ ))
# do
#   PORT=$((START_PORT + i))
#   DEVICE=$((START_DEVICE + i))
#   LOG_FILE="$LOG_DIR/node_$i.log"
#   msit llm dump --exec "/export/home/wangyunlong.115/workspace/xllm/build/xllm/core/server/xllm \
#     --model $MODEL_PATH \
#     --devices="npu:$DEVICE" \
#     --port $PORT \
#     --master_node_addr=$MASTER_NODE_ADDR \
#     --nnodes=$NNODES \
#     --max_memory_utilization=0.90 \
#     --model_id=Qwen2.5-VL-7B \
#     --max_tokens_per_batch=8000 \
#     --max_seqs_per_batch=2 \
#     --enable_mla=false \
#     --block_size=128 \
#     --enable_prefix_cache=false \
#     --enable_chunked_prefill=false \
#     --enable_schedule_overlap=false \
#     --node_rank=$i  \
#     --communication_backend="lccl" \
#     --use_contiguous_input_buffer=false \
#     --enable_shm=true \
#     --enable_graph=false \
#     --enable_graph_no_padding=false \
#     --task="generate" \
#     --backend vlm" --type model tensor -o /export/home/wangyunlong.115/workspace/check -child False -er 0,6 > $LOG_FILE 2>&1 &
# done


# I20260303 19:57:34.829602 1453664 request.cpp:90] x-request-id: , x-request-time: , request_id: chatcmpl-8808701038679435387-s5qbKBis8cMXquimpR8FAG, sequence 0, max_tokens: 100, temperature: 0, finish_reason: length, prompt_tokens: 702, generated_tokens: 100, ttft: 119.0ms, total_latency: 1655.3ms, avg tpot: 15.5ms, generation speed: 65.1 tokens/s
# I20260303 19:57:44.697070 1453661 request.cpp:90] x-request-id: , x-request-time: , request_id: chatcmpl-8808701038679435387-RoECJg93CRf7xcQMjzCH72, sequence 0, max_tokens: 100, temperature: 0, finish_reason: length, prompt_tokens: 702, generated_tokens: 100, ttft: 121.0ms, total_latency: 1674.3ms, avg tpot: 15.7ms, generation speed: 64.4 tokens/s
# I20260303 19:57:52.333576 1453662 request.cpp:90] x-request-id: , x-request-time: , request_id: chatcmpl-8808701038679435387-nr4PQNJ7J6WwuRwFumg8Wk, sequence 0, max_tokens: 100, temperature: 0, finish_reason: length, prompt_tokens: 702, generated_tokens: 100, ttft: 118.0ms, total_latency: 1663.6ms, avg tpot: 15.6ms, generation speed: 64.7 tokens/s
