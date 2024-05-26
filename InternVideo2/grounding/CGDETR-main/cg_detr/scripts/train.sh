dset_name=hl
ctx_mode=video_tef
v_feat_types=intern
t_feat_type=intern
results_root=results_qvhighlights
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features/qvhighlight

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"intern"* ]]; then
  v_feat_dirs+=(${feat_root}/qvhighlight_internvideo2_videoclip_6b_w2s)
  (( v_feat_dim += 768 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
fi
if [[ ${t_feat_type} == *"intern"* ]]; then
  t_feat_dir=(${feat_root}/qvhighlight_internvideo2_llama_text_feature)
  t_feat_dim=4096
fi


#### training
bsz=32
enc_layers=3
dec_layers=3
t2v_layers=2
moment_layers=1
dummy_layers=2
sent_layers=1
max_v_l=75
max_q_l=32

PYTHONPATH=$PYTHONPATH:. python cg_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--lr 0.0002 \
--results_root ${results_root} \
--exp_id ${exp_id} \
--enc_layers ${enc_layers} \
--dec_layers ${dec_layers} \
--t2v_layers ${t2v_layers} \
--moment_layers ${moment_layers} \
--dummy_layers ${dummy_layers} \
--sent_layers ${sent_layers} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
${@:1}
