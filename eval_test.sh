python -W ignore tools/run.py --tasks captioning --datasets m4c_textcaps --model cnmt \
--config configs/cnmt_config.yml \
--save_dir save/eval/ \
--run_type inference --evalai_inference 1 \
--resume_file save/cnmt/m4c_textcaps_cnmt/best.ckpt