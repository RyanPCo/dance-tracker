# Run AlphaPose
python scripts/demo_inference.py \
  --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
  --checkpoint pretrained_models/fast_res50_256x192.pth \
  --video ../videos/colorful_test.mp4 \
  --outdir ../output_colorful \
  --pose_track \
  --vis_fast \
  --save_video

python scripts/demo_inference.py \
  --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
  --checkpoint pretrained_models/fast_res50_256x192.pth \
  --video ../videos/boywithluv_real.mov \
  --outdir ../output_boywithluv \
  --pose_track \
  --vis_fast \
  --save_video

python scripts/demo_inference.py \
  --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
  --checkpoint pretrained_models/fast_res50_256x192.pth \
  --video ../videos/boywithluv_real.mov \
  --outdir ../output \
  --pose_track

# JSON Output Format
-image_id: frame number
-category_id: which object was detected (1 for human)
-keypoints: 17 each, (x, y, confidence)
-score: overall confidence
-box: (x, y, width, height)
-idx: person label

# Archived Commands
python run_alphapose.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint pretrained_models/fast_res50_256x192.pth \
    --video ../videos/boywithluv_real.mov \
    --outdir ../output \
    --pose_track --save_video

python run_alphapose.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint pretrained_models/fast_res50_256x192.pth \
    --video ../videos/colorful_test.mp4 \
    --outdir ../output \
    --pose_track --save_video
