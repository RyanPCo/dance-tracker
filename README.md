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
