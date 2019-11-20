python main.py test \
   --model_name="ResNet18" \
   --batch_size=128 \
   --num_workers=8 \
   --pic_size=32 \
   --dataset="Cifar" \
   --test_root='./data/cifar10/' \
   --test_name="cifar10_noisy_0.3.txt" \
   --num_classes=10 \
   --label_mode='SelNLPL' \
   --load_model_path="checkpoints/ResNet18_Cifar_300_epoch_SelPLNL_Negetive=10/299.ckpt" \
   --result_name="result/test.csv"

