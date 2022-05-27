# python eval2.py --model './checkpoints/kitti.pth' --dataset 'kitti'
# Validation KITTI: 0.623273, 1.368153
python eval2.py --model './checkpoints/sintel.pth' --dataset 'sintel' --mixed_precision
# python eval2.py --model './checkpoints/sintel.pth' --dataset 'sintel'
# Validation (clean) EPE: 0.872696, 1px: 0.886820, 3px: 0.958008, 5px: 0.973773
# Validation (final) EPE: 1.231935, 1px: 0.850802, 3px: 0.935167, 5px: 0.957903
# python eval2.py --model './checkpoints/things.pth' --dataset 'kitti'
# Validation KITTI: 8.836173, 27.598381
# python eval2.py --model './checkpoints/things.pth' --dataset 'sintel'
# Validation (clean) EPE: 1.708840, 1px: 0.831500, 3px: 0.936761, 5px: 0.957700
# Validation (final) EPE: 3.085267, 1px: 0.782291, 3px: 0.896233, 5px: 0.924515





# python eval2.py --model './checkpoints/kitti.pth' --dataset 'sintel'
# Validation (clean) EPE: 3.478741, 1px: 0.754895, 3px: 0.889514, 5px: 0.919136
# Validation (final) EPE: 4.811804, 1px: 0.712206, 3px: 0.852324, 5px: 0.887256s