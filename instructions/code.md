NOTE: All commands must be run from inside the `point-policy` directory. In `point_policy/cfg/config.yaml`, set `root_dir`  to `path/to/repo` and `data_dir` to `path/to/data/expert_demos`. Also, set 'root_dir' in `point_policy/cfg/suite/point_cfg.yaml` to `path/to/repo`.

## Training

- BC
```
python train.py agent=baku suite=baku dataloader=baku eval=false suite/task/franka_env=<task_name> experiment=bc
```

- BC w/ Depth
```
python train.py agent=baku suite=baku dataloader=baku eval=false suite/task/franka_env=<task_name> suite.gt_depth=true experiment=bc_with_depth
```

- MT-π
```
python train.py agent=mtpi suite=mtpi dataloader=mtpi eval=false suite.use_robot_points=true suite.use_object_points=true suite/task/franka_env=<task_name> experiment=mtpi
```

- P3PO
```
python train.py agent=p3po suite=p3po dataloader=p3po eval=false suite.use_robot_points=true suite.use_object_points=true suite/task/franka_env=<task_name> experiment=p3po
```

- Point Policy
```
python train.py agent=point_policy suite=point_policy dataloader=point_policy eval=false suite.use_robot_points=true suite.use_object_points=true suite/task/franka_env=<task_name> experiment=point_policy
```

## Inference

- BC
```
python eval.py agent=baku suite=baku dataloader=baku eval=true experiment=eval_bc suite/task/franka_env=<task_name> bc_weight=path/to/bc/weight
```

- BC w/ Depth
```
python eval.py agent=baku suite=baku dataloader=baku eval=true experiment=eval_bc_with_depth suite.gt_depth=true suite/task/franka_env=<task_name> bc_weight=path/to/bc/weight
```

- MT-π
```
python eval_point_track.py agent=mtpi suite=mtpi dataloader=mtpi eval=true suite.use_robot_points=true suite.use_object_points=false experiment=eval_mtpi suite/task/franka_env=<task_name> bc_weight=path/to/bc/weight
```

- P3PO
```
python eval_point_track.py agent=p3po suite=p3po dataloader=p3po eval=true suite.use_robot_points=true suite.use_object_points=true experiment=eval_p3po suite/task/franka_env=<task_name> bc_weight=path/to/bc/weight
```

- Point Policy
```
python eval_point_track.py agent=point_policy suite=point_policy dataloader=point_policy eval=true suite.use_robot_points=true suite.use_object_points=true experiment=eval_point_policy suite/task/franka_env=<task_name> bc_weight=path/to/bc/weight
```
