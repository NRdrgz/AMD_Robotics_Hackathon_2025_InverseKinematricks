# Mission 2 Commands

This file contains all `lerobot-record` commands for Mission 2 tasks.

## Black Sort

### Record Dataset - Black Sort

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=black_follower \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=black_leader \
  --dataset.repo_id="giacomoran/hackathon_amd_mission2_black_sort" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Pick up the package and place it inside the taped square on the table whose color matches the tape on top of the package (red package-tape to red square, yellow package-tape to yellow square). Place the package fully within the square boundaries." \
  --dataset.root="/home/nico/hackathon_amd_mission2_black_sort_dataset/" \
  --resume=true
```

### Eval - Black Sort (ACT)

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=black_follower \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=giacomoran/eval_hackathon_amd_mission2_black_sort \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick up the package and place it inside the taped square on the table whose color matches the tape on top of the package (red package-tape to red square, yellow package-tape to yellow square). Place the package fully within the square boundaries." \
  --policy.path=giacomoran/hackathon_amd_mission2_black_sort \
  --dataset.push_to_hub=false
```

### Eval - Black Sort (SMOLVLA)

```bash
rm -rf /home/nico/.cache/huggingface/lerobot/giacomoran/eval_hackathon_amd_mission2_black_sort_smolvla \
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=black_follower \
  --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --policy.empty_cameras=1 \
  --dataset.repo_id=giacomoran/eval_hackathon_amd_mission2_black_sort_smolvla \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick up the package and place it inside the taped square on the table whose color matches the tape on top of the package (red package-tape to red square, yellow package-tape to yellow square). Place the package fully within the square boundaries." \
  --policy.path=giacomoran/hackathon_amd_mission2_black_sort_smolvla \
  --dataset.push_to_hub=false
```

## Blue Pick

### Record Dataset - Blue Pick (MacBook Nico)

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A680116011 \
  --robot.id=blue_follower \
  --robot.cameras="{top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5A460824651 \
  --teleop.id=blue_leader \
  --dataset.repo_id="giacomoran/hackathon_amd_mission2_blue_pick" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Pick up the package and place it on the black conveyor belt." \
  --resume=true
```

### Record Dataset - Blue Pick (AMD Laptop)

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM3 \
  --robot.id=blue_follower \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM2 \
  --teleop.id=blue_leader \
  --dataset.repo_id="giacomoran/hackathon_amd_mission2_blue_pick" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Pick up the package and place it on the black conveyor belt." \
  --resume=true
```

### Eval - Blue Pick ACT (MacBook Nico)

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A680116011 \
  --robot.cameras="{top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --robot.id=blue_follower \
  --display_data=false \
  --dataset.repo_id=giacomoran/eval_blue_pick_act \
  --dataset.single_task="Pick up the package and place it on the black conveyor belt." \
  --policy.path=giacomoran/hackathon_amd_mission2_blue_pick \
  --dataset.push_to_hub=false
```

### Eval - Blue Pick ACT (AMD Laptop)

```bash
rm -rf /home/nico/.cache/huggingface/lerobot/giacomoran/eval_blue_pick
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM3 \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}" \
  --robot.id=blue_follower \
  --display_data=false \
  --dataset.repo_id=giacomoran/eval_blue_pick \
  --dataset.single_task="Pick up the package and place it on the black conveyor belt." \
  --policy.path=giacomoran/hackathon_amd_mission2_blue_pick \
  --dataset.push_to_hub=false
```

### Eval - Blue Pick ACT v2

```bash
rm -rf /home/nico/.cache/huggingface/lerobot/giacomoran/eval_blue_pick_act_v2
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM3 \
  --dataset.episode_time_s=1200 \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}" \
  --robot.id=blue_follower \
  --display_data=false \
  --dataset.repo_id=giacomoran/eval_blue_pick_act_v2 \
  --dataset.single_task="Pick up the package and place it on the black conveyor belt." \
  --policy.path=giacomoran/hackathon_amd_mission2_blue_pick_act_v2 \
  --dataset.push_to_hub=false
```


### Eval - Blue Pick SMOLVLA (MacBook Nico)

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A680116011 \
  --robot.cameras="{camera1: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --policy.empty_cameras=1 \
  --robot.id=blue_follower \
  --display_data=false \
  --dataset.repo_id=giacomoran/eval_blue_pick_smolvla \
  --dataset.single_task="Pick up the package and place it on the black conveyor belt." \
  --policy.path=giacomoran/hackathon_amd_mission2_blue_pick_smolvla \
  --dataset.push_to_hub=false
```

### Eval - Blue Pick SMOLVLA (AMD Laptop)

```bash
rm -rf /home/nico/.cache/huggingface/lerobot/giacomoran/eval_blue_pick_smolvla
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM3 \
  --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}" \
  --policy.empty_cameras=1 \
  --robot.id=blue_follower \
  --display_data=false \
  --dataset.repo_id=giacomoran/eval_blue_pick_smolvla \
  --dataset.single_task="Pick up the package and place it on the black conveyor belt." \
  --policy.path=giacomoran/hackathon_amd_mission2_blue_pick_smolvla \
  --dataset.push_to_hub=false
```

## Black Flip

### Record Dataset - Black Flip

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=black_follower \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=black_leader \
  --dataset.repo_id="giacomoran/hackathon_amd_mission2_black_flip" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Flip the package over on the conveyor belt so that the colored tape is visible, then place it back in its original position." \
  --dataset.root="/home/nico/hackathon_amd_mission2_black_flip_dataset/" \
  --resume=true
```

### Eval - Black Flip ACT

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=black_follower \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=giacomoran/eval_hackathon_amd_mission2_black_flip_act \
  --dataset.num_episodes=1 \
  --dataset.single_task="Flip the package over on the conveyor belt so that the colored tape is visible, then place it back in its original position." \
  --policy.path=giacomoran/hackathon_amd_mission2_black_flip_act \
  --dataset.push_to_hub=false
```

### Eval - Black Flip (SMOLVLA)

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=black_follower \
  --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --policy.empty_cameras=1 \
  --dataset.repo_id=giacomoran/eval_hackathon_amd_mission2_black_flip_smolvla \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick up the package and place it inside the taped square on the table whose color matches the tape on top of the package (red package-tape to red square, yellow package-tape to yellow square). Place the package fully within the square boundaries." \
  --policy.path=giacomoran/hackathon_amd_mission2_black_flip_smolvla \
  --dataset.push_to_hub=false
```

