bash ./scripts/run.sh 4 configs/gpu4_1d.py dim layernorm
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_1d.py bs layernorm
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_2d.py dim layernorm
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_2d.py bs layernorm
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_2p5d.py dim layernorm
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_2p5d.py bs layernorm
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_1d.py dim layernorm
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_1d.py bs layernorm
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_2p5d.py dim layernorm
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_2p5d.py bs layernorm
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_3d.py dim layernorm
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_3d.py bs layernorm
