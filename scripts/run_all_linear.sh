bash ./scripts/run.sh 4 configs/gpu4_1d.py dim linear
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_1d.py bs linear
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_2d.py dim linear
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_2d.py bs linear
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_2p5d.py dim linear
sleep 5
bash ./scripts/run.sh 4 configs/gpu4_2p5d.py bs linear
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_1d.py dim linear
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_1d.py bs linear
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_2p5d.py dim linear
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_2p5d.py bs linear
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_3d.py dim linear
sleep 5
bash ./scripts/run.sh 8 configs/gpu8_3d.py bs linear