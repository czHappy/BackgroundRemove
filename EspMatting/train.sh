# shell
# ------------------------------------------
# --dataDir=./dataset/ --saveDir=./pre_trained/ --trainData=human_matting --trainList=train.txt --load=attunet --nThreads=4 --patch_size=256 --train_batch=64 --lr=1e-3 --lrdecayType=keep --nEpochs=3 --save_epoch=1
python3 main.py \
	--dataDir=./dataset \
	--saveDir=./pre_trained \
	--trainData=human_matting \
	--trainList=train.txt \
	--load=attunet \
	--nThreads=4 \
	--patch_size=256 \
	--train_batch=64 \
	--lr=1e-3 \
	--lrdecayType=keep \
	--nEpochs=500 \
	--save_epoch=1 \

python3 main.py \
	--dataDir=./dataset \
	--saveDir=./pre_trained \
	--trainData=human_matting \
	--trainList=train.txt \
	--load=attunet \
	--nThreads=4 \
	--patch_size=256 \
	--train_batch=64 \
	--lr=1e-4 \
	--lrdecayType=keep \
	--nEpochs=800 \
	--save_epoch=1 \
	--finetuning \

python3 main.py \
	--dataDir=./dataset \
	--saveDir=./pre_trained \
	--trainData=human_matting \
	--trainList=train.txt \
	--load=attunet \
	--nThreads=3 \
	--patch_size=256 \
	--train_batch=64 \
	--lr=1e-5 \
	--lrdecayType=keep \
	--nEpochs=1200 \
	--save_epoch=1 \
	--finetuning \
	--train_refine
