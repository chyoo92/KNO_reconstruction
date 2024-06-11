# KNO event reconstruction

KNO_pid / KNO_vertex folder not used  --updated 20240604

----------- after 20240604 use only combined folder------------


## Conda install
    conda create -n env_name python=3.11

    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

    conda install -c pytorch torchdata

    pip install h5py matplotlib jupyter ipykernel pandas uproot scikit-learn tqdm

    conda install -c conda-forge root


## Training example
    python training.py --config config_file.yaml -o output_folder \
                                                        --type 0 \
                                                        --padding 0 \
                                                        --datasetversion 0 \
                                                        --device 0 \
                                                        --fea 0 \
                                                        --cla 0 \
                                                        --cross_head 0 \
                                                        --cross_dim 0 \
                                                        --self_head 0 \
                                                        --self_dim 0 \
                                                        --n_layers 0 \
                                                        --num_latents 0 \
                                                        --dropout_ratio 0.1 \
                                                        --vtx_1000 0 \
                                                        --nDataLoaders 0 \
                                                        --epoch 0 \
                                                        --batch 0 \
                                                        --learningRate 0.1 \
                                                        --randomseed 1 \
                                                        --loss_type 0
    
model hyper parameter setting in args \
training parameter setting in args or config file \
event path and type(label) setting in config file \

## MC events
MC generation git
https://github.com/chyoo92/KNO_simulation

MC file
https://www.notion.so/changhyun0417/e3f627f41f054cb48718a9866d36d50d

    
    