# Section 1 Training Example 

# 1.0 Typical usage 

conda activate ebp 
python src/model_point_torch.py

# 1.1 Run experiment for all 10 classes 

for cate in night_stand toilet monitor chair bathtub sofa bed table desk dresser 
do
    python src/model_point_torch.py -category ${cate}
done

# 1.2 Try different learning rate for a single class 

lrs=(1e-2 2e-2 5e-2 1e-1)
for i in 0 1 2 3
do
    python src/model_point_torch.py -lr ${lrs[i]} -output_dir lr${lrs[i]}
done

# 1.3 Distribute to multiple GPU

i=0
for cate in night_stand toilet monitor chair bathtub sofa bed table desk dresser 
do 
    python src/model_point_torch.py -category ${cate} -output_dir examples -cuda ${i} &
    i=$(((${i}+1)%4))
done 

# 1.4 Run single training on multiple GPU
# In this way, count batch size manually. e.g. 4 GPU with -batch_size 32 is equals to a real batch size of 128.

python src/model_point_torch.py -cuda 0,1,2,3



