#!/bin/bash
### get the raw-sbn data: raw-sbn and corresponding sentences... ###
#cd SBN-data
#wget "https://pmb.let.rug.nl/releases/pmb-4.0.0.zip"
#unzip pmb-4.0.0.zip
#mv pmb-4.0.0 4.0.0
#rm pmb-4.0.0.zip

cd /home/p289796/SBN-data/4.0.0/data/en

echo "-------------Getting all the sbn data...------------------"
for type in gold silver; do
  cd ${type}/
   #find ./${*}/${*}/ -type f -name "en.drs.sbn" | xargs -I{} sh -c "cat {}; echo ''" > sbn.txt
   python /home/p289796/SBN-data/SBN_data_preprocess/0_get_raw_file.py -i sbn.txt -ipath ./ -o all_sbn.txt
  cd -
done

echo "-------------Splitting gold data to get dev and test---"
raw_sbn_path=/home/p289796/SBN-data/4.0.0/data/en
bash_dir=/home/p289796/SBN-data/
mkdir -p $bash_dir/gold-silver
mkdir -p $bash_dir/gold

python /home/p289796/SBN-data/SBN_data_preprocess/1_split_sbn_data.py -i $raw_sbn_path/gold/all_sbn.txt -n 1000 -ipath $raw_sbn_path/gold -o1 $bash_dir/gold/train.txt -o2 $bash_dir/gold/dev.txt -o3 $bash_dir/gold/test.txt


# Gold + silver


cat $bash_dir/gold/train.txt $raw_sbn_path/silver/all_sbn.txt > $bash_dir/gold-silver/train.txt
cat $bash_dir/gold/train.txt.raw $raw_sbn_path/silver/all_sbn.txt.raw > $bash_dir/gold-silver/train.txt.raw
cp $bash_dir/gold/test* $bash_dir/gold-silver/
cp $bash_dir/gold/dev* $bash_dir/gold-silver/
