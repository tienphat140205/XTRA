XTRA: Cross-Lingual Topic Modeling with Topic and Representation Alignments
1. Prepare Environment
To set up the required environment, install the following dependencies:
gensim==4.3.3
matplotlib==3.7.2
numpy==1.26.4
PyYAML==6.0.2
scikit_learn==1.2.2
scipy==1.15.3
torch==2.6.0+cu124

2. Prepare doc embeddings
run this file create_embeddings_all_datasets.py:
python create_embeddings_all_datasets.py
3. Training
To run the training and evaluation, execute the following command:
cd XTRA
bash run.sh

4. CNPMI Score Computation
To compute the CNPMI score, follow these steps:

Clone the CNPMI evaluation toolkit:
git clone https://github.com/BobXWu/CNPMI.git


Compute the CNPMI score using the following command:
cd CNPMI
python CNPMI.py \
    --topics1 {path_to_language1_topics} \
    --topics2 {path_to_language2_topics} \
    --ref_corpus_config ./configs/ref_corpus/{lang1_lang2}.yaml



