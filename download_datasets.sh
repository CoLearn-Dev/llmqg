if [ ! -d "data" ]; then
  mkdir data
fi

# HotpotQA
if [ ! -d "data/hotpotqa-fullwiki" ]; then
  mkdir -p data/hotpotqa-fullwiki
fi

if [ ! -f "data/hotpotqa-fullwiki/hotpot_dev_fullwiki_v1.json" ]; then
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json -P data/hotpotqa-fullwiki/
fi

if [ ! -f "data/hotpotqa-fullwiki/hotpot_train_v1.1.json" ]; then
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -P data/hotpotqa-fullwiki/
fi

# TriviaQA
if [ ! -d "data/triviaqa-unfiltered" ]; then
  wget https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz -P data/
  tar -xvzf data/triviaqa-unfiltered.tar.gz -C data/ 
  rm data/triviaqa-unfiltered.tar.gz
fi