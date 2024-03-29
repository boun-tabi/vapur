[![Vapur](https://tabilab.cmpe.boun.edu.tr/vapur/static/vapur.jpeg)](https://tabilab.cmpe.boun.edu.tr/vapur)

Vapur is an online entity-oriented search engine for the COVID-19 anthology. Vapur is empowered with a semantic inverted index that is created through named entity recognition and relation extraction on CORD-19 abstracts.

In order to run scripts from scratch, please follow these:
* Download [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
* Install [Genia Sentence Splitter](http://www.nactem.ac.uk/y-matsu/geniass/)
* Run [BERN](https://github.com/dmis-lab/bern) on 127.0.0.1:8888
* Download our pretrained binary relation extraction model from [Google Drive](https://drive.google.com/file/d/19_eKPAAwug49JNlNneJoNVhl5VDMP1VH/view?usp=sharing) and run the relevant script to find related protein - compound pairs in COVID-19 literature.

Check out our search engine at https://tabilab.cmpe.boun.edu.tr/vapur and related Flask code in [Demo](https://github.com/boun-tabi/vapur/tree/master/Demo) folder.

[Appendix](https://docs.google.com/spreadsheets/d/1Hu87-Qm4CQQkdE0wwVG4p3QcPxaXCKE3ZRTvDQw48Bo/edit?usp=sharing)
