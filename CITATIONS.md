# Academic Citations & Attributions

This project reproduces and extends the research methodology presented in:

---

## 1. Primary Research Reference

```bibtex
@inproceedings{deng2023llms,
  title={What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis},
  author={Deng, Xiang and Bashlovkina, Vasilisa and Han, Feng and Baumgartner, Simon and Bendersky, Michael},
  booktitle={Companion Proceedings of the ACM Web Conference 2023 (WWW '23 Companion)},
  pages={326--330},
  year={2023},
  publisher={Association for Computing Machinery},
  address={Austin, TX, USA},
  doi={10.1145/3543873.3587352},
  url={https://arxiv.org/abs/2212.11311}
}
```

---

## 2. Benchmark Datasets

### FiQA 2018 Task 1 (Financial Opinion Mining and Question Answering)
```bibtex
@inproceedings{maia201818,
  title={WWW'18 Open Challenge: Financial Opinion Mining and Question Answering},
  author={Maia, Macedo and Handschuh, Siegfried and Freitas, Andr{\'e} and Davis, Brian and McDermott, Ross and Zarrouk, Manel and Balahur, Alexandra},
  booktitle={Companion of the The Web Conference 2018 on The Web Conference 2018},
  pages={1941--1942},
  year={2018}
}
```

---

## 3. Pretrained Financial Transformer Models

### FinBERT (ProsusAI)
```bibtex
@article{araci2019finbert,
  title={FinBERT: Financial Sentiment Analysis with Pre-trained Language Models},
  author={Araci, Dogu},
  journal={arXiv preprint arXiv:1908.10063},
  year={2019},
  url={https://huggingface.co/ProsusAI/finbert}
}
```

### FinBERT-Tone (HKUST)
```bibtex
@article{yang2020finbert,
  title={FinBERT: A Large Language Model for Extracting Information from Financial Text},
  author={Yang, Yi and Uy, Mark Christopher S and Huang, Allen},
  journal={Contemporary Accounting Research},
  volume={37},
  number={4},
  pages={2379--2405},
  year={2020},
  url={https://huggingface.co/yiyanghkust/finbert-tone}
}
```

### DistilBERT (Student Model Backbone)
```bibtex
@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}
```

---

## 4. Key Libraries & Software Frameworks

- **Hugging Face Transformers:** Wolf et al., *Transformers: State-of-the-Art Natural Language Processing*, EMNLP 2020.
- **PyTorch:** Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019.
- **Scikit-Learn:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011.
- **VADER Sentiment:** Hutto & Gilbert, *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*, ICWSM 2014.
- **Streamlit:** Interactive application deployment framework.
- **Plotly:** Interactive scientific and financial graphing library.
- **yfinance:** Python interface to historical market price data.

---

## 5. Explicit Attribution of Work

| Component | Origin | Description |
|:---|:---|:---|
| **6-Shot CoT Prompting** | Deng et al. (2023) | In-context demonstration structure and financial TL;DR reasoning instructions |
| **8-Path Stochastic Sampling** | Deng et al. (2023) | Generating $K=8$ paths at $T=0.5$ for direct evaluation via majority vote |
| **Soft Agreement Score** | Deng et al. (2023) | Calculating continuous teacher consensus targets in $[-1.0, 1.0]$ |
| **Consistency Filtering** | Deng et al. (2023) | Pruning weak labels with agreement $< 5/8$ before student distillation |
| **Regression Distillation** | Deng et al. (2023) | Training student model using MSE loss on soft teacher targets |
| **Provider Abstraction** | Our Implementation | Decoupled client supporting OpenAI, Gemini, Claude, Ollama, and Mock providers |
| **DistilBERT Modernization** | Our Implementation | Modern sub-word transformer replacement for proprietary Charformer |
| **Traditional Baselines** | Our Extension | TF-IDF + Logistic Regression, Linear SVM, Naive Bayes, and VADER |
| **Financial Market Analysis**| Our Extension | Daily signal aggregation, forward returns (1d, 3d, 5d), lead/lag correlations, event studies |
| **Interactive Dashboard** | Our Extension | 8-Page academic Streamlit research application |
