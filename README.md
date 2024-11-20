<p align="center">   <a href="https://arxiv.org/abs/2411.11930" style="text-decoration:none;">     <h1><strong>AtomThink: A Slow Thinking Framework for Multimodal Mathematical Task</strong></h1>   </a> </p>

<p align="center">
  <img src="figures/logo.jpg" alt="Project Logo" width="400">
</p>

<p align="center">
  <a href="https://github.com/Quinn777/AtomThink/stargazers">
    <img src="https://img.shields.io/github/stars/Quinn777/AtomThink?style=flat-square" alt="GitHub stars">
  </a>
  <a href="https://github.com/Quinn777/AtomThink/issues">
    <img src="https://img.shields.io/github/issues/Quinn777/AtomThink?style=flat-square" alt="GitHub issues">
  </a>
  <a href="https://arxiv.org/abs/2411.11930">
    <img src="https://img.shields.io/badge/arXiv-2411.11930-red?style=flat-square&logo=arXiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://github.com/Quinn777/AtomThink/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Quinn777/AtomThink?style=flat-square" 
</p>

<p align="center">
🎉Thank you for exploring AtomThink! We warmly invite you to ⭐️ star this repository, share your feedback via issues, and contribute to the project.
</p>

## Contents

- [News](#News)
- [Features](#Features)
- [Datasets](#Datasets)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

---



## News

- **[Upcoming]** Our code, model and datasets will be released soon. Stay tuned for updates!  

- **[2024-11-20]** The paper *AtomThink: A Slow Thinking Framework for Multimodal Mathematical Task* is now available on [arXiv](https://arxiv.org/abs/2411.11930)!

- **[2024-11-20]** Thank you for visiting this repository!

  

## Features

✨ **Key Features**  

- 🧠 Introduces **GPT-o1** style reasoning via long CoT for complex multimodal mathematical tasks.
- 🛠️ Combines a CoT annotation engine, atomic step fine-tuning, and policy search strategies to enhance reasoning.
- 📊 A capability evaluation metric to perform a quality assessment of each reasoning steps.
- ⚡ Test-time scaling law in MLLM.

- 📈 State-of-the-art performance in multimodal mathematical reasoning tasks.

  

📖 **Abstract**

> In this paper, we address the challenging task of multimodal mathematical reasoning by incorporating the ability of  “slow thinking” into multimodal large language models (MLLMs). Contrary to existing methods that rely on direct or fast thinking, our key idea is to construct long chains of thought (CoT) consisting of atomic actions in a step-by-step manner, guiding MLLMs to perform complex reasoning. To this end, we design a novel AtomThink framework composed of three key modules: (i) a CoT annotation engine that automatically generates high-quality CoT annotations to address the lack of high-quality visual mathematical data; (ii) an atomic step fine-tuning strategy that jointly optimizes an MLLM and a policy reward model (PRM) for step-wise reasoning; and (iii) four different search strategies that can be applied with the PRM to complete reasoning. Additionally, we propose AtomMATH, a large-scale multimodal dataset of long CoTs, and an atomic capability evaluation metric for mathematical tasks. Extensive experimental results show that the proposed AtomThink significantly improves the performance of baseline MLLMs, achieving approximately 50\% relative accuracy gains on MathVista and 120\% on MathVerse.

[Read the full paper](https://arxiv.org/abs/your-paper-id)



## Examples

Example1 of AMATH-SFT dataset

<p align="center">   <img src="figures/asft1.png" alt="Description of Image" width="800"> </p>



Example2 of AMATH-SFT dataset

<p align="center">   <img src="figures/asft2.png" alt="Description of Image" width="800"> </p>



Example3 of AMATH-SFT dataset

<p align="center">   <img src="figures/asft3.png" alt="Description of Image" width="800"> </p>



## Usage

⚙️  Our code, model and datasets will be released soon. Stay tuned for updates!  



## Citation

If you find this project useful, please cite our paper:
```
@article{xiang2024atomthink,
  title={AtomThink: A Slow Thinking Framework for Multimodal Mathematical Reasoning
},
  author={Kun Xiang},
  journal={arXiv preprint arXiv:2411.11930},
  year={2024},
  doi={https://doi.org/10.48550}
}
```



## License

📄 **License**  
This project is licensed under the [MIT License](LICENSE).  



<p align="center">✨ Thank you for your interest in our work! ✨</p>
