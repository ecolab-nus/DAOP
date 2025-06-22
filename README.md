# 🎻 DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference [[paper]](https://arxiv.org/pdf/2501.10375)

(This repository is a proof-of-concept and still under heavy construction)

DAOP is an on-device MoE inference engine to optimize parallel GPU-CPU execution. DAOP dynamically allocates experts between CPU and GPU based on per-sequence activation patterns, and selectively pre-calculates predicted experts on CPUs to minimize transfer latency. This approach enables efficient resource utilization across various expert cache ratios while maintaining model accuracy through a novel graceful degradation mechanism. It allows you to run **unquantized Mixtral-8x7B model (>90GB of parameters) with >4.5 token/s and unquantized Phi-3.5 MoE model (>80GB of parameters) with >8.2 token/s on a single A6000 GPU**.

## Update
- [2024/12] We published an [arxiv preprint](https://arxiv.org/abs/2501.10375)
- [2025/02] We released the repository.

## Usage
```bash
pip install -r requirements.txt
bash testing.sh
```

## Contact
Please feel free to contact us if there are any issues or discrepancies with the experimental data.

## Citation
If you use DAOP in your research, please cite the following paper. 
```
@article{zhang2024daop,
  title={DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference},
  author={Zhang, Yujie and Aggarwal, Shivam and Mitra, Tulika},
  journal={arXiv preprint arXiv:2501.10375},
  year={2024}
}
@inproceedings{zhang2025daop,
  title={DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference},
  author={Zhang, Yujie and Aggarwal, Shivam and Mitra, Tulika},
  booktitle={2025 Design, Automation \& Test in Europe Conference (DATE)},
  pages={1--7},
  year={2025},
  organization={IEEE}
}
```
