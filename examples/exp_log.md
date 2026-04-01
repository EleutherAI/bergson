currently prototype.py:

with pythia 70m:
64 tok:  Spearman=0.2153  mean_ratio=-0.066  ratio_range=[-9.496, 9.443]

with gpt2:
64 tok:  Spearman=0.9131  mean_ratio=inf
128 tok:  Spearman=0.9342  mean_ratio=0.747
256 tok:  Spearman=0.8740  mean_ratio=inf
512 tok:  Spearman=0.8855  mean_ratio=inf

with gpt2 eps_root=1e-4 instead of 1e-2
64 tok:  LDS Spearman=0.5669

with gpt2 eps_root=1e-8
64 tok:  LDS Spearman=0.4707

with gpt2 eps_root=1e-1
64 tok:  LDS Spearman=0.8977

with gpt2 disabling autocast
64 tok:  Spearman=0.1552  mean_ratio=0.013

gpt 2 with eps_root 1e-2, lds from double backwards
64 tok:  LDS Spearman=0.8782
128 tok:  LDS Spearman=0.9489
256 tok:  LDS Spearman=0.8722
512 tok:  LDS Spearman=0.9098

above, but with truncate and pad, instead of chunk and tokenize
64 tok:  LDS Spearman=0.7293
128 tok:  LDS Spearman=0.8571
256 tok:  LDS Spearman=-0.2812
512 tok:  LDS Spearman=0.8391

above, but with truncate and pad, but first filtering out empty sequences
64 tok:  LDS Spearman=0.7820
128 tok:  LDS Spearman=0.8361
256 tok:  LDS Spearman=0.6647
512 tok:  LDS Spearman=0.8842

200 seqs, 1e-2
64 tok:  LDS Spearman=0.9579

with lr schedule from the paper, 1e-2
512 tok:  LDS Spearman=-0.1624

with lr schedule from double backwards, 1e-2
512 tok:  LDS Spearman=0.8612

without weight decay

512 tok:  LDS Spearman=0.9338
