| dtype | share | SMs | batch | prefill tok/s off | guard | guard/off | generate tok/s off | guard | guard/off | calls rewritten | guard-off result | guard-on result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 100 % | 132 | 1 | 19638 | 19486 | 99 % | 39.2 | 38.2 | 97 % | 44 % | correct | correct |
| bf16 | 100 % | 132 | 8 | 30585 | 30426 | 99 % | 296.8 | 287.7 | 97 % | 62 % | correct | correct |
| bf16 | 100 % | 132 | 32 | 31159 | 31136 | 100 % | 964.6 | 964.6 | 100 % | 54 % | correct | correct |
| bf16 | 70 % | 92 | 1 | 18624 | 18460 | n/a | 38.9 | 38.9 | n/a | 100 % | **wrong** logit_rel=inf, top1=0.00, gen_match=0.00 | correct |
| bf16 | 70 % | 92 | 8 | 24683 | 24637 | n/a | 288.9 | 288.9 | n/a | 100 % | **wrong** logit_rel=inf, top1=0.38, gen_match=0.38 | correct |
| bf16 | 70 % | 92 | 32 | 25515 | 25174 | 99 % | 918.0 | 912.5 | 99 % | 100 % | correct | correct |
| bf16 | 50 % | 66 | 1 | 15390 | 15380 | 100 % | 39.7 | 39.2 | 99 % | 100 % | correct | correct |
| bf16 | 50 % | 66 | 8 | 18809 | 18827 | 100 % | 286.3 | 286.0 | 100 % | 97 % | correct | correct |
| bf16 | 50 % | 66 | 32 | 19303 | 19328 | 100 % | 851.4 | 849.7 | 100 % | 98 % | correct | correct |
| bf16 | 40 % | 52 | 1 | 12738 | 12980 | n/a | 38.8 | 38.9 | n/a | 100 % | **wrong** logit_rel=inf, top1=0.00, gen_match=0.00 | correct |
| bf16 | 40 % | 52 | 8 | 15015 | 15081 | n/a | 274.1 | 274.6 | n/a | 100 % | **wrong** logit_rel=inf, top1=0.50, gen_match=0.00 | correct |
| bf16 | 40 % | 52 | 32 | 15531 | 15511 | n/a | 781.4 | 777.5 | n/a | 100 % | **wrong** logit_rel=inf, top1=0.75, gen_match=0.00 | correct |
| bf16 | 25 % | 32 | 1 | 8899 | 8859 | 100 % | 38.9 | 38.5 | 99 % | 100 % | correct | correct |
| bf16 | 25 % | 32 | 8 | 9733 | 9700 | 100 % | 258.1 | 254.3 | 99 % | 100 % | correct | correct |
| bf16 | 25 % | 32 | 32 | 9832 | 9801 | 100 % | 641.8 | 636.6 | 99 % | 100 % | correct | correct |
| bf16 | 14 % | 18 | 1 | 5145 | 5159 | n/a | 37.2 | 36.7 | n/a | 64 % | **wrong** logit_rel=0.0e+00, top1=1.00, gen_match=0.02 | correct |
| bf16 | 14 % | 18 | 8 | 5506 | 5507 | n/a | 222.3 | 215.9 | n/a | 68 % | **wrong** logit_rel=0.0e+00, top1=1.00, gen_match=0.02 | correct |
| bf16 | 14 % | 18 | 32 | 5554 | 5562 | n/a | 461.5 | 456.8 | n/a | 69 % | **wrong** logit_rel=8.6e-02, top1=1.00, gen_match=0.02 | correct |
| fp16 | 100 % | 132 | 1 | 19849 | 19243 | 97 % | 39.7 | 38.5 | 97 % | 61 % | correct | correct |
| fp16 | 100 % | 132 | 8 | 29796 | 29782 | 100 % | 295.8 | 276.0 | 93 % | 87 % | correct | correct |
| fp16 | 100 % | 132 | 32 | 30676 | 30611 | 100 % | 981.5 | 967.7 | 99 % | 54 % | correct | correct |
| fp16 | 40 % | 52 | 1 | 12785 | 12963 | n/a | 39.3 | 38.5 | n/a | 100 % | **wrong** logit_rel=inf, top1=0.00, gen_match=0.00 | correct |
| fp16 | 40 % | 52 | 8 | 14981 | 15161 | n/a | 271.1 | 272.2 | n/a | 100 % | **wrong** logit_rel=inf, top1=0.50, gen_match=0.00 | correct |
| fp16 | 40 % | 52 | 32 | 15504 | 15581 | n/a | 776.4 | 774.3 | n/a | 100 % | **wrong** logit_rel=inf, top1=0.75, gen_match=0.00 | correct |
| fp16 | 25 % | 32 | 1 | 8915 | 8886 | 100 % | 39.6 | 38.9 | 98 % | 100 % | correct | correct |
| fp16 | 25 % | 32 | 8 | 9703 | 9725 | 100 % | 261.6 | 257.0 | 98 % | 100 % | correct | correct |
| fp16 | 25 % | 32 | 32 | 9781 | 9798 | 100 % | 642.9 | 641.9 | 100 % | 100 % | correct | correct |
