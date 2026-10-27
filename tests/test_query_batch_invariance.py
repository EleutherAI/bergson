"""Splitting a weighted DPO query batch preserves every pair's adapter gradient."""
import torch
from types import SimpleNamespace
from bergson.config import IndexConfig,PreprocessConfig
from bergson.collector.gradient_collectors import GradientCollector
from bergson.collector.collector import fwd_bwd_factory
from bergson.gradients import GradientProcessor
from bergson.data import expand_dpo_dataset,pad_and_tensor
from datasets import Dataset
from peft import LoraConfig,TaskType,get_peft_model
from transformers import Olmo3Config,Olmo3ForCausalLM
from bergson.utils.worker_utils import extract_peft_target_modules


def test_two_pair_query_batch_matches_individual_pairs():
    torch.manual_seed(42)
    c=Olmo3Config(vocab_size=41,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,pad_token_id=0,eos_token_id=1,use_cache=False)
    m=get_peft_model(Olmo3ForCausalLM(c).double(),LoraConfig(task_type=TaskType.CAUSAL_LM,r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']));m.eval()
    rows=[]
    for n in (9,11):
        x=torch.randint(1,41,(n,)).tolist();z=torch.randint(1,41,(n-2,)).tolist()
        rows.append(dict(chosen_input_ids=x,rejected_input_ids=z,chosen_labels=[-100,-100]+x[2:],rejected_labels=[-100,-100]+z[2:],chosen_length=len(x),rejected_length=len(z),reference_chosen_logps=-3.7,reference_rejected_logps=-3.8,weight=.3+n*.01))
    data=expand_dpo_dataset(Dataset.from_list(rows));cfg=IndexConfig(run_path='/unused',loss_fn='dpo',dpo_beta=5,example_weight_column='weight',skip_index=True,skip_hessians=True,projection_dim=0)
    fn=fwd_bwd_factory(cfg);results=[]
    for batches in ([[0,1,2,3]],[[0,2],[1,3]]):
        found={}
        def scorer(indices,grads):
            for row,index in enumerate(indices):found[index]={k:v[row].detach().clone() for k,v in grads.items()}
        collector=GradientCollector(model=m.base_model,target_modules=extract_peft_target_modules(m),data=data,cfg=cfg,processor=GradientProcessor(),preprocess_cfg=PreprocessConfig(),scorer=scorer)
        with collector:
            for indices in batches:
                batch=data[indices];x,y,mask=pad_and_tensor(batch['input_ids'],labels=batch['labels'],device=torch.device('cpu'),sync_max_len=False)
                collector.with_batch(mask);loss=fn(m,x,y,batch);collector.process_batch(indices,losses=loss)
        results.append(found)
    assert results[0].keys()==results[1].keys()=={0,1}
    for i in results[0]:
        for key in results[0][i]:torch.testing.assert_close(results[0][i][key],results[1][i][key],rtol=2e-5,atol=2e-8)
