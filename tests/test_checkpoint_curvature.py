"""MLP recomputation preserves KFAC factors and collects each forward only once."""
import copy
import pytest
from collections import Counter
import torch
from peft import LoraConfig,TaskType,get_peft_model
from transformers import Olmo3Config,Olmo3ForCausalLM
from bergson.config import IndexConfig,HessianConfig
from bergson.collector.collector import fwd_bwd_hessian_factory
from bergson.hessians.kfac import CovarianceCollector
from bergson.score.output_model import install_output_loss_head
from bergson.utils.activation_offload import save_on_cpu_preserve_attention_bias
from bergson.utils.recompute import checkpoint_mlps
from bergson.utils.worker_utils import extract_peft_target_modules


@pytest.mark.parametrize("attention",[False,True])
@pytest.mark.parametrize("phase",["kfac","lambda"])
def test_checkpoint_preserves_all_adapter_covariances(tmp_path,attention,phase):
    torch.manual_seed(42)
    cfg=Olmo3Config(vocab_size=41,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,pad_token_id=0,layer_types=['full_attention','sliding_attention'],sliding_window=4)
    cfg._attn_implementation='sdpa'
    model=get_peft_model(Olmo3ForCausalLM(cfg).double(),LoraConfig(task_type=TaskType.CAUSAL_LM,r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    for name,param in model.named_parameters():
        if 'lora_B' in name:torch.nn.init.normal_(param,std=.02)
    model.eval();other=copy.deepcopy(model)
    assert checkpoint_mlps(other)==2
    if attention:
        from bergson.utils.recompute import checkpoint_attentions
        assert checkpoint_attentions(other)==2
    x=torch.randint(1,41,(4,13));y=x.clone();y[:,:3]=-100;y[1,-2:]=-100
    config=IndexConfig(run_path='/unused',loss_fn='dpo',dpo_beta=5)
    run=fwd_bwd_hessian_factory(config,HessianConfig())
    results=[]
    for ordinal,m in enumerate((model,other)):
        install_output_loss_head(m,chunk_rows=3,checkpoint_reverse=True)
        arguments=dict(model=m.base_model,target_modules=extract_peft_target_modules(m),path=str(tmp_path/str(ordinal)),offload_to_cpu=True)
        if phase=="kfac":
            collector=CovarianceCollector(**arguments,dtype=torch.float64)
        else:
            from bergson.collector.collector import HookCollectorBase
            from bergson.hessians.eigenvectors import LambdaCollector
            from safetensors.torch import save_file
            info=HookCollectorBase.discover_targets(m.base_model,arguments['target_modules'])
            for name,axis in (("eigen_activation_sharded",1),("eigen_gradient_sharded",0)):
                directory=tmp_path/str(ordinal)/name;directory.mkdir(parents=True)
                save_file({key:torch.eye(shape[axis],dtype=torch.float64).cpu() for key,(_,shape,_) in info.items()},str(directory/'shard_0.safetensors'))
            collector=LambdaCollector(**arguments,is_dpo=True)

        counts=Counter();old=collector.forward_hook
        def hook(module,a):counts[module._name]+=1;old(module,a)
        collector.forward_hook=hook
        with collector:
            collector.with_batch(y!=-100)
            with save_on_cpu_preserve_attention_bias():
                loss=run(m,x,y,{'reference_logps':[-3.,-2.9,-3.1,-3.2]})
            if phase=="kfac":
                a,s=collector.A_cov_dict,collector.S_cov_dict
            else:
                a,s=collector.eigenvalue_corrections,{}
                assert not collector.transformed_a_cache
            results.append((loss.detach(),{k:v.clone() for k,v in a.items()},{k:v.clone() for k,v in s.items()},counts))
    assert len(results[0][3])==28
    assert results[0][3]==results[1][3] and set(results[1][3].values())=={1}
    torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-10,atol=1e-10)
    for section in (1,2):
        for name in results[0][section]:torch.testing.assert_close(results[0][section][name],results[1][section][name],rtol=1e-8,atol=1e-9)
