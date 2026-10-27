"""Chunked checkpointed losses preserve query/DPO-GGN gradients and bound saves."""
import copy
import pytest
import torch
from peft import LoraConfig,TaskType,get_peft_model
from transformers import Olmo3Config,Olmo3ForCausalLM
from bergson.config import IndexConfig,HessianConfig
from bergson.collector.collector import fwd_bwd_factory,fwd_bwd_hessian_factory
from bergson.score.output_model import install_output_loss_head,TOKEN_LOSS_REQUEST

@pytest.mark.parametrize('objective',['ce','dpo','ggn'])
@pytest.mark.parametrize('checkpoint_mlp',[False,'mlp','all'])
def test_streamed_checkpointed_collector_matches_full_logits(objective, checkpoint_mlp):
    torch.manual_seed(42)
    cfg=Olmo3Config(vocab_size=41,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,pad_token_id=0,layer_types=['full_attention','sliding_attention'],sliding_window=4)
    cfg._attn_implementation='sdpa'
    model=get_peft_model(Olmo3ForCausalLM(cfg).double(),LoraConfig(task_type=TaskType.CAUSAL_LM,r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    for name,param in model.named_parameters():
        if 'lora_B' in name:torch.nn.init.normal_(param,std=.02)
    model.eval();other=copy.deepcopy(model)
    install_output_loss_head(model,chunk_rows=3)
    model._bergson_output_loss_head=False
    install_output_loss_head(other,chunk_rows=3,checkpoint_reverse=True)
    if checkpoint_mlp:
        from bergson.utils.recompute import checkpoint_mlps
        assert checkpoint_mlps(other)==2
        if checkpoint_mlp=="all":
            from bergson.utils.recompute import checkpoint_attentions
            assert checkpoint_attentions(other)==2
    x=torch.randint(1,41,(4,13));y=x.clone();y[:,:3]=-100;y[1,-2:]=-100
    batch={'reference_logps':[-3.,-2.9,-3.1,-3.2]}
    config=IndexConfig(run_path='/unused',loss_fn='ce' if objective=='ce' else 'dpo',dpo_beta=5)
    run=fwd_bwd_hessian_factory(config,HessianConfig()) if objective=='ggn' else fwd_bwd_factory(config)
    results=[]
    for m in (model,other):
        grads={};handles=[]
        for name,param in m.named_parameters():
            if param.requires_grad:
                handles.append(param.register_hook(lambda value,key=name:grads.__setitem__(key,value.detach().clone())))
        loss=run(m,x,y,batch)
        results.append((loss.detach(),grads))
        for h in handles:h.remove()
    torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-10,atol=1e-10)
    assert results[0][1].keys()==results[1][1].keys()
    for key in results[0][1]:torch.testing.assert_close(results[0][1][key],results[1][1][key],rtol=1e-9,atol=1e-10)

def test_checkpoint_does_not_offload_vocab_weight_per_chunk():
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.head=torch.nn.Linear(16,101,bias=False);self.head.requires_grad_(False)
        def get_output_embeddings(self):return self.head
    m=Model();install_output_loss_head(m,chunk_rows=4,checkpoint_reverse=True)
    hidden=torch.randn(2,31,16,requires_grad=True);labels=torch.randint(0,101,(2,31))
    saved=[]
    def pack(value):saved.append(tuple(value.shape));return value.detach().clone()
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda value:value):
        token=TOKEN_LOSS_REQUEST.set((labels,0.0))
        try:loss=m.head(hidden).sum()
        finally:TOKEN_LOSS_REQUEST.reset(token)
        loss.backward()
    assert (101,16) not in saved and (16,101) not in saved
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
