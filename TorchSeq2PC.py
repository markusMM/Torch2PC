import torch
from typing import Any
from utils import fout_wrapper

print('Running TorchSeq2PC.py with fout_wrapper Integration')

# NEWER

# Perform a forward pass on a Sequential model
# where X,Y are one batch of inputs,labels
# Returns activations for all layers (vhat), loss, and gradient of loss
# wrt last-layer activations (dLdy)
# vhat,Loss,dLdy=FwdPassPlus(model,LossFun,X,Y)
def FwdPassPlus(model,LossFun,X,Y) -> tuple[list[None], Any, torch.Tensor]:
    
    # Number of layers, counting the input as layer 0 
    DepthPlusOne=len(model)+1 

    # Forward pass
    vhat=[None]*DepthPlusOne
    # additional wrapper to ensure that the output of each layer is a single tensor
    vhat[0]=fout_wrapper(X)
    for layer in range(1,DepthPlusOne):
      f=model[layer-1]        
      vhat[layer]=fout_wrapper(f(vhat[layer-1]))
    Loss = LossFun(vhat[-1], Y)

    # Compute gradient of loss with respect to output
    dLdy=torch.autograd.grad(Loss, vhat[-1])[0]

    return vhat,Loss,dLdy


# Compute prediction errors (epsilon) and beliefs (v)
# using predictive coding algorithm modified by
# the fixed prediction assumption
# see: Millidge, Tschantz, and Buckley. Predictive coding approximates backprop along arbitrary computation graphs.
# v,epsilon=FixedPredPCPredErrs(model,vhat,dLdy,eta=1,n=None)
def FixedPredPCPredErrs(model,vhat,dLdy,eta=1,n=None):

    # Number of layers, counting the input as layer 0 
    DepthPlusOne=len(model)+1 

    if n==None:
      n=len(model)

    # Initialize epsilons
    epsilon=[None]*DepthPlusOne
    epsilon[-1]=dLdy

    # Initialize v to a copy of vhat with no gradients needed
    v=[None]*DepthPlusOne
    for layer in range(DepthPlusOne):
      v[layer]=fout_wrapper(vhat[layer]).clone().detach()
    
    # Iterative updates of v and epsilon using stored values of vhat    
    for i in range(n):
      for layer in reversed(range(DepthPlusOne-1)):  
        epsilon[layer]=vhat[layer]-v[layer]
        # vjp demands a function that returns a single tensor, so we wrap the model call with fout_wrapper
        _,epsdfdv=torch.autograd.functional.vjp(
            lambda inp: fout_wrapper(model[layer](inp)), 
            vhat[layer], 
            epsilon[layer+1]
        )               
        dv=epsilon[layer]-epsdfdv
        v[layer]=v[layer]+eta*dv
      # This helps free up memory
      with torch.no_grad():
        for layer in range(1,DepthPlusOne-1):
          v[layer]=v[layer].clone()
          epsilon[layer]=epsilon[layer].clone()      
    return v,epsilon


# Compute prediction errors (epsilon) and beliefs (v)
# using a strict interpretation of predictive coding
# without the fixed prediction assumption.
# v,epsilon=StrictPCPredErrs(model,vinit,LossFun,Y,eta,n)
def StrictPCPredErrs(model,vinit,LossFun,Y,eta,n):

    with torch.no_grad():
      # Number of layers, counting the input as layer 0 
      DepthPlusOne=len(model)+1 

      # Initialize epsilons
      epsilon=[None]*DepthPlusOne

    # Initialize v to a copy of vinit with no gradients needed
    v=[None]*DepthPlusOne
    for layer in range(DepthPlusOne):
      v[layer]=fout_wrapper(vinit[layer]).clone()    

    # Iterative updates of v and epsilon 
    for i in range(n):
      model.zero_grad()
      layer=DepthPlusOne-1
      vtilde=fout_wrapper(model[layer-1](v[layer-1]))
      Loss=LossFun(vtilde,Y)
      epsilon[layer]=torch.autograd.grad(Loss,vtilde,retain_graph=False)[0]
      for layer in reversed(range(1,DepthPlusOne-1)):
        epsilon[layer]=fout_wrapper(model[layer-1](v[layer-1]))-v[layer]
        # Lambda-Wrapper for vjp, so that the output tuple is unpacked
        _,epsdfdv=torch.autograd.functional.vjp(
            lambda inp: fout_wrapper(model[layer](inp)), 
            v[layer], 
            epsilon[layer+1]
        )               
        dv=epsilon[layer]-epsdfdv
        v[layer]=v[layer]+eta*dv
      # This helps free up memory
      with torch.no_grad():
        for layer in range(1,DepthPlusOne-1):
          v[layer]=v[layer].clone()
          epsilon[layer]=epsilon[layer].clone()
        
    return v,epsilon


# Compute exact prediction errors (epsilon) and beliefs (v)
# epsilon is defined as the gradient of the loss wrt to
# the activations and v=vhat-epsilon where vhat are the
# activations from a forward pass.
# v,epsilon=ExactPredErrs(model,LossFun,X,Y,vhat=None)
def ExactPredErrs(model,LossFun,X,Y,vhat=None):

    # Number of layers, counting the input as layer 0 
    DepthPlusOne=len(model)+1

    # Forward pass if it wasn't passed in
    if vhat==None:
      vhat=[None]*DepthPlusOne
      vhat[0]=fout_wrapper(X)
      for layer in range(1,DepthPlusOne):
        f=model[layer-1]
        vhat[layer]=fout_wrapper(f(vhat[layer-1]))

    Loss = LossFun(vhat[-1], Y)
    
    epsilon=[None]*DepthPlusOne
    v=[None]*DepthPlusOne

    for layer in range(1,DepthPlusOne):          
      epsilon[layer]=torch.autograd.grad(Loss,vhat[layer],allow_unused=True,retain_graph=True)[0]      
      v[layer]=vhat[layer]-epsilon[layer]
    
    return v,epsilon


# Set gradients of model params based on PC approximations
def SetPCGrads(model,epsilon,X,v=None):
    
    # Number of layers, counting the input as layer 0 
    DepthPlusOne=len(model)+1

    # Forward pass if v wasn't passed in
    if v==None:
      v=[None]*DepthPlusOne
      v[0]=fout_wrapper(X)
      for layer in range(1,DepthPlusOne):
        f=model[layer-1]
        v[layer]=fout_wrapper(f(v[layer-1]))

    # Compute new parameter values    
    for layer in range(0,DepthPlusOne-1):
      with torch.no_grad():
        vtemp0=v[layer].clone()
        vtemp0.requires_grad=True
      # unpack the output of the layer to ensure it's a single tensor
      vtemp1=fout_wrapper(model[layer](vtemp0))
      for p in model[layer].parameters():
        if p.requires_grad:
          dtheta=torch.autograd.grad(vtemp1,p,grad_outputs=epsilon[layer+1],allow_unused=True,retain_graph=True)[0]
          p.grad = dtheta      


# Perform a whole PC inference step
# Returns activations (vhat), loss, gradient of the loss wrt output (dLdy), 
# beliefs (v), and prediction errors (epsilon)
# vhat,Loss,dLdy,v,epsilon=PCInfer(model,LossFun,X,Y,ErrType="FixedPred",eta=.1,n=20,vinit=None)
def PCInfer(model,LossFun,X,Y,ErrType,eta=.1,n=20,vinit=None):
  
  # Fwd pass (plus return vhat and dLdy)
  vhat,Loss,dLdy=FwdPassPlus(model,LossFun,X,Y)

  # Get beliefs and prediction errors
  if ErrType=="FixedPred":
    v,epsilon=FixedPredPCPredErrs(model,vhat,dLdy,eta,n)
    SetPCGrads(model,epsilon,X,vhat)
  elif ErrType=="Strict":
    if vinit==None:
      vinit=vhat
    v,epsilon=StrictPCPredErrs(model,vhat,LossFun,Y,eta,n)
    SetPCGrads(model,epsilon,X,v)
  elif ErrType=="Exact":
    v,epsilon=ExactPredErrs(model,LossFun,X,Y)
    SetPCGrads(model,epsilon,X,vhat)
  else:
    raise ValueError('ErrType must be \"FixedPred\", \"Strict\", or \"Exact\"')

  return vhat,Loss,dLdy,v,epsilon
