import torch
from torch.nn.functional import softmax
from .generator.jacobian import Jacobian
from .layercollection import LayerCollection


def FIM_MonteCarlo(model,
                   loader,
                   representation,
                   variant='classif_logits',
                   trials=1,
                   device='cpu',
                   function=None,
                   layer_collection=None):

    if function is None:
        def function(*d):
            return model(d[0].to(device))

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    if variant == 'classif_logits':

        def fim_function(*d):
            log_softmax = torch.log_softmax(function(*d), dim=1)
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, trials,
                                                replacement=True)
            return trials ** -.5 * torch.gather(log_softmax, 1,
                                                sampled_targets)
    elif variant == 'classif_logsoftmax':

        def fim_function(*d):
            log_softmax = function(*d)
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, trials,
                                                replacement=True)
            return trials ** -.5 * torch.gather(log_softmax, 1,
                                                sampled_targets)
    else:
        raise NotImplementedError

    generator = Jacobian(layer_collection=layer_collection,
                         model=model,
                         function=fim_function,
                         n_output=trials)
    return representation(generator=generator, examples=loader)


def FIM(model,
        loader,
        representation,
        n_output,
        variant='classif_logits',
        device='cpu',
        function=None,
        layer_collection=None):
    

    if function is None:
        def function(*d):
            return model(d[0].to(device))

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    if variant == 'classif_logits':

        def function_fim(*d):
            log_probs = torch.log_softmax(function(*d), dim=1)
            probs = torch.exp(log_probs).detach()
            return (log_probs * probs**.5)

    elif variant == 'regression':

        def function_fim(*d):
            estimates = function(*d)
            return estimates
    else:
        raise NotImplementedError

    generator = Jacobian(layer_collection=layer_collection,
                         model=model,
                         function=function_fim,
                         n_output=n_output)
    return representation(generator=generator, examples=loader)
