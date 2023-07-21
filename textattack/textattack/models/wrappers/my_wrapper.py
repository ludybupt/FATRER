"""
ERC Model Wrapper
--------------------------
"""

import torch
import transformers

import textattack
from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer

from .pytorch_model_wrapper import PyTorchModelWrapper

torch.cuda.empty_cache()


class ERCModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer):
        '''
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."
        '''
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 300
        
    def batch_to_device(self, batch, target_device):
        """
        send a batch to a device
        Originally the tensorize was done in the smart batch,
        We did this in the dataset_wrapper

        :param batch: column * batchsize * other, column is a list
        :param target_device:
        :return: the batch sent to the device
        """
        # features = batch['features']
        # labels = batch['labels'].to(target_device)
        # return features, labels
        # return features, labels
        # for i in range(len(batch)):
        #     batch[i].to(target_device)

        # better not use batch[i] = batch[i].to(target_device)
        # this will left reference to cudas preventing the deletion
        # for i in range(len(batch)):
        #     batch[i] = batch[i].to(target_device)
        features = []
        for i in range(len(batch)):
            features.append(batch[i].to(target_device))
        return features
    
    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
                               
        inputs_dict = self.move_to(inputs_dict, model_device)
        
        outputs = None
        if isinstance(inputs_dict, list):
            for part in inputs_dict:
                with torch.no_grad():
                    tmp = (self.model(**part)).logits
                    tmp = tmp.squeeze(0)
                    if not torch.is_tensor(outputs):
                        outputs = tmp
                    else:
                        outputs = torch.cat((outputs, tmp), 0)

        return outputs
        '''
        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits
        '''

    def move_to(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                #res[k] = v.to(device) #self.move_to(v, device)
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        return obj.to(device)

            
    def data_to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            data_device = dict()
            for key, val in data.items():
                data_device[key] = val.to(device)
            return data_device
        elif isinstance(data, list):
            data_device = []
            for val in data:
                data_device.append(val.to(device))
            return data_device
        else:
            raise Exception("Not implement")
        
    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
