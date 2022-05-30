import torch.nn as nn


# TODO: Replace with this classifier instead of a FFN and check the difference
class Classifier(nn.Module):
    def __init__(self, input_dim):
        """Classifier used for the predictions
        Inputs:
            input_dim - Dimension of the input. Default is 4*300 (AWE embedding)
        """
        super().__init__()

        # initialize the classifier
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 3)
        )

    def forward(self, sentence_embeddings):
        """
        Inputs:
            sentence_embeddings - Tensor of sentence representations of shape varying shape. AWE is [B, 4*300]
        Outputs:
            predictions - Tensor of predictions (entailment, neutral, contradiction) of shape [B, 3]
        """
        print("forward in classifier")
        print(sentence_embeddings.shape)
        # pass sentence embeddings through model
        predictions = self.net(sentence_embeddings)

        # return the  predictions
        return predictions
