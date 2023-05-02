import torch
from torch import nn


class ValenceArousal(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 1290
        self.hidden = nn.Linear(in_features=self.in_features, out_features=self.in_features)
        self.hidden_activation = nn.LeakyReLU()
        self.hidden_batchnorm = nn.BatchNorm1d(num_features=self.in_features)
        self.hidden_dropout = nn.Dropout(p=0.55)

        self.valence_head = nn.Linear(in_features=self.in_features, out_features=1)
        self.arousal_head = nn.Linear(in_features=self.in_features, out_features=1)

        self.head_activation = nn.Tanh()

    def forward(self, extracted_features):
        output = self.hidden(extracted_features)
        output = self.hidden_batchnorm(output)
        output = self.hidden_activation(output)
        output = self.hidden_dropout(output)

        valence_output = self.head_activation(self.valence_head(output))
        arousal_output = self.head_activation(self.arousal_head(output))

        return valence_output, arousal_output


class Expression(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 1290
        self.hidden = nn.Linear(in_features=self.in_features, out_features=self.in_features)
        self.hidden_activation = nn.LeakyReLU()
        self.hidden_batchnorm = nn.BatchNorm1d(num_features=self.in_features)
        self.hidden_dropout = nn.Dropout(p=0.4)

        self.expression_head = nn.Linear(in_features=self.in_features, out_features=8)

        self.head_activation = nn.Softmax(dim=1)

    def forward(self, extracted_features):
        output = self.hidden(extracted_features)
        output = self.hidden_batchnorm(output)
        output = self.hidden_activation(output)
        output = self.hidden_dropout(output)

        expression_output = self.head_activation(self.expression_head(output))

        return expression_output


class ActionUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 1290
        self.hidden = nn.Linear(in_features=self.in_features, out_features=self.in_features)
        self.hidden_activation = nn.LeakyReLU()
        self.hidden_batchnorm = nn.BatchNorm1d(num_features=self.in_features)
        self.hidden_dropout = nn.Dropout(p=0.6)

        self.action_unit_head = nn.Linear(in_features=self.in_features, out_features=12)

        self.head_activation = nn.Sigmoid()

    def forward(self, extracted_features):
        output = self.hidden(extracted_features)
        output = self.hidden_batchnorm(output)
        output = self.hidden_activation(output)
        output = self.hidden_dropout(output)

        action_unit_output = self.head_activation(self.action_unit_head(output))

        return action_unit_output


class Ensemble(nn.Module):
    def __init__(self, model_va, model_ex, model_au):
        super().__init__()
        self.model_va = model_va
        self.model_ex = model_ex
        self.model_au = model_au

        self.model_va.eval()
        self.model_ex.eval()
        self.model_au.eval()

    def forward(self, x):
        with torch.no_grad():
            va_output, ar_output = self.model_va(x)
            ex_output = self.model_ex(x)
            au_output = self.model_au(x)

        return va_output, ar_output, ex_output, au_output

