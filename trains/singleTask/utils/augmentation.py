import torch
import torch.nn as nn
import torch.nn.functional as F

ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

class Unit(nn.Module):
    """
    One MLP layer. It orders the operations as: norm -> fc -> act_fn -> dropout
    """

    def __init__(
            self,
            normalization: str,
            in_features: int,
            out_features: int,
            activation: str,
            dropout_prob: float,
    ):
        """
        Parameters
        ----------
        normalization
            Name of activation function.
        in_features
            Dimension of input features.
        out_features
            Dimension of output features.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        """
        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(in_features)
        else:
            self.norm = None
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = ALL_ACT_LAYERS[activation]()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # pre normalization
        if self.norm is not None:
            x = self.norm(x)
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        # Encoder P(Z|X)
        encoder_layers = []
        dims = [input_dim] + hidden_dim
        for i in range(len(dims) - 1):
            encoder_layers.append(
                Unit(
                    normalization="layer_norm",
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    activation="relu",
                    dropout_prob=0.5,
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)

        self.encoder_fc_z_mu = nn.Linear(self.hidden_dim[-1], self.z_dim)
        self.encoder_fc_z_logvar = nn.Linear(self.hidden_dim[-1], self.z_dim)

        # Decoder P(X|Z)
        decoder_layers = []
        dims = [input_dim] + hidden_dim + [z_dim]

        for i in range(len(dims) - 1, 0, -1):
            decoder_layers.append(
                Unit(
                    normalization="layer_norm",
                    in_features=dims[i],
                    out_features=dims[i - 1],
                    activation="relu",
                    dropout_prob=0.5,
                )
            )
        self.decoder = nn.Sequential(*decoder_layers)

        # self.init_parameters()

    def init_parameters(self):
        self.decoder[-1].fc.weight.data.zero_()
        self.decoder[-1].fc.bias.data.zero_()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)
        z_mu, z_logvar = self.encoder_fc_z_mu(hidden), self.encoder_fc_z_logvar(hidden)
        z = self.reparameterize(z_mu, z_logvar)

        noise_x = self.decoder(z)
        recon_x = x + noise_x
        return recon_x, z_mu, z_logvar


class AugmentNetwork(nn.Module):
    def __init__(self, feature_dim, n_modality, z_dim, arch='mlp_vae', n_layer=4):
        super().__init__()
        # self.config = config
        self.feature_dim = feature_dim
        # self.adapter_out_dim = adapter_out_dim
        d = self.feature_dim * n_modality
        step = int((d - z_dim) / (n_layer + 1))

        if arch == "mlp_vae":
            hidden = [*range(d - step, z_dim + step, -step)]
            self.augnets = VAE(input_dim=d, hidden_dim=hidden, z_dim=z_dim)
        else:
            raise NotImplementedError

        self.name_to_id = self.get_layer_ids()

    def forward(self, x):
        return self.augnets(x)

    def get_layer_ids(
            self,
    ):
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0
        return name_to_id

    def l2_regularize(self, x, x_new):
        return F.mse_loss(x_new, x, reduction="mean")

    def kld(self, m, v):
        return -0.5 * torch.sum(1 + v - m.pow(2) - v.exp())