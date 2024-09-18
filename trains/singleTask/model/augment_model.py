import torch
import torch.nn as nn
from ..utils.augmentation import AugmentNetwork


class AUG(nn.Module):
    def __init__(self, args):
        super(AUG, self).__init__()
        # if args.dataset_name == 'gofundme':
        #     self.meta_dim = 22
        # elif args.dataset_name == 'indiegogo':
        #     self.meta_dim = 41

        self.d_l = args.text_dim
        self.d_v = args.image_dim
        self.meta_dim = args.meta_dim

        self.ae_dim = args.ae_image_dim

        self.modality_dim = args.modality_dim
        self.disentangle_dim = args.disentangle_dim

        self.z_dim = args.vae_dim

        self.cross_static_dim = args.modality_cross_dim
        self.cross_meta_dim = args.meta_cross_dim
        self.static_cross_head = args.modality_cross_head
        self.meta_cross_head = args.meta_cross_head
        self.cross_static_scale = self.cross_static_dim ** -0.5
        self.cross_meta_scale = self.cross_meta_dim ** -0.5

        # 1. Preprocess
        self.ae_v = nn.Sequential(
            nn.Linear(self.d_v, self.ae_dim),
            nn.Linear(self.ae_dim, self.d_v)
        )
        self.compact_image = nn.Sequential(
            nn.Conv1d(self.d_v, self.d_v // 4, kernel_size=1),
            nn.Conv1d(self.d_v // 4, self.modality_dim, kernel_size=1),
        )
        self.fc_l = nn.Sequential(
            nn.Linear(self.d_l, self.modality_dim),
            nn.Dropout(0.1),
        )

        # 2. Augmentation Network
        self.augmenter = AugmentNetwork(self.disentangle_dim, 2, self.z_dim, arch='mlp_vae')

        # 3. Disentanglement
        self.specific_l = nn.Conv1d(self.modality_dim, self.disentangle_dim, kernel_size=1)
        self.specific_v = nn.Conv1d(self.modality_dim, self.disentangle_dim, kernel_size=1)
        self.invariant = nn.Conv1d(self.modality_dim, self.disentangle_dim, kernel_size=1)

        # 4. Cross-Attention for modalities
        self.query_generator_text = nn.Sequential(
            nn.Linear(self.disentangle_dim, self.cross_static_dim),
            nn.ReLU(),
        )
        self.key_generator_text = nn.Sequential(
            nn.Linear(self.disentangle_dim, self.cross_static_dim),
            nn.ReLU(),
        )
        self.value_generator_text = nn.Sequential(
            nn.Linear(self.disentangle_dim, self.cross_static_dim),
            nn.ReLU(),
        )
        self.query_generator_image = nn.Sequential(
            nn.Linear(self.disentangle_dim, self.cross_static_dim),
            nn.ReLU(),
        )
        self.key_generator_image = nn.Sequential(
            nn.Linear(self.disentangle_dim, self.cross_static_dim),
            nn.ReLU(),
        )
        self.value_generator_image = nn.Sequential(
            nn.Linear(self.disentangle_dim, self.cross_static_dim),
            nn.ReLU(),
        )

        # 5. Cross-Attention for metadata
        self.query_generator_meta = nn.Linear(self.meta_dim, self.cross_meta_dim)
        self.key_generator_meta = nn.Linear(self.meta_dim, self.cross_meta_dim)
        self.value_generator_meta = nn.Linear(self.meta_dim, self.cross_meta_dim)
        self.query_generator_modal_s = nn.Linear(2 * self.cross_static_dim + 2 * self.disentangle_dim,
                                                 self.cross_meta_dim)
        self.key_generator_modal_s = nn.Linear(2 * self.cross_static_dim + 2 * self.disentangle_dim,
                                               self.cross_meta_dim)
        self.value_generator_modal_s = nn.Linear(2 * self.cross_static_dim + 2 * self.disentangle_dim,
                                                 self.cross_meta_dim)

        # 6. Prediction Layer
        self.layer_out_norm = nn.LayerNorm(2 * self.cross_meta_dim, eps=1e-6)
        self.cross_out_layer = nn.Sequential(
            nn.Linear(2 * self.cross_meta_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Reconstruction loss
        self.text_recon = nn.Conv1d(2 * self.disentangle_dim, self.modality_dim, kernel_size=1)
        self.image_recon = nn.Conv1d(2 * self.disentangle_dim, self.modality_dim, kernel_size=1)

    def text_cross_image(self, text, image):
        batch_size = text.size(0)
        text_q = self.query_generator_text(text).view(batch_size, self.static_cross_head, -1)
        text_k = self.key_generator_text(text).view(batch_size, self.static_cross_head, -1)
        text_v = self.value_generator_text(text).view(batch_size, self.static_cross_head, -1)
        image_q = self.query_generator_image(image).view(batch_size, self.static_cross_head, -1)
        image_k = self.key_generator_image(image).view(batch_size, self.static_cross_head, -1)
        image_v = self.value_generator_image(image).view(batch_size, self.static_cross_head, -1)

        out_text = (((text_q * image_k) * self.cross_static_scale).softmax(dim=-1)) * image_v
        out_image = (((image_q * text_k) * self.cross_static_scale).softmax(dim=-1)) * text_v

        out_text = out_text.contiguous().view(batch_size, -1)
        out_image = out_image.contiguous().view(batch_size, -1)

        out = torch.cat((out_text, out_image), dim=-1)

        return out

    def meta_static_cross(self, meta, static):
        batch_size = meta.size(0)
        meta_q = self.query_generator_meta(meta).view(batch_size, self.meta_cross_head, -1)
        meta_k = self.key_generator_meta(meta).view(batch_size, self.meta_cross_head, -1)
        meta_v = self.value_generator_meta(meta).view(batch_size, self.meta_cross_head, -1)
        static_q = self.query_generator_modal_s(static).view(batch_size, self.meta_cross_head, -1)
        static_k = self.key_generator_modal_s(static).view(batch_size, self.meta_cross_head, -1)
        static_v = self.value_generator_modal_s(static).view(batch_size, self.meta_cross_head, -1)

        out_m = (((meta_q * static_k) * self.cross_meta_scale).softmax(dim=-1)) * static_v
        out_s = (((static_q * meta_k) * self.cross_meta_scale).softmax(dim=-1)) * meta_v

        out_m = out_m.contiguous().view(batch_size, -1)
        out_s = out_s.contiguous().view(batch_size, -1)

        out = torch.cat((out_m, out_s), dim=-1)

        return out

    def invariant_augment(self, meta, text_specific, image_specific, text_invariant, image_invariant):
        specific = self.text_cross_image(text_specific, image_specific)
        invariant = torch.cat((text_invariant, image_invariant), dim=-1)

        static = torch.cat((specific, invariant), dim=-1)

        out = self.meta_static_cross(meta, static)

        normed_out = self.layer_out_norm(out)
        logits = self.cross_out_layer(normed_out)

        return logits

    def forward(self, meta, text, image, beta_1):
        """
        meta: B * 1 * D_meta
        text: B * 1 * D_text
        image: B * 1 * D_image
        """
        image = self.ae_v(image)
        image = self.compact_image(image.transpose(1, 2)).squeeze()
        text = self.fc_l(text).squeeze()

        text = text.unsqueeze(-1)
        image = image.unsqueeze(-1)
        text_specific = self.specific_l(text).squeeze(-1)
        image_specific = self.specific_v(image).squeeze(-1)
        text_invariant = self.invariant(text).squeeze(-1)
        image_invariant = self.invariant(image).squeeze(-1)

        if self.training:
            detached_feature = torch.cat((text_invariant, image_invariant), dim=1).detach().clone()
            after_augment, m, v = self.augmenter(detached_feature)
            regularize_loss = self.augmenter.l2_regularize(detached_feature, after_augment)
            vae_loss = (
                    self.augmenter.kld(m, v) / after_augment.size()[0] / self.z_dim
            )

            after_augment_text, after_augment_image = after_augment[:, :self.disentangle_dim], after_augment[:,
                                                                                               self.disentangle_dim:]

            after_augment_text.register_hook(
                lambda grad: grad * -beta_1
            )
            after_augment_image.register_hook(
                lambda grad: grad * -beta_1
            )
            aug_logits = self.invariant_augment(meta, text_specific, image_specific, after_augment_text,
                                                after_augment_image)
        else:
            aug_logits = 0
            regularize_loss = 0
            vae_loss = 0

        recon_text = self.text_recon(torch.cat((text_specific, text_invariant), dim=1).unsqueeze(-1)).squeeze(-1)
        recon_image = self.image_recon(torch.cat((image_specific, image_invariant), dim=1).unsqueeze(-1)).squeeze(-1)

        ori_logits = self.invariant_augment(meta, text_specific, image_specific, text_invariant, image_invariant)

        output = {
            'ori_logits': ori_logits,
            'aug_logits': aug_logits,
            'regularize_loss': regularize_loss,
            'vae_loss': vae_loss,
            'text': text.squeeze(-1),
            'image': image.squeeze(-1),
            'specific_text': text_specific,
            'specific_image': image_specific,
            'invariant_text': text_invariant,
            'invariant_image': image_invariant,
            'recon_text': recon_text,
            'recon_image': recon_image,
        }

        return output
