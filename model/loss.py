import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PortaSpeechLoss(nn.Module):
    """ PortaSpeech Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(PortaSpeechLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.sum_loss = ForwardSumLoss()

    def kl_loss(self, z_p, logs_q, mask):
        """
        z_p, logs_q: [batch_size, dim, max_time]
        mask -- [batch_size, 1, max_time]
        """
        m_p, logs_p = torch.zeros_like(z_p), torch.zeros_like(z_p)
        z_p = z_p.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        logs_q = logs_q.float()
        mask = mask.float()

        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
        kl = torch.sum(kl * mask)
        l = kl / torch.sum(mask)
        return l

    def mle_loss(self, z, logdet, mask):
        """
        z, logdet: [batch_size, dim, max_time]
        mask -- [batch_size, 1, max_time]
        """
        logs = torch.zeros_like(z * mask)
        l = torch.sum(logs) + 0.5 * \
            torch.sum(torch.exp(-2 * logs) * (z**2))
        l = l - torch.sum(logdet)
        l = l / \
            torch.sum(torch.ones_like(z * mask))
        l = l + 0.5 * math.log(2 * math.pi)
        return l

    def forward(self, inputs, predictions):
        (
            mel_targets,
            *_,
        ) = inputs[11:]
        (
            mel_predictions,
            postnet_outputs,
            log_duration_predictions,
            duration_roundeds,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            _,
            dist_info,
            src_w_masks,
            _,
            alignment_logprobs,
        ) = predictions
        log_duration_targets = torch.log(duration_roundeds.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        mel_targets.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(
            src_w_masks)
        log_duration_targets = log_duration_targets.masked_select(src_w_masks)

        mel_predictions = mel_predictions.masked_select(
            mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)

        duration_loss = self.mse_loss(
            log_duration_predictions, log_duration_targets)

        kl_loss = self.kl_loss(*dist_info)

        z, logdet = postnet_outputs
        postnet_loss = self.mle_loss(z, logdet, mel_masks.unsqueeze(1))

        ctc_loss = torch.zeros(1).to(mel_targets.device)
        for alignment_logprob in alignment_logprobs:
            ctc_loss += self.sum_loss(alignment_logprob, src_lens, mel_lens)
        ctc_loss = ctc_loss.mean()

        total_loss = (
            mel_loss + kl_loss + postnet_loss + duration_loss + ctc_loss
        )

        return (
            total_loss,
            mel_loss,  # L_VG
            kl_loss,  # L_KL
            postnet_loss,  # L_PN
            duration_loss,  # L_dur
            ctc_loss,
        )


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss
