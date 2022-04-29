import torch
import torch.nn as nn


class AdaSpeechLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(AdaSpeechLoss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions, phoneme_level_loss):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            avg_mel_phs
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            avg_mel_ph_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            avg_mel_phs_encode
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        if phoneme_level_loss:
            avg_mel_ph_predictions = avg_mel_ph_predictions.masked_select(src_masks.unsqueeze(-1))
            avg_mel_phs_encode = avg_mel_phs_encode.masked_select(src_masks.unsqueeze(-1))
            avg_ph_mel_loss = self.mse_loss(avg_mel_phs_encode, avg_mel_ph_predictions)


            total_loss = (
                mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + avg_ph_mel_loss
            )
        else:
            avg_mel_ph_predictions = avg_mel_ph_predictions.masked_select(src_masks.unsqueeze(-1))
            avg_mel_phs_encode = avg_mel_phs_encode.masked_select(src_masks.unsqueeze(-1))
            avg_ph_mel_loss = self.mse_loss(avg_mel_phs_encode, avg_mel_ph_predictions)

            total_loss = (
                mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
            )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            avg_ph_mel_loss,
        )
